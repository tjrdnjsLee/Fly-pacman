"""PPO 신경망 · 갱신 · 버퍼 · RND. pacman-ai(MIT, NOTICE)의 `agents/networks.py`, `agents/ppo.py`,
`agents/rollout.py`, `agents/rnd.py`, `training/trainer.py`(RunningMeanStd)를 옮겼다.

원 저장소와 다른 곳:
- 숫자 값은 설정 파일에서 받는다 (Adam eps, 보상 정규화 자르기, RND 표본 수).
- 섞기 · 표본 뽑기는 넘파이 전역 난수 대신 넘겨받은 Generator를 쓴다 (체크포인트 재개 뒤 같은 결과).
- 행동 번호는 절대 방향(시계방향, core/directions.py)이다. 원 저장소는 상 · 하 · 좌 · 우였다.
- 버퍼의 관측 격자는 uint8로 담고(값이 0 · 1뿐이라 정확히 같다) 갱신마다 장치로 한 번만 옮긴다 (속도).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """CNN 공유 몸통 + 정책 · 가치 머리."""

    def __init__(self, grid_channels: int, grid_h: int, grid_w: int, num_scalars: int, num_actions: int,
                 cnn_channels, cnn_kernels, cnn_strides, shared_hidden: int, head_hidden: int):
        super().__init__()
        layers, in_ch = [], grid_channels
        for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers += [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2), nn.ReLU()]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, grid_channels, grid_h, grid_w)).view(1, -1).shape[1]
        self.shared = nn.Sequential(nn.Linear(cnn_out + num_scalars, shared_hidden), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(shared_hidden, head_hidden), nn.ReLU(),
                                    nn.Linear(head_hidden, num_actions))
        self.value = nn.Sequential(nn.Linear(shared_hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(self, grid, scalars, legal_mask):
        h = self.shared(torch.cat([self.cnn(grid).flatten(1), scalars], dim=1))
        logits = self.policy(h).masked_fill(~legal_mask, -1e8)
        return logits, self.value(h)


def build_network(ncfg, grid_shape: tuple[int, int, int], num_scalars: int) -> ActorCritic:
    C, H, W = grid_shape
    return ActorCritic(C, H, W, num_scalars, 4, list(ncfg["cnn_channels"]), list(ncfg["cnn_kernels"]),
                       list(ncfg["cnn_strides"]), int(ncfg["shared_hidden"]), int(ncfg["head_hidden"]))


class RunningMeanStd:
    """보상 정규화용 이동 평균 · 분산."""

    def __init__(self, clip: float):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(x.mean()), float(x.var()), x.shape[0]
        delta = bm - self.mean
        total = self.count + bc
        self.mean += delta * bc / total
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class RolloutBuffer:
    def __init__(self, num_envs: int, rollout_steps: int, grid_shape, num_scalars: int, num_actions: int = 4):
        T, N = rollout_steps, num_envs
        self.num_envs, self.rollout_steps = N, T
        self.obs_grids = np.zeros((T, N, *grid_shape), dtype=np.uint8)  # 0 · 1 격자
        self.obs_scalars = np.zeros((T, N, num_scalars), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.legal_masks = np.zeros((T, N, num_actions), dtype=bool)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def insert(self, grid, scalars, action, log_prob, value, reward, done, legal) -> None:
        t = self._step
        self.obs_grids[t], self.obs_scalars[t] = grid, scalars
        self.actions[t], self.log_probs[t], self.values[t] = action, log_prob, value
        self.rewards[t], self.dones[t], self.legal_masks[t] = reward, done, legal
        self._step += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, lam: float) -> None:
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_value = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            nonterm = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_value * nonterm - self.values[t]
            gae = delta + gamma * lam * nonterm * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def batch_generator(self, minibatch_size: int, device, rng: np.random.Generator):
        total = self.rollout_steps * self.num_envs
        idx_all = torch.as_tensor(rng.permutation(total), device=device)
        flat = {
            "obs_grid": self.obs_grids.reshape(total, *self.obs_grids.shape[2:]),
            "obs_scalars": self.obs_scalars.reshape(total, -1),
            "actions": self.actions.reshape(total),
            "old_log_probs": self.log_probs.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "legal_masks": self.legal_masks.reshape(total, -1),
            "old_values": self.values.reshape(total),
        }
        on_dev = {k: torch.as_tensor(v, device=device) for k, v in flat.items()}  # 갱신마다 한 번만 옮긴다
        for start in range(0, total, minibatch_size):
            idx = idx_all[start:start + minibatch_size]
            batch = {k: v[idx] for k, v in on_dev.items()}
            batch["obs_grid"] = batch["obs_grid"].float()
            yield batch


def grid_tensor(grid, device) -> torch.Tensor:
    """uint8 관측 격자 → 장치의 float32 텐서."""
    return torch.as_tensor(grid, device=device).float()


class PPO:
    def __init__(self, network: ActorCritic, pcfg, device: torch.device):
        self.network = network.to(device)
        self.device = device
        self.cfg = pcfg
        self.optimizer = torch.optim.Adam(network.parameters(), lr=float(pcfg["learning_rate"]),
                                          eps=float(pcfg["adam_eps"]))
        self.entropy_coef = float(pcfg["entropy_coef_start"])

    def _t(self, x):
        return torch.as_tensor(x, device=self.device)

    @torch.no_grad()
    def select_action(self, grid, scalars, legal):
        logits, values = self.network(grid_tensor(grid, self.device), self._t(scalars), self._t(legal))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), values.squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def get_value(self, grid, scalars, legal) -> np.ndarray:
        _, values = self.network(grid_tensor(grid, self.device), self._t(scalars), self._t(legal))
        return values.squeeze(-1).cpu().numpy()

    def update(self, rollout: RolloutBuffer, rng: np.random.Generator) -> dict:
        c = self.cfg
        clip = float(c["clip_epsilon"])
        adv = rollout.advantages.reshape(-1)
        rollout.advantages = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(rollout.advantages.shape)
        tot_p = tot_v = tot_e = 0.0
        n = 0
        for _ in range(int(c["num_epochs"])):
            for b in rollout.batch_generator(int(c["minibatch_size"]), self.device, rng):
                logits, values = self.network(b["obs_grid"], b["obs_scalars"], b["legal_masks"])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b["actions"])
                entropy = dist.entropy().mean()
                ratio = torch.exp(logp - b["old_log_probs"])
                A = b["advantages"]
                policy_loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A).mean()
                v = values.squeeze(-1)
                v_clip = b["old_values"] + torch.clamp(v - b["old_values"], -clip, clip)
                value_loss = 0.5 * torch.max((v - b["returns"]) ** 2, (v_clip - b["returns"]) ** 2).mean()
                loss = policy_loss + float(c["value_loss_coef"]) * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), float(c["max_grad_norm"]))
                self.optimizer.step()
                tot_p += policy_loss.item()
                tot_v += value_loss.item()
                tot_e += entropy.item()
                n += 1
        n = max(n, 1)
        return {"policy_loss": tot_p / n, "value_loss": tot_v / n, "entropy": tot_e / n}

    def anneal_lr(self, update: int, total: int) -> None:
        if not bool(self.cfg["lr_anneal"]):
            return
        lr = float(self.cfg["learning_rate"]) * (1.0 - update / total)
        for g in self.optimizer.param_groups:
            g["lr"] = lr


class RNDModule(nn.Module):
    """호기심 보상 (Random Network Distillation). 고정된 무작위 망을 예측하는 오차가 보상이다."""

    def __init__(self, input_channels: int, grid_h: int, grid_w: int, hidden_size: int, output_size: int,
                 learning_rate: float):
        super().__init__()

        def cnn():
            return nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1), nn.LeakyReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            flat = cnn()(torch.zeros(1, input_channels, grid_h, grid_w)).shape[1]
        self.target = nn.Sequential(cnn(), nn.Linear(flat, output_size))
        self.predictor = nn.Sequential(cnn(), nn.Linear(flat, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, output_size))
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    @torch.no_grad()
    def intrinsic_reward(self, grid: torch.Tensor) -> torch.Tensor:
        return ((self.target(grid) - self.predictor(grid)) ** 2).mean(dim=1)

    def update(self, grid: torch.Tensor) -> float:
        with torch.no_grad():
            t = self.target(grid)
        loss = ((t - self.predictor(grid)) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
