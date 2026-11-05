"""YAML 설정 불러오기 (TRD 원칙 3: 숫자는 설정 파일에만)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import yaml


class Config(Mapping):
    """중첩 사전을 속성으로도 읽게 하는 읽기 전용 감싸개. cfg.odor.D 와 cfg["odor"]["D"] 모두 가능."""

    def __init__(self, data: Mapping):
        self._data = dict(data)

    def __getitem__(self, key):
        v = self._data[key]
        return Config(v) if isinstance(v, Mapping) else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))

    def hash(self) -> str:
        """설정 해시 (재생 로그 메타데이터용)."""
        blob = json.dumps(self._data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"설정 파일의 최상위는 사전이어야 한다: {path}")
    return Config(data)
