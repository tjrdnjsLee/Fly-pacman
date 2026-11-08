import numpy as np
import pytest

from flypac.replay import schema as s


def _arrays(T=5, N=7, E=2, K=3, M=6, brain=False, phero=False):
    a = {}
    for f in s.FIELDS:
        if f.group == "brain" and not brain:
            continue
        if f.group == "phero" and not phero:
            continue
        dims = {"T": T, "N": N, "E": E, "K": K, "M": M}
        shape = tuple(dims[x] if isinstance(x, str) else x for x in f.shape)
        a[f.name] = np.zeros(shape, dtype=f.dtype)
    a[s.META_KEY] = np.array(s.make_meta("regime1", 0, 1, "abc", "h", "original"))
    return a


def test_valid_base_and_brain_and_phero():
    assert s.validate(_arrays())["format_version"] == s.FORMAT_VERSION
    s.validate(_arrays(brain=True, phero=True))


def test_missing_required_field():
    a = _arrays()
    del a["odor_field"]
    with pytest.raises(s.SchemaError):
        s.validate(a)


def test_brain_group_all_or_nothing():
    a = _arrays(brain=True)
    del a["dan"]
    with pytest.raises(s.SchemaError):
        s.validate(a)


def test_wrong_dtype_shape_and_dims():
    a = _arrays()
    a["pac_cell"] = a["pac_cell"].astype("int32")
    with pytest.raises(s.SchemaError):
        s.validate(a)
    a = _arrays()
    a["fly_cell"] = np.zeros((5, 3), "int16")
    with pytest.raises(s.SchemaError):
        s.validate(a)
    a = _arrays()
    a["pac_dir"] = np.zeros(6, "int8")  # T 불일치
    with pytest.raises(s.SchemaError):
        s.validate(a)


def test_version_and_unknown_field():
    a = _arrays()
    a["junk"] = np.zeros(1)
    with pytest.raises(s.SchemaError):
        s.validate(a)
    a = _arrays()
    meta = s.make_meta("x", 0, 0, "", "", "")
    assert f'"format_version": {s.FORMAT_VERSION}' in meta
    a[s.META_KEY] = np.array(meta.replace(f'"format_version": {s.FORMAT_VERSION}', '"format_version": 1'))
    with pytest.raises(s.SchemaError):
        s.validate(a)


def test_kind_codes_cover_all_kinds_and_are_stable():
    from flypac.core.events import Kind

    assert set(s.KIND_CODES) == {k.value for k in Kind}
    assert sorted(s.KIND_CODES.values()) == list(range(len(Kind)))
    assert s.KIND_CODES["CATCH"] == 0 and s.KIND_CODES["ALL_RELEASED"] == 10
    assert all(v < 127 for v in s.KIND_CODES.values())  # int8
