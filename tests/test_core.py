"""One runnable check for the core claim: injected offset is recovered within
+/-1 frame at the dataset level, and injectors round-trip. Skips if the LIBERO
cache file isn't present (download it via the README command)."""
import glob, os
import numpy as np
import pytest

from roboinfer.data import load_libero
from roboinfer import inject
from roboinfer.estimate import vision_joint_offset
from roboinfer.signals import vision_motion, joint_motion


def _libero_file():
    repo = os.path.join(os.path.dirname(__file__), "..", "data", "libero", "*.hdf5")
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/"
        "snapshots/*/libero_10/STUDY_SCENE1*demo.hdf5")
    hits = glob.glob(repo) or glob.glob(cache)     # repo-local data first
    return hits[0] if hits else None


def test_injectors_roundtrip():
    """Pure-numpy check, no data file needed."""
    from roboinfer.data import Episode
    T = 100
    rng = np.random.default_rng(0)
    ep = Episode(rgb=rng.integers(0, 255, (T, 8, 8, 3), np.uint8),
                 joints=rng.standard_normal((T, 7)), actions=rng.standard_normal((T, 7)),
                 ts=np.arange(T) / 20.0, name="synthetic")
    assert inject.inject_offset(ep, 5)[0].T == T - 5
    assert inject.inject_frozen(ep, 10, 20)[0].T == T
    fro = inject.inject_frozen(ep, 10, 20)[0]
    assert np.array_equal(fro.rgb[10], fro.rgb[25])           # frozen span identical
    sw = inject.inject_swap(ep, 1, 3)[0]
    assert np.array_equal(sw.joints[:, 1], ep.joints[:, 3])   # columns swapped


def test_confidence_separates_signal_from_noise():
    """Permutation-null confidence: a real lagged signal reads verifiable; two
    independent signals read ~unverifiable."""
    from roboinfer.estimate import estimate_offset
    rng = np.random.default_rng(1)
    latent = rng.standard_normal(400)                # broadband, sharply localizable lag
    a = latent + 0.05 * rng.standard_normal(400)
    b = np.r_[np.zeros(4), latent[:-4]] + 0.05 * rng.standard_normal(400)   # delayed 4
    est = estimate_offset(a, b, rng=np.random.default_rng(0))
    assert abs(abs(est.offset) - 4) <= 1.0           # recovers the lag magnitude
    assert est.confidence >= 0.3                     # and trusts it
    noise = estimate_offset(a, np.cumsum(rng.standard_normal(400)),
                            rng=np.random.default_rng(0))
    assert noise.confidence < est.confidence         # unrelated => less trusted


def test_audit_abstains_on_low_fps():
    """Audit must not cry wolf: too-coarse fps => ABSTAIN, no flags."""
    from roboinfer.data import Episode
    from roboinfer.detect import dataset_audit
    rng = np.random.default_rng(0)
    eps = [Episode(rgb=rng.integers(0, 255, (80, 8, 8, 3), np.uint8),
                   joints=rng.standard_normal((80, 7)), actions=rng.standard_normal((80, 7)),
                   ts=np.arange(80) / 3.0, name=f"e{i}", hz=3.0) for i in range(6)]
    r = dataset_audit(eps)
    assert r["verdict"].startswith("ABSTAIN") and "anom_ct" not in r


def test_ts_jitter_score():
    """Uniform clock scores 1.0 (never flags); a dropped frame spikes the ratio."""
    from roboinfer.data import Episode
    from roboinfer.detect import ts_jitter_score, TS_JITTER_K
    T = 50
    z = np.zeros((T, 4, 4, 3), np.uint8)
    j = np.zeros((T, 7)); a = np.zeros((T, 7))
    uniform = Episode(rgb=z, joints=j, actions=a, ts=np.arange(T) / 20.0, name="u")
    assert abs(ts_jitter_score(uniform) - 1.0) < 1e-6        # regular clock: no flag
    ts = np.arange(T) / 20.0
    ts[25:] += 0.15                                          # a ~3-frame gap (dropped frames)
    dropped = Episode(rgb=z, joints=j, actions=a, ts=ts, name="d")
    assert ts_jitter_score(dropped) >= TS_JITTER_K           # flagged


@pytest.mark.skipif(_libero_file() is None, reason="LIBERO cache missing")
def test_offset_recovery_within_one_frame():
    eps = list(load_libero(_libero_file(), limit=40))   # a task's demos = product's pool unit
    base = np.median([vision_joint_offset(e).offset for e in eps])
    for k in (-4, -2, 0, 3, 6):
        est = np.median([vision_joint_offset(inject.inject_offset(e, k)[0]).offset
                         for e in eps])
        assert abs((est - base) - k) <= 1.0, f"offset {k}: recovered {est-base:.2f}"


if __name__ == "__main__":
    test_injectors_roundtrip()
    f = _libero_file()
    if f:
        test_offset_recovery_within_one_frame()
        print("PASS: injectors + offset recovery within 1 frame")
    else:
        print("PASS: injectors (offset test skipped, no LIBERO cache)")
