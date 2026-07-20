"""Temporal offset estimation by cross-correlation of two 1-D motion signals.

offset = lag (in frames) you must shift signal `a` to align it onto `b`.
Positive offset => `a` lags `b` (a's motion happens `offset` frames later).
Confidence = normalized peak height vs the correlation background (SNR-ish).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate, correlation_lags
from .signals import zscore, vision_motion, joint_motion, action_motion


@dataclass
class OffsetEstimate:
    offset: float        # sub-frame lag, frames (a relative to b)
    offset_int: int      # integer peak lag
    confidence: float    # 0..1, peak prominence over background
    peak_corr: float     # normalized correlation at peak (-1..1)


def _corr_peak(a, b, max_lag):
    """Sub-frame peak lag + normalized peak correlation of a vs b (z-scored inputs)."""
    n = len(a)
    full = correlate(a, b, mode="full")
    lags = correlation_lags(n, len(b), mode="full")
    keep = np.abs(lags) <= max_lag
    corr, lags = full[keep] / n, lags[keep]     # ~Pearson at each lag
    k = int(np.argmax(corr))
    peak = float(corr[k])
    off = float(lags[k])
    if 0 < k < len(corr) - 1:                    # parabolic sub-frame interpolation
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            off += 0.5 * (y0 - y2) / denom
    return off, int(lags[k]), peak


def _phase_randomize(x, rng):
    """Surrogate with x's power spectrum but random phases: destroys any real
    cross-alignment with b while preserving autocorrelation structure."""
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0                                  # keep DC real
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


# z (sigmas above the phase-randomized null) at which we call an estimate fully
# trustworthy. Statistical significance scale, not a magic corpus constant — the
# null is rebuilt per episode from the signal's own spectrum. Tuned on real data
# (Day 6): ALOHA good estimates clear it, DROID unestimable ones stay near 0.
CONF_Z_FULL = 6.0


def estimate_offset(a: np.ndarray, b: np.ndarray, max_lag: int = 15,
                    n_null: int = 48, rng=None) -> OffsetEstimate:
    a, b = zscore(np.asarray(a, float)), zscore(np.asarray(b, float))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    off, off_int, peak = _corr_peak(a, b, max_lag)

    # confidence from a per-episode permutation null: how far the real peak stands
    # above peaks of phase-randomized surrogates of `a` correlated against `b`.
    rng = np.random.default_rng(0) if rng is None else rng
    null = np.array([_corr_peak(_phase_randomize(a, rng), b, max_lag)[2]
                     for _ in range(n_null)])
    mu, sd = float(null.mean()), float(null.std() + 1e-9)
    z = (peak - mu) / sd
    confidence = float(np.clip(z / CONF_Z_FULL, 0.0, 1.0))
    return OffsetEstimate(offset=off, offset_int=off_int,
                          confidence=confidence, peak_corr=peak)


def vision_joint_offset(ep, method: str = "auto", max_lag: int = 15) -> OffsetEstimate:
    """Camera-vs-proprioception offset. Positive => camera lags joints.
    Defaults to optical flow (sub-frame precision) when opencv is installed."""
    return estimate_offset(vision_motion(ep.rgb, method), joint_motion(ep.joints), max_lag)


def action_state_offset(ep, max_lag: int = 15) -> OffsetEstimate:
    """Commanded-action vs resulting joint-motion offset (catches off-by-k pairing)."""
    return estimate_offset(action_motion(ep.actions), joint_motion(ep.joints), max_lag)
