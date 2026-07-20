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


def estimate_offset(a: np.ndarray, b: np.ndarray, max_lag: int = 15) -> OffsetEstimate:
    a, b = zscore(np.asarray(a, float)), zscore(np.asarray(b, float))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    full = correlate(a, b, mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")
    keep = np.abs(lags) <= max_lag
    corr, lags = full[keep], lags[keep]
    corr_norm = corr / n                       # ~Pearson at each lag (z-scored inputs)
    k = int(np.argmax(corr_norm))
    peak = corr_norm[k]

    # sub-frame parabolic interpolation around the integer peak
    off = float(lags[k])
    if 0 < k < len(corr_norm) - 1:
        y0, y1, y2 = corr_norm[k - 1], corr_norm[k], corr_norm[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            off += 0.5 * (y0 - y2) / denom

    # confidence: how far the peak stands above the rest of the correlation curve
    others = np.delete(corr_norm, k)
    bg_mu, bg_sd = others.mean(), others.std() + 1e-9
    snr = (peak - bg_mu) / bg_sd
    confidence = float(np.clip(snr / 8.0, 0.0, 1.0)) * float(np.clip(peak, 0.0, 1.0))
    return OffsetEstimate(offset=off, offset_int=int(lags[k]),
                          confidence=confidence, peak_corr=float(peak))


def vision_joint_offset(ep, method: str = "auto", max_lag: int = 15) -> OffsetEstimate:
    """Camera-vs-proprioception offset. Positive => camera lags joints.
    Defaults to optical flow (sub-frame precision) when opencv is installed."""
    return estimate_offset(vision_motion(ep.rgb, method), joint_motion(ep.joints), max_lag)


def action_state_offset(ep, max_lag: int = 15) -> OffsetEstimate:
    """Commanded-action vs resulting joint-motion offset (catches off-by-k pairing)."""
    return estimate_offset(action_motion(ep.actions), joint_motion(ep.joints), max_lag)
