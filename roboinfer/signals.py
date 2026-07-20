"""Per-frame 1-D motion signals. Everything downstream cross-correlates these.

Vision motion default = mean abs frame difference (zero-dep, cheapest signal
that spikes on motion). Optical-flow magnitude (opencv Farneback) is opt-in via
method="flow" and only worth it if frame-diff misses the +/-1 frame target.
"""
from __future__ import annotations
import numpy as np


def _pad0(x: np.ndarray) -> np.ndarray:
    """Prepend one zero so a diff-based signal keeps length T (aligned to frame t)."""
    return np.concatenate([[0.0], x])


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def vision_motion(rgb: np.ndarray, method: str = "diff") -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,) motion energy. Signal at frame t = change t-1 -> t.

    method: "diff" (fast, ~1-frame offset precision) | "flow" (optical-flow
    magnitude, sub-frame precision, needs opencv) | "auto" (flow if available).
    """
    if method == "auto":
        method = "flow" if _have_cv2() else "diff"
    if method == "flow":
        return _optical_flow_mag(rgb)
    f = rgb.astype(np.float32)
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))   # (T-1,)
    return _pad0(d)


def _optical_flow_mag(rgb: np.ndarray) -> np.ndarray:
    import cv2  # optional; only imported when method="flow"
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    out = []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(np.linalg.norm(flow, axis=2).mean())
    return _pad0(np.asarray(out))


def joint_motion(joints: np.ndarray) -> np.ndarray:
    """(T,J) -> (T,) proprioceptive speed = ||joint_states[t]-joint_states[t-1]||."""
    return _pad0(np.linalg.norm(np.diff(joints, axis=0), axis=1))


def action_motion(actions: np.ndarray) -> np.ndarray:
    """(T,A) -> (T,) commanded magnitude. LIBERO actions are OSC deltas already."""
    return np.linalg.norm(actions, axis=1)


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()
