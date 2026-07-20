"""Load robot episodes into a common Episode shape.

Only LeRobot-style needs (camera + proprio + actions on one clock). For now:
LIBERO original HDF5 (yifengzhu-hf/LIBERO-datasets). RLDS/LeRobot loaders
plug in later behind the same Episode dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import h5py


@dataclass
class Episode:
    rgb: np.ndarray      # (T,H,W,3) uint8, main camera
    joints: np.ndarray   # (T,J) float, proprioception
    actions: np.ndarray  # (T,A) float, commanded
    ts: np.ndarray       # (T,) float seconds
    name: str
    hz: float = 20.0     # LIBERO control_freq

    def __post_init__(self):
        assert len(self.rgb) == len(self.joints) == len(self.actions) == len(self.ts), \
            f"length mismatch {self.name}: {len(self.rgb)},{len(self.joints)},{len(self.actions)},{len(self.ts)}"

    @property
    def T(self) -> int:
        return len(self.rgb)

    def copy(self, **kw) -> "Episode":
        return replace(self, **{k: (v.copy() if isinstance(v, np.ndarray) else v)
                                 for k, v in {**self.__dict__, **kw}.items()})


def load_libero(path: str, limit: int | None = None, camera: str = "agentview_rgb"):
    """Yield Episodes from a LIBERO task HDF5 (50 demos, 20Hz, synchronized)."""
    with h5py.File(path, "r") as h:
        data = h["data"]
        # numeric demo order, not lexicographic
        demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        if limit:
            demos = demos[:limit]
        for name in demos:
            d = data[name]
            rgb = d["obs"][camera][:]                 # (T,128,128,3) uint8
            joints = d["obs"]["joint_states"][:].astype(np.float64)
            actions = d["actions"][:].astype(np.float64)
            T = len(rgb)
            ts = np.arange(T, dtype=np.float64) / 20.0
            yield Episode(rgb=rgb, joints=joints, actions=actions, ts=ts, name=f"{name}")
