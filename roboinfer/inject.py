"""Known-corruption injectors. Sim data is synchronized by construction, so the
injected parameter IS ground truth. Each injector returns (corrupted_episode,
Corruption label) for scoring offset recovery and flag precision/recall.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .data import Episode


@dataclass
class Corruption:
    kind: str                       # clean|offset|drift|frozen|offbyk|swap|boundary
    offset: float | None = None     # injected camera-vs-joint offset in frames
    params: dict = field(default_factory=dict)

    @property
    def corrupt(self) -> bool:
        return self.kind != "clean"


def clean(ep: Episode) -> tuple[Episode, Corruption]:
    return ep, Corruption("clean")


def inject_offset(ep: Episode, k: int) -> tuple[Episode, Corruption]:
    """Camera lags joints/actions by k frames (k<0 => camera leads). Trims overlap."""
    T = ep.T
    if k >= 0:
        rgb, joints, actions = ep.rgb[:T - k], ep.joints[k:], ep.actions[k:]
    else:
        j = -k
        rgb, joints, actions = ep.rgb[j:], ep.joints[:T - j], ep.actions[:T - j]
    ts = ep.ts[:len(rgb)]
    out = Episode(rgb=rgb, joints=joints, actions=actions, ts=ts,
                  name=f"{ep.name}+off{k}", hz=ep.hz)
    return out, Corruption("offset", offset=float(k), params={"k": k})


def inject_drift(ep: Episode, kmax: float) -> tuple[Episode, Corruption]:
    """Camera timeline drifts: offset grows 0 -> kmax linearly across the episode."""
    T = ep.T
    drift = np.linspace(0, kmax, T)
    src = np.clip(np.arange(T) - drift, 0, T - 1)      # fractional -> nearest
    idx = np.round(src).astype(int)
    out = ep.copy(rgb=ep.rgb[idx], name=f"{ep.name}+drift{kmax}")
    return out, Corruption("drift", offset=kmax / 2.0, params={"kmax": kmax})


def inject_frozen(ep: Episode, start: int, length: int) -> tuple[Episode, Corruption]:
    """Camera freezes: a span of frames all equal the first frozen frame."""
    rgb = ep.rgb.copy()
    start = min(start, ep.T - 1)
    length = min(length, ep.T - start)
    rgb[start:start + length] = rgb[start]
    out = ep.copy(rgb=rgb, name=f"{ep.name}+frozen{start}:{length}")
    return out, Corruption("frozen", params={"start": start, "length": length})


def inject_offbyk(ep: Episode, k: int) -> tuple[Episode, Corruption]:
    """Actions paired off-by-k against states (shift actions relative to joints)."""
    actions = np.roll(ep.actions, k, axis=0)
    actions[:max(k, 0)] = ep.actions[0]
    out = ep.copy(actions=actions, name=f"{ep.name}+obk{k}")
    return out, Corruption("offbyk", params={"k": k})


def inject_swap(ep: Episode, i: int, j: int) -> tuple[Episode, Corruption]:
    """Two joint-state columns swapped (wiring / ordering bug)."""
    joints = ep.joints.copy()
    joints[:, [i, j]] = joints[:, [j, i]]
    out = ep.copy(joints=joints, name=f"{ep.name}+swap{i},{j}")
    return out, Corruption("swap", params={"i": i, "j": j})


def inject_boundary(ep: Episode, other: Episode, n: int) -> tuple[Episode, Corruption]:
    """Episode boundary contaminated: first n frames come from another episode."""
    m = min(n, other.T, ep.T - 1)
    rgb = np.concatenate([other.rgb[:m], ep.rgb[m:]])
    joints = np.concatenate([other.joints[:m], ep.joints[m:]])
    actions = np.concatenate([other.actions[:m], ep.actions[m:]])
    out = Episode(rgb=rgb, joints=joints, actions=actions, ts=ep.ts[:len(rgb)],
                  name=f"{ep.name}+bound{m}", hz=ep.hz)
    return out, Corruption("boundary", params={"n": m})
