"""Per-episode corruption detectors + confidence.

Design split (from the plan): corruption is always-bad (flag it); a constant
systematic offset is a rig property (measure + report, never silently flag).
So the offset estimate is reported as a number; only *anomalous* offset
(off-by-k, drift) and hard corruption (frozen, boundary, swap) raise flags.

The action-consistency residual is the workhorse: LIBERO actions (OSC deltas)
linearly predict joint deltas, so one linear model catches swapped joints,
off-by-k pairing, and boundary contamination as elevated residual.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .signals import vision_motion, joint_motion, action_motion
from .estimate import estimate_offset, vision_joint_offset, action_state_offset


@dataclass
class ActionModel:
    W: np.ndarray            # (A, J) least-squares map actions -> joint_delta
    res_med: float           # median per-episode residual on the fit pool
    res_mad: float           # robust scale of residual


def fit_action_model(eps) -> ActionModel:
    """Fit joint_delta ~ actions on a pool assumed mostly clean (robust threshold)."""
    A, D = [], []
    for e in eps:
        A.append(e.actions[:-1])
        D.append(np.diff(e.joints, axis=0))
    A, D = np.vstack(A), np.vstack(D)
    W, *_ = np.linalg.lstsq(A, D, rcond=None)
    res = np.array([_residual(e, W) for e in eps])
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med))) + 1e-9
    return ActionModel(W=W, res_med=med, res_mad=mad)


def _residual(ep, W) -> float:
    pred = ep.actions[:-1] @ W
    return float(np.linalg.norm(np.diff(ep.joints, axis=0) - pred, axis=1).mean())


@dataclass
class EpisodeReport:
    name: str
    vj_offset: float                 # camera vs joints, frames
    as_offset: float                 # action vs state, frames
    confidence: float                # 0..1; low => unverifiable
    flags: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return len(self.flags) > 0

    @property
    def verifiable(self) -> bool:
        return self.confidence >= 0.3


def detect_frozen(ep, motion=None, min_run=5) -> tuple[bool, int]:
    """Camera frozen: vision motion ~0 while joints still moving. Returns (flag, run)."""
    vm = vision_motion(ep.rgb) if motion is None else motion
    jm = joint_motion(ep.joints)
    jmov = jm > (0.2 * jm.mean() + 1e-9)
    frozen = (vm <= 1e-6) & jmov            # camera dead but robot moving
    run = best = 0
    for f in frozen:
        run = run + 1 if f else 0
        best = max(best, run)
    return best >= min_run, best


def drift_score(ep, max_lag=15) -> float:
    """Offset measured on first vs second half; large gap => drifting offset."""
    T = ep.T
    if T < 60:
        return 0.0
    h = T // 2
    vm, jm = vision_motion(ep.rgb), joint_motion(ep.joints)
    o1 = estimate_offset(vm[:h], jm[:h], max_lag).offset
    o2 = estimate_offset(vm[h:], jm[h:], max_lag).offset
    return abs(o1 - o2)


def ts_jitter_score(ep) -> float:
    """Frame-timing irregularity from the REAL recorded clock: max gap / median
    gap. 1.0 = perfectly regular. >1 => dropped frames / clock desync — the
    production failure sim data can't have (LIBERO's ts is synthesized uniform,
    so this stays 1.0 and never flags there)."""
    dt = np.diff(np.asarray(ep.ts, float))
    if len(dt) < 2:
        return 1.0
    med = np.median(dt)
    return float(dt.max() / med) if med > 0 else float("inf")


TS_JITTER_K = 1.5   # ponytail: gap 1.5x nominal = likely drop; tune per-rig from data


def _flow_halfgap(ep, max_lag=15) -> float:
    """Within-episode measurement noise on the SAME (flow) signal as the offset:
    offset on first vs second half. Overestimates full-episode noise (shorter
    windows) so the anomaly threshold stays conservative."""
    vm, jm = vision_motion(ep.rgb, "auto"), joint_motion(ep.joints)
    n = min(len(vm), len(jm)); h = n // 2
    if h < 20:
        return 0.0
    o1 = estimate_offset(vm[:h], jm[:h], max_lag).offset
    o2 = estimate_offset(vm[h:h + h], jm[h:h + h], max_lag).offset
    return abs(o1 - o2)


def dataset_audit(eps, conf_min=0.3):
    """Corruption-vs-rig audit for one real dataset (no ground truth).

    The credibility rule: a different rig is *allowed* to have a different
    systematic offset — that is never corruption. We only flag an episode whose
    alignment deviates from *this dataset's own* systematic offset by more than
    *this dataset's own* measurement-noise floor. Cross-rig comparison never
    enters. Runs only the two signals proven to generalize (offset, frozen);
    reports over confidence-verifiable episodes only.
    """
    hz0 = eps[0].hz if eps else 20.0
    if hz0 < 5:                                            # too coarse: skip heavy compute
        return dict(n=len(eps), n_verifiable=0, hz=hz0,
                    verdict=f"ABSTAIN (fps {hz0:.0f} too coarse for temporal offset)")
    rows = [(e.name, vision_joint_offset(e), detect_frozen(e)[0],
             _flow_halfgap(e) if e.T >= 60 else np.nan, e.hz) for e in eps]
    name = [r[0] for r in rows]
    off = np.array([r[1].offset for r in rows])
    conf = np.array([r[1].confidence for r in rows])
    frozen = np.array([r[2] for r in rows], bool)
    g = np.array([r[3] for r in rows])                       # within-episode half-gap (noise)
    hz = rows[0][4] if rows else 20.0
    ver = conf >= conf_min
    n, nver = len(eps), int(ver.sum())
    out = dict(n=n, n_verifiable=nver, hz=hz)

    # abstain guards: don't cry wolf where the method can't measure
    if hz < 5:
        out["verdict"] = f"ABSTAIN (fps {hz:.0f} too coarse for temporal offset)"
        return out
    valid = ver & ~np.isnan(g)                              # verifiable AND long enough to self-check
    if int(valid.sum()) < 5:
        out["verdict"] = "ABSTAIN (too few verifiable, long-enough episodes to calibrate)"
        return out

    o = off[ver]
    systematic = float(np.median(o))
    cross_mad = float(np.median(np.abs(o - systematic)))     # cross-episode spread
    noise = float(np.median(g[valid]))                       # within-episode meas. noise
    if noise < 0.1:                                          # degenerate floor => can't threshold
        out["verdict"] = "ABSTAIN (noise floor uncalibratable — coarse/quantized offsets)"
        return out
    tau = max(2.0, 4.0 * noise)
    # a real misalignment: internally self-consistent (small half-gap) yet far from
    # the rig's own systematic offset — not merely a noisy estimate that passed the gate
    self_consistent = g < noise
    anom = valid & self_consistent & (np.abs(off - systematic) > tau)
    frozen_v = ver & frozen
    corrupt = bool(anom.sum() or frozen_v.sum())
    out.update(systematic_f=systematic, systematic_ms=systematic * 1000.0 / hz,
               cross_mad=cross_mad, noise_floor=noise, tau=tau,
               anom_ct=int(anom.sum()), anom_names=[name[i] for i in np.where(anom)[0]][:12],
               frozen_ct=int(frozen_v.sum()),
               frozen_names=[name[i] for i in np.where(frozen_v)[0]][:12],
               verdict="CORRUPTION FOUND" if corrupt else "CLEAN (internally consistent)")
    return out


def score_episode(ep, model: ActionModel, baseline_vj=0.0, baseline_as=0.0) -> EpisodeReport:
    vj = vision_joint_offset(ep)
    as_ = action_state_offset(ep)
    frozen, run = detect_frozen(ep)
    drift = drift_score(ep)
    res = _residual(ep, model.W)
    res_z = (res - model.res_med) / model.res_mad
    jitter = ts_jitter_score(ep)

    r = EpisodeReport(name=ep.name, vj_offset=vj.offset, as_offset=as_.offset,
                      confidence=vj.confidence,
                      scores=dict(vj_conf=vj.confidence, as_conf=as_.confidence,
                                  frozen_run=run, drift=drift, res_z=res_z,
                                  ts_jitter=jitter))
    if frozen:
        r.flags.append(f"frozen_camera(run={run})")
    if jitter >= TS_JITTER_K:
        r.flags.append(f"ts_jitter(x{jitter:.1f})")
    # drift is a gap between two half-window offset estimates; only trust it when
    # the offset signal itself is verifiable, else unreliable estimates flag noise
    # as drift (seen on DROID in-the-wild: 78% false drift at conf~0).
    if drift >= 3.0 and vj.confidence >= 0.15:
        r.flags.append(f"offset_drift({drift:.1f}f)")
    if abs(vj.offset - baseline_vj) >= 2.0 and vj.confidence >= 0.15:
        r.flags.append(f"camera_offset_anomaly({vj.offset - baseline_vj:+.1f}f)")
    if abs(as_.offset - baseline_as) >= 1.0 and as_.confidence >= 0.15:
        r.flags.append(f"action_offby({as_.offset - baseline_as:+.1f}f)")
    if res_z >= 6.0:
        r.flags.append(f"action_inconsistent(z={res_z:.1f})")
    return r
