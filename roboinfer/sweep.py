"""Injected-corruption sweep: the Day 1-4 proof harness.

Takes clean (synchronized) sim episodes, injects known corruption into a
balanced set, runs detection, and reports offset-recovery error + flag
precision/recall. No robot needed — sim data is ground truth.
"""
from __future__ import annotations
import random
import numpy as np
from . import inject
from .detect import fit_action_model, score_episode
from .estimate import vision_joint_offset, action_state_offset, estimate_offset
from .signals import vision_motion, joint_motion
from .metrics import offset_recovery_error, prf


OFFSETS = [-6, -4, -2, -1, 0, 1, 2, 3, 5, 7, 9]


def _offset_at(vf, ep, k):
    """Estimate recovered offset for inject_offset(ep, k) using cached clean flow vf.
    inject_offset only trims frames, so the vision signal is a slice of vf."""
    T = ep.T
    if k >= 0:
        v = vf[:T - k]; joints = ep.joints[k:]
    else:
        j = -k; v = np.concatenate([[0.0], vf[j + 1:]]); joints = ep.joints[:T - j]
    jm = joint_motion(joints)
    n = min(len(v), len(jm))
    return estimate_offset(v[:n], jm[:n]).offset


def _offset_experiment(eps, base_vj):
    """Same offset injected across ALL episodes; pooled (median) estimate is the
    dataset-level offset the product actually reports. This is the kill metric."""
    vf = [vision_motion(e.rgb, "auto") for e in eps]      # flow once per clean episode
    inj, rec = [], []
    for k in OFFSETS:
        ests = [_offset_at(vf[i], e, k) for i, e in enumerate(eps)]
        inj.append(float(k)); rec.append(float(np.median(ests)))
    return offset_recovery_error(inj, rec, bias=base_vj)


def _detection_experiment(eps, model, base_vj, base_as, rng):
    """Mixed corruption, one kind per episode; episode-level flag precision/recall."""
    n = len(eps)
    reports, truth, kinds = [], [], []
    for i, e in enumerate(eps):
        kind = ["clean", "offset", "frozen", "offbyk", "swap", "boundary", "drift"][i % 7]
        if kind == "clean":
            ce, lab = inject.clean(e)
        elif kind == "offset":
            ce, lab = inject.inject_offset(e, rng.choice([-6, -4, 4, 6, 8]))
        elif kind == "frozen":
            ce, lab = inject.inject_frozen(e, rng.randint(20, e.T // 2), rng.randint(8, 20))
        elif kind == "offbyk":
            ce, lab = inject.inject_offbyk(e, rng.choice([2, 3, 4, 5]))
        elif kind == "swap":
            ce, lab = inject.inject_swap(e, *rng.sample(range(7), 2))
        elif kind == "boundary":
            ce, lab = inject.inject_boundary(e, eps[(i + 1) % n], rng.randint(10, 25))
        else:  # drift
            ce, lab = inject.inject_drift(e, rng.choice([6, 8, 10]))
        reports.append(score_episode(ce, model, base_vj, base_as))
        truth.append(lab.corrupt); kinds.append(kind)
    pred = [r.flagged for r in reports]
    return reports, prf(truth, pred), _recall_by_kind(kinds, truth, pred)


def run_sweep(eps, seed=0):
    rng = random.Random(seed)
    eps = list(eps)
    n = len(eps)
    fit_pool = eps[: max(4, n // 3)]           # clean reference for the action model
    model = fit_action_model(fit_pool)

    # dataset baselines (systematic, reported not flagged) from clean episodes
    base_vj = float(np.median([vision_joint_offset(e).offset for e in fit_pool]))
    base_as = float(np.median([action_state_offset(e).offset for e in fit_pool]))

    reports, flags, per_kind = _detection_experiment(eps, model, base_vj, base_as, rng)
    result = dict(
        n=n,
        baseline_vj_frames=base_vj, baseline_vj_ms=base_vj * (1000 / eps[0].hz),
        baseline_as_frames=base_as,
        offset=_offset_experiment(eps, base_vj),
        flags=flags,
        per_kind_recall=per_kind,
    )
    return result, reports


def _recall_by_kind(kinds, truth, pred):
    out = {}
    for k in set(kinds):
        idx = [i for i, kk in enumerate(kinds) if kk == k and truth[i]]
        if idx:
            out[k] = sum(pred[i] for i in idx) / len(idx)
    return out
