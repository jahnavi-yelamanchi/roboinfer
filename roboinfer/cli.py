"""roboinfer CLI.

  python -m roboinfer.cli sweep PATH.hdf5 [--limit N]   # Day 1-4 injected proof
  python -m roboinfer.cli scan  PATH.hdf5 [--limit N]   # report offsets+flags, no injection
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from .data import load_libero, load_lerobot
from .sweep import run_sweep
from .detect import fit_action_model, score_episode
from .estimate import vision_joint_offset, action_state_offset


def _load(path, limit, camera=None):
    """Dispatch by format: .hdf5 => LIBERO sim; a dir => LeRobot v3 (real)."""
    if path.endswith(".hdf5") or os.path.isfile(path):
        return list(load_libero(path, limit=limit))
    return list(load_lerobot(path, limit=limit, camera=camera))


def _sweep(args):
    eps = _load(args.path, args.limit, args.camera)
    res, _ = run_sweep(eps, seed=args.seed)
    o, f = res["offset"], res["flags"]
    print(f"episodes: {res['n']}")
    print(f"systematic camera offset: {res['baseline_vj_frames']:+.2f} frame "
          f"({res['baseline_vj_ms']:+.0f} ms)  [reported, not flagged]")
    print(f"offset recovery: median {o['median']:.2f}f  max {o['max']:.2f}f  "
          f"within1 {o['within1']*100:.0f}%  (kill: max<=1.0)")
    print(f"flag P/R/F1: {f['precision']:.2f} / {f['recall']:.2f} / {f['f1']:.2f}  "
          f"(tp{f['tp']} fp{f['fp']} fn{f['fn']})")
    print("recall by kind: " + "  ".join(f"{k}={v:.2f}" for k, v in
                                         sorted(res["per_kind_recall"].items())))
    verdict = "PASS" if o["max"] <= 1.0 else "FAIL"
    print(f"\nDay-4 kill checkpoint: {verdict} (offset recovery max {o['max']:.2f} frame)")
    if args.json:
        print(json.dumps(res, indent=2, default=float))


def _scan(args):
    from collections import Counter
    eps = _load(args.path, args.limit, args.camera)
    model = fit_action_model(eps[: max(4, len(eps) // 3)])
    base_vj = float(np.median([vision_joint_offset(e).offset for e in eps]))
    base_as = float(np.median([action_state_offset(e).offset for e in eps]))
    reports = [score_episode(e, model, base_vj, base_as) for e in eps]
    flagged = [r for r in reports if r.flagged]
    unver = [r for r in reports if not r.verifiable]
    conf = np.median([r.confidence for r in reports])
    print(f"episodes: {len(eps)}   systematic camera offset {base_vj:+.2f}f   "
          f"median confidence {conf:.2f}")
    print(f"flagged: {len(flagged)}/{len(eps)} ({len(flagged)/len(eps)*100:.0f}%)   "
          f"unverifiable (conf<0.3): {len(unver)}/{len(eps)}")
    kinds = Counter(f.split("(")[0] for r in flagged for f in r.flags)
    if kinds:
        print("flag kinds: " + "  ".join(f"{k}={v}" for k, v in kinds.most_common()))
    for r in flagged[:30]:
        print(f"  {r.name:22s} conf {r.confidence:.2f}  {','.join(r.flags)}")


def main():
    p = argparse.ArgumentParser(prog="roboinfer")
    sub = p.add_subparsers(required=True)
    for name, fn in [("sweep", _sweep), ("scan", _scan)]:
        s = sub.add_parser(name)
        s.add_argument("path")
        s.add_argument("--limit", type=int, default=40)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--camera", default=None,
                       help="substring to pick a LeRobot camera (e.g. wrist)")
        s.add_argument("--json", action="store_true")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
