# roboinfer

Preflight checks for robot training data. Robot learning datasets pair three
streams that were never recorded in lockstep — camera frames, proprioception,
and actions — then index them together anyway. roboinfer estimates the temporal
offset between streams and flags corrupted episodes, so a two-frame lag doesn't
silently cost you a policy.

Two things hide under "misalignment" and get different treatment:

- **Corruption** (dropped/frozen frames, off-by-k action pairing, swapped joint
  order, boundary contamination) — always bad. Detected and flagged.
- **Systematic latency** (camera consistently lags joints by a fixed amount) — a
  property of the rig, possibly present on the deployed robot too. **Measured and
  reported, never silently "fixed."**

## Status

**Day 1–4 (sim, PASS).** Core estimator + detectors validated on LIBERO by
injecting known corruption into synchronized sim data. Dataset-level offset
recovery **max 0.43 frame, 100% within ±1** (injected −6…+9); detection
P/R/F1 = **0.92 / 0.81 / 0.86**. Measures a real **+1.32 f (+66 ms)** systematic
camera latency in LIBERO — reported, not flagged.

**Day 5 (real teleop, mixed — the honest one).** Ran the same pipeline on real
LeRobot v3.0 datasets. **The sim numbers largely do not transfer out of the box**
(full write-up in `DATA.md`):

| | LIBERO (sim) | ALOHA (real, structured) | DROID (real, in-the-wild) |
|--|--|--|--|
| offset recovery max | 0.43 f | 1.24 f | 5.80 f |
| within ±1 | 100% | 82% | 36% |
| verifiable episodes | most | 0/50 | 0/100 |

- The estimator works on structured real teleop (ALOHA ~0.9 f) but breaks on
  in-the-wild footage (DROID 5.8 f) — the vision-vs-joint correlation needs the
  camera to see coherent arm motion.
- **Confidence is miscalibrated on real data** → the tool marks every real
  episode "unverifiable" and abstains (never false-alarms; precision 0.91–1.00).
  Recalibration is the #1 next fix.
- Frozen-camera detection transfers cleanly (recall 1.00 everywhere). The
  action-consistency family is tied to LIBERO's action space and does not.

**Day 6–7 (recalibrate + hunt corruption in OXE).** Replaced the confidence with
a per-episode permutation null → ALOHA went **0/50 → 50/50 verifiable** (97% track
a known injected shift) while DROID stayed abstained. Added an `audit` mode that
separates a rig's systematic offset (reported) from real per-episode
inconsistency (flagged only if internally self-consistent yet a cohort outlier),
and abstains when it can't measure. Result across ALOHA / DROID / OXE (viola,
fractal): **no corruption survived verification** — two initial "found" flags
were artifacts the hardening killed. ALOHA is a real-teleop success (+56 ms
latency, internally consistent); everything else abstains.

Bottom line: a **working, honest measurement instrument on dense-motion real
teleop**, with a QA-grade precision discipline (it abstains rather than guess).
**Not** yet proof that catchable corruption exists at scale — curated data comes
back clean/unverifiable, and the method's visibility envelope (needs
camera-filling motion) is narrow. See `DATA.md` for the full POC verdict.

## How it works

- **Offset**: cross-correlate a vision motion signal (optical-flow magnitude, or
  cheap frame-difference energy as a fallback) against joint-velocity norm;
  sub-frame peak via parabolic interpolation. Action-vs-state-delta offset
  catches off-by-k pairing.
- **Detectors**: frozen camera (vision still while joints move), offset drift
  (first-half vs second-half offset gap), off-by-k, and an action-consistency
  residual — LIBERO's OSC actions linearly predict joint deltas, so one linear
  model catches swapped joints, off-by-k, and boundary contamination as elevated
  residual.
- **Confidence**: cross-correlation peak prominence. Low-motion / ambiguous
  episodes are marked unverifiable, not guessed at.

## Install

```bash
pip install -e .              # core (numpy/scipy/h5py) — frame-diff signal, ~±1 frame
pip install -e '.[flow]'      # + opencv optical flow + mp4 decode (default when present)
pip install -e '.[flow,lerobot]'  # + pyarrow/pandas to scan real LeRobot v3 datasets
```

## Use

Data is **not** committed (see `.gitignore`). Download the LIBERO task file used
for Day 1–4 into `data/libero/` (see `DATA.md` for the exact file + features):

```python
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs("data/libero", exist_ok=True)
p = hf_hub_download("yifengzhu-hf/LIBERO-datasets",
    "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
    repo_type="dataset", token=False)   # stored HF OAuth token may be broken; token=False
shutil.copy(p, "data/libero/")
```

```bash
# LIBERO sim (.hdf5): the Day-4 injected-corruption proof
python -m roboinfer.cli sweep data/libero/*.hdf5
# Real LeRobot v3 dataset (a directory): scan = report flags, sweep = injected GT
python -m roboinfer.cli scan  data/lerobot/droid_100 --camera wrist
python -m roboinfer.cli sweep data/lerobot/aloha_static_tape
```

Download a real dataset with `huggingface_hub.snapshot_download("lerobot/droid_100",
repo_type="dataset", token=False, local_dir="data/lerobot/droid_100")`.

## Caveats

- Offset recovery is a **dataset-level** number (pooled across a task's demos).
  Per-episode offsets are noisier and reported with confidence.
- **Real-data confidence is miscalibrated** (Day 5): the LIBERO-tuned confidence
  reads ~0 on real teleop, so the tool abstains ("unverifiable") even when the
  pooled estimate is good. Recalibration pending.
- The estimator needs the camera to see coherent arm motion — it works on
  structured teleop (ALOHA), not in-the-wild exterior views (DROID).
- Loaders: LIBERO HDF5 (sim) and LeRobot **v3.0** (real). v2.x and RLDS plug in
  behind the same `Episode` dataclass but aren't written.
