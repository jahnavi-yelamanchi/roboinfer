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

## Status (Day 1–4)

Core estimator + detectors validated on LIBERO by injecting known corruption
into synchronized sim data and measuring recovery.

**Day-4 kill checkpoint: PASS.** On a LIBERO task (49 demos), dataset-level
offset recovery error is **max 0.43 frame, 100% within ±1** across injected
offsets −6…+9. Detection precision/recall/F1 = **0.92 / 0.81 / 0.86**
(frozen 1.00, boundary/off-by-k/swap 0.86, offset 0.71, drift 0.57).

It also measures a real **+1.32 frame (+66 ms) systematic camera-vs-joint
latency** in LIBERO — reported, not flagged.

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
pip install -e .            # core (numpy/scipy/h5py) — frame-diff signal, ~±1 frame
pip install -e '.[flow]'    # + opencv optical flow (sub-frame precision, default when present)
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
python -m roboinfer.cli sweep data/libero/*.hdf5   # injected-corruption proof (the Day-4 metric)
python -m roboinfer.cli scan  data/libero/*.hdf5   # report offsets + flags on a dataset, no injection
```

## Caveats

- Offset recovery is a **dataset-level** number (pooled across a task's demos).
  Per-episode offsets are noisier and reported with confidence.
- The default frame-diff signal is marginal (~1 frame) at small camera-lead
  offsets; optical flow (`.[flow]`) clears ±1 robustly and is the default when
  installed.
- LIBERO/RLDS/LeRobot loaders plug in behind the `Episode` dataclass; only the
  LIBERO HDF5 loader exists so far.
