# Data + Day 1–4 build journal

## The data used

**Source:** HuggingFace dataset `yifengzhu-hf/LIBERO-datasets` (the original
LIBERO benchmark demos, HDF5 format, from the LIBERO author).

**Exact file used** (copied into `data/libero/`, ~897 MB, git-ignored):

```
libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5
```

Downloaded with `hf_hub_download(..., token=False)` — the machine's stored HF
OAuth token is broken (401 / "signature verification failed"), so all HF calls
in this project pass `token=False`.

Why LIBERO: it is **simulator** data, so the camera / proprioception / action
streams are **synchronized by construction**. That makes it ground truth — inject
a *known* offset or corruption, then measure how well the tool recovers it. No
robot needed. (Full `yifengzhu-hf/LIBERO-datasets` is 132 per-task files across
suites `libero_10/90/spatial/object/goal`, ~1 GB each, 50 demos each.)

### File structure / features

One HDF5 file = one task = 50 demonstrations. Layout: `data/demo_{i}/...`.
Per demo (T = episode length, variable, ~170–234 frames):

| Key | Shape | dtype | Meaning |
|-----|-------|-------|---------|
| `obs/agentview_rgb` | (T,128,128,3) | uint8 | 3rd-person camera **(used)** |
| `obs/eye_in_hand_rgb` | (T,128,128,3) | uint8 | wrist camera |
| `obs/joint_states` | (T,7) | float64 | 7 Panda joint angles **(used)** |
| `obs/ee_pos` / `ee_ori` | (T,3) | float64 | end-effector pos / orientation |
| `obs/ee_states` | (T,6) | float64 | end-effector pose |
| `obs/gripper_states` | (T,2) | float64 | gripper |
| `actions` | (T,7) | float64 | commanded OSC_POSE delta (6 pose + 1 gripper) **(used)** |
| `robot_states` | (T,9) | float64 | sim robot state |
| `states` | (T,45) | float64 | full sim state |
| `dones`, `rewards` | (T,) | uint8 | — |

Recording metadata (from HDF5 attrs): robot = **Panda**, controller =
**OSC_POSE** (`control_delta=true`), **control_freq = 20 Hz** → **1 frame = 50 ms**,
cameras 128×128, offscreen render. This is the clock the whole tool reasons in:
offsets are measured in frames, convertible to ms via 50 ms/frame.

`roboinfer/data.py::load_libero` reads exactly `agentview_rgb`, `joint_states`,
`actions`, synthesizes `ts = arange(T)/20`, and yields an `Episode`.

---

## Day 1–4 progression: what was tried, what failed, what changed

The plan's binding kill checkpoint (Day 4): **offset recovery within ±1 frame on
injected corruption**, else stop and pivot. Everything below builds toward that
number, honestly.

### Day 1 — harness, injectors, metrics

Built `Episode` loader, six ground-truth corruption injectors (`inject.py`:
fixed offset, drifting offset, frozen frames, off-by-k pairing, swapped joints,
boundary contamination), and metrics (`metrics.py`: offset recovery error, flag
precision/recall). Environment: macOS, Apple **MPS only, no CUDA** — Day 1–4 is
all CPU/numpy, so this was fine.

Confirmed the HDF5 actually contains RGB (file size ~1 GB ⇒ images stored, not
low-dim only — LIBERO sometimes ships states-only files needing sim replay;
these are not that). Verified by reading the HDF5 header remotely via `fsspec`
before committing to a 1 GB download.

### Day 2 — offset estimator

Approach (as planned): cross-correlate a **vision motion signal** against
**joint-velocity norm**; peak lag = offset; parabolic interpolation for
sub-frame; confidence from peak prominence (SNR) over the correlation background.

First signal tried: **mean absolute frame-difference** (`signals.vision_motion`,
zero-dependency, cheapest thing that spikes on motion) — climbed the ladder
before reaching for optical flow.

First result (8 episodes, per-episode):
- Clean **action-vs-state** offset ≈ **−0.06 frame** (tight) → actions genuinely
  predict joint deltas. Validated that whole detector family.
- Clean **vision-vs-joint** offset ≈ **+1 frame** with some outliers (up to 3.6).
- Injected **positive** offsets (camera lags) recovered well (median |err|
  0.3–0.8). Injected **negative** offsets and a few episodes recovered badly
  (max err 16–18).

### Day 3 — detectors, confidence, the full sweep

**Failure 1 — per-episode offsets are noisy.** Tried to gate the bad episodes by
confidence. `peak_corr` correlated with error only −0.29; even at a strict gate,
per-episode "within 1 frame" capped at ~75–82%. Conclusion: **per-episode offset
is the wrong unit.** The product estimates a **dataset-level** offset by pooling
episodes. Pooling (median across 40 episodes) gave **max err 0.87 frame** across
injected −5…+10 → already within the kill target.

Built the rest of the detectors (`detect.py`): frozen-camera (vision still while
joints move), offset-drift (first-half vs second-half offset gap), off-by-k, and
an **action-consistency residual** — one linear `actions → joint_delta` model
whose residual spikes for swapped joints, off-by-k, *and* boundary contamination
(one detector, three corruptions; ladder win).

**Failure 2 — first full sweep FAILED at 1.42 frame.** Cause: the sweep applied a
*different* offset to each single episode, so it measured per-episode (noisy)
recovery, not the pooled dataset estimate. **Change:** split the sweep into two
experiments — (a) offset recovery = *same* offset across all episodes, pooled
median (the real product metric); (b) detection = mixed corruption for
precision/recall. Re-run: offset **max 0.71**, detection **F1 0.86**.

Also added a per-episode **camera-offset-anomaly** flag (an episode whose offset
deviates from the dataset baseline is corruption, distinct from the systematic
baseline itself, which is reported not flagged).

### Day 4 — kill checkpoint, and the one thing that didn't clear ±1

The self-check test kept **failing at offset k = −2** (err 1.18–1.27). Not sample
noise — reproducible. Net offset there ≈ −1.4 frame after the systematic bias,
right where the frame-difference signal has a ~1-frame precision floor.

- **Tried: 4× upsampling** the signals before cross-correlation for finer
  sub-frame lag. Result 1.18 → 1.06. Barely moved → the limit is signal *content*,
  not interpolation. **Rejected.**
- **Tried: optical-flow magnitude** (Farneback, opencv) — the signal the plan
  actually named. Cached flow once per clean episode and sliced it per offset
  (offset injection only trims frames), so the sweep stayed cheap. Result:
  **worst 0.79 frame** at 40 episodes; k = −2 dropped 1.06 → 0.79. Robustly under
  ±1 everywhere. **Accepted.**

**Change:** made optical flow the **default** for the offset estimate
(`method="auto"` → flow when opencv is installed, frame-diff fallback otherwise).
Kept fast frame-diff for the detection signals (frozen etc.), where speed matters
and precision needs are looser. Cost: ~14 s flow per 49-demo task — fine for a
tool run once per dataset.

### Final Day-4 result (accepted)

Full task, 49 demos, optical-flow default:

```
systematic camera offset: +1.32 frame (+66 ms)   [reported, not flagged]
offset recovery: median 0.20f  max 0.43f  within1 100%   (kill: max<=1.0)  → PASS
flag P/R/F1: 0.92 / 0.81 / 0.86  (tp34 fp3 fn8)
recall by kind: frozen 1.00  boundary/offbyk/swap 0.86  offset 0.71  drift 0.57
```

**Why accepted:**
- Offset recovery **max 0.43 frame, 100% within ±1** clears the binding kill
  criterion decisively — with the plan's named signal, on real robot video.
- Clean action-vs-state offset ≈ 0 independently validates the detector family.
- Detection F1 0.86; every corruption kind is caught.
- The **+1.32 frame (66 ms)** systematic camera latency is a real LIBERO rig
  property the tool *measures and reports* rather than silently "fixes" — exactly
  the corruption-vs-latency distinction the product is built around.

**Known weak spots (honest, carried forward):** drift-detection recall 0.57 is
the worst detector; 3 false positives on clean episodes (precision 0.92);
per-episode (vs dataset-level) offset remains noisy by design and is reported
with a confidence gate ("unverifiable, not guessed").

### Reproduce

```bash
pip install -e '.[flow,dev]'
python -m roboinfer.cli sweep data/libero/*.hdf5   # prints the block above
pytest -q                                          # injectors + offset-recovery ≤1 frame
```

---

## Day 5 — the real-data reality check (the honest one)

Days 1–4 proved recovery of *injected* corruption on *sim* data (LIBERO). That
proves the detector recovers offsets we planted; it does **not** prove the tool
works on real teleop, where corruption actually lives. Day 5 ran the same
pipeline on two real public datasets. The result is sobering and important —
**the LIBERO numbers largely do not transfer to real data out of the box.**

### New data used (real teleop, LeRobot v3.0, both git-ignored)

| Dataset | Size | Ep | fps | Robot | Cameras |
|---------|------|----|-----|-------|---------|
| `lerobot/droid_100` | 464 MB | 100 | 15 | Franka 7-DOF, in-the-wild | 2 exterior + 1 wrist (180×320) |
| `lerobot/aloha_static_tape` | ~1.2 GB | 50 | 50 | bimanual 14-DOF, tabletop | 4 fixed (480×640) |

New loader `data.py::load_lerobot` reads LeRobot **v3.0** (episodes concatenated
into shards; `meta/episodes/*.parquet` maps each episode to a parquet row-range +
per-camera video frame-range). Low-dim via pyarrow, video decoded sequentially
with cv2 (opencv 5 handles the av1/yuv420p codec — no extra dep), downscaled to
128px. Same `Episode` dataclass, so estimate/detect/sweep are format-blind.
CLI auto-dispatches by format (`.hdf5`→LIBERO, dir→LeRobot) and takes `--camera`.

### What happened (both `scan`, no ground truth, and `sweep`, injected GT)

```
                          LIBERO(sim)   ALOHA(real,structured)   DROID(real,wild)
offset recovery max          0.43 f            1.24 f                 5.80 f
offset recovery within±1      100%              82%                    36%
median confidence            (high)            0.07                   ~0.04
episodes "verifiable"        most              0 / 50                 0 / 100
detection F1                 0.86              0.44                   0.30
  frozen-camera recall       1.00              1.00                   1.00
  action-family recall       0.86         swap0/offbyk.17/bnd.40      ~0.00
```

### Three honest findings

1. **The offset estimator partially transfers — structure-dependent.** On ALOHA
   (fixed cameras, tabletop, arms fill the frame) pooled recovery is **0.91 f
   median / 1.24 f max** — nearly the kill target; the estimator basically works.
   On DROID (in-the-wild, varied scenes, arm small in an exterior view) it is
   **5.80 f** — broken. The vision-motion-vs-joint-velocity correlation needs the
   camera to actually *see* coherent arm motion; unstructured footage kills it.

2. **Confidence is miscalibrated for real data.** The peak-SNR confidence was
   tuned on LIBERO's clean correlation, so it reads ~0.07 on *all* real episodes —
   even ALOHA ones whose pooled estimate is a good 0.91 f. Consequence: the tool
   currently marks **every** real episode "unverifiable" and abstains. That is
   honest (it never false-alarms — precision held at 0.91–1.00) but it means the
   tool cannot yet *positively* verify real data. Recalibrating confidence on real
   correlation distributions is the #1 fix.

3. **Only frozen-camera detection is embodiment-agnostic.** Frozen recall = 1.00
   everywhere. The action-consistency family (off-by-k / swap / boundary) is tied
   to LIBERO's OSC-delta action space where `action → joint_delta` is linear; on
   DROID's action representation it gives ~0 recall. Drift stayed 0 (already the
   worst detector; see below). ts_jitter never fired — DROID's LeRobot conversion
   **regenerated uniform timestamps** (dt ≡ 1/fps exactly), discarding the real
   capture clock, so the real-timestamp signal isn't available on this release.

### A real precision bug, found and fixed on real data

First DROID `scan` flagged **78/100** episodes, almost all `offset_drift` (values
3–30 f). Cause: the drift flag fired on first-half-vs-second-half offset *gaps*
computed from unverifiable (~0-confidence) estimates — pure noise flagged as
drift. Fix: gate the drift flag on `vj.confidence ≥ 0.15` (matching the other
offset flags). After the fix, DROID false-positive drift → 0 (sweep precision
1.00). This is exactly the precision-over-recall failure mode a QA gate must
avoid: at 76k episodes, 78% false flags would be catastrophic.

### Compute (answers "how long to scan my dataset")

All local CPU, single-thread. DROID `scan` = **~1.4 s/episode** at 128px
(140 s / 100 ep, decode + flow + all detectors); ALOHA heavier (larger frames).
Extrapolated to DROID-scale (76k episodes): ~30 CPU-hours single-thread →
~2–4 h on 8–16 cores, embarrassingly parallel per episode. **No cloud GPU was
needed for validation** (Modal/vast only matter for a future full-dataset
benchmark, not this go/no-go).

### Verdict — viability, stated plainly

- **The tool is currently a sim + structured-teleop tool, not an in-the-wild
  one.** LIBERO ±0.4 f and ALOHA ~0.9 f work; DROID does not. The "audit any
  fleet" pitch does **not** hold yet — it holds for controlled-rig data.
- **The existential question is not yet answered.** Neither curated public
  release showed obvious *real* corruption — the flags we got were mostly detector
  artifacts, not defects. Curated benchmark datasets are the wrong hunting ground;
  real corruption lives in **raw, un-curated teleop logs**, which these polished HF
  releases are not. Proving the problem exists needs raw operator logs (Day 6).
- **The failures are concentrated and named** (confidence calibration on real
  correlations, robustness to unstructured motion, action-rep-agnostic
  consistency), so this is a "needs Day 6–7 work," not necessarily a "dead idea."
  But the honest state before applying: **real-data efficacy is unproven, and one
  of two real datasets fails the kill metric.**

### Reproduce (Day 5)

```bash
python -m roboinfer.cli scan  data/lerobot/droid_100 --camera wrist   # 100% unverifiable, honest abstain
python -m roboinfer.cli sweep data/lerobot/aloha_static_tape          # 0.91f median (near ±1)
python -m roboinfer.cli sweep data/lerobot/droid_100 --camera wrist   # 5.80f — FAIL, does not transfer
```

---

## Day 6–7 — recalibrate confidence, then hunt corruption where it should live

Day 5 said two opposite things: the method *works on structured teleop* (ALOHA
0.91f) but *abstains on everything* because the confidence metric was tuned on
LIBERO. And the curated datasets showed no real corruption — because curated HF
releases are **pre-cleaned**. So Day 6–7 does two things: (A) fix confidence so
real estimates become trustworthy; (B) hunt in a heterogeneous, less-cleaned
corpus (Open-X-Embodiment), with a binding go/no-go written before running.

### Phase A — permutation-null confidence (the fixable finding)

Replaced the LIBERO-tuned confidence (`clip(snr/8)·peak`) with a **per-episode
permutation null**: phase-randomize the vision signal (same power spectrum, no
real cross-alignment), recompute the correlation peak K=48 times, and set
confidence from how many sigmas the real peak stands above that null
(`estimate.py::estimate_offset`, `CONF_Z_FULL=6`). The null is rebuilt per
episode from the signal's own spectrum — no constant tuned on one corpus.

**Validated against the injected sweep as ground truth:**

| | before | after |
|--|--|--|
| ALOHA verifiable | 0 / 50 | **50 / 50** |
| ALOHA episodes that track a known injected shift | — | **97%** |
| DROID verifiable (honesty guard: must stay low) | 0 | **~20%** (stays abstained) |
| LIBERO Day-4 offset recovery | pass | **pass** (pytest green) |

Recalibration converted ALOHA from "abstain" to "verify" **and** kept DROID
abstained — it didn't just inflate everything. Gate passed.

### Phase B — the OXE audit and the corruption-vs-rig test

**Loader** (`data.py::load_lerobot`) extended for real OXE: LeRobot v3.0,
partial download (first shards only via `allow_patterns`), and load-time gating
that drops any dataset without v3.0 layout / `observation.state` / a non-wrist
camera, filtering the shard by `episode_index` (robust to multi-shard global
row indexing).

**The credibility core — how we separate *silent corruption* from a
*legitimately different rig*** (`detect.py::dataset_audit`): never compare across
rigs. Per dataset: the **median** per-episode offset is that rig's systematic
latency (*reported, never flagged*). The **within-episode** offset half-gap is
that dataset's own measurement-noise floor. An episode is flagged only if it is
**internally self-consistent** (small half-gap) yet deviates from its *own rig's*
systematic offset by more than 4× the noise floor. Plus abstain guards: fps < 5,
too few verifiable/long episodes, or an uncalibratable noise floor → **ABSTAIN**,
not a guess.

**Results (hardened):**

| Dataset | Robot / role | fps | verifiable | systematic | verdict |
|---------|--------------|-----|------------|-----------|---------|
| aloha_static_tape | bimanual tabletop (**positive control**) | 50 | **50/50** | +2.82f (+56 ms) | CLEAN (internally consistent) |
| droid_100 (wrist) | Franka in-the-wild | 15 | 12/50 | +0.12f (+8 ms) | CLEAN (abstains on 38) |
| viola (OXE) | Franka tabletop, far 3rd-person | 20 | 10/60, noise 9.9f | −10.6f (unreliable) | CLEAN (abstains in substance) |
| fractal / RT-1 (OXE) | Google robot | 3 | — | — | **ABSTAIN** (fps too coarse) |

**Two false positives, found and killed — the precision discipline that matters.**
A first pass reported "CORRUPTION FOUND": 14 fractal episodes and 1 DROID episode.
Both were artifacts. Fractal's within-episode noise floor collapsed to 0.00f at
3 fps / short episodes → threshold defaulted to 2f → normal cross-episode spread
tripped it. DROID's single flag was a bare-threshold outlier that failed the
self-consistency test. The hardening (fps guard + self-consistency requirement +
uncalibratable-floor abstain) drove both to **0**. At 76k episodes those artifacts
would have been thousands of false accusations.

**Dataset-availability reality:** of three OXE picks, only **viola** loaded
cleanly. `bridge_data_v2` and `fractal` are third-party v3.0 conversions with
non-standard sharding (data and video shard indices unrelated; global row
offsets), and fractal runs at 3 fps — below the method's temporal resolution.

### POC verdict — against the binding criteria written before running

- **Corruption-existence gate: FAIL (honest negative).** No corruption survived
  verification in any dataset the tool can trust. Every "found" flag was an
  artifact we caught. The only OXE set we could partly see (viola) abstains in
  substance; the others were unloadable or too coarse.
- **But the method and its honesty machinery are validated.** ALOHA is an
  end-to-end success on *real* teleop: 100% verifiable, a plausible **+56 ms**
  systematic camera latency, internally consistent. Recalibration works. The
  audit refuses to cry wolf — it killed its own false positives and abstains
  where it cannot see.
- **Newly exposed, decision-relevant limitation — a narrow visibility envelope.**
  The method needs dense, camera-filling motion (bimanual / close overhead like
  ALOHA, or wrist egomotion). Most of OXE is single-arm 3rd-person with the arm
  small in frame → it abstains. So even if corruption exists in OXE, this method
  can only audit the ALOHA-class slice of it.

**Bottom line before applying:** roboinfer is a *working, honest measurement
instrument* on suitable (dense-motion) real teleop, with a confidence gate that
correctly abstains and a QA-grade precision discipline. It is **not** yet proof
that catchable corruption exists at scale — every accessible curated dataset came
back clean or unverifiable, and the corpus we could load is narrow. The
existential question ("does silent corruption exist in data people actually
train on, at a rate worth selling against?") remains **open**, gated now less by
the detector than by (a) access to raw un-curated logs and (b) the method's
motion-density envelope. That is the honest state to decide the pivot on.

### Reproduce (Day 6–7)

```bash
python -m roboinfer.cli audit data/lerobot/aloha_static_tape   # 50/50 verifiable, CLEAN, +56ms
python -m roboinfer.cli audit data/lerobot/droid_100 --camera wrist   # abstains on 38/50, CLEAN
python -m roboinfer.cli audit data/lerobot/viola              # abstains in substance
python -m roboinfer.cli audit data/lerobot/fractal            # ABSTAIN (3 fps)
```
