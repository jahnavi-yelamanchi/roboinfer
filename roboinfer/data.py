"""Load robot episodes into a common Episode shape.

Only LeRobot-style needs (camera + proprio + actions on one clock):
- LIBERO original HDF5 (yifengzhu-hf/LIBERO-datasets), synchronized sim.
- LeRobot v2 (parquet + per-episode mp4), real teleop with a real timestamp col.
Both yield the same Episode dataclass, so estimate/detect/sweep are format-blind.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import glob, json, os
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


def _read_frames(cap, count: int, size: int) -> np.ndarray:
    """Read `count` consecutive frames from an open cv2 VideoCapture -> RGB uint8."""
    import cv2
    out = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    return np.asarray(out, np.uint8)


def load_lerobot(path: str, limit: int | None = None, camera: str | None = None,
                 size: int = 128):
    """Yield Episodes from a LeRobot **v3.0** dataset dir (DROID, ALOHA, ...).

    v3 concatenates many episodes into shard files; meta/episodes/*.parquet maps
    each episode to a parquet row-range and a per-camera video frame-range. We
    slice the low-dim rows and decode the video shard sequentially (no per-frame
    seek — av1 keyframe seeking is unreliable). `camera` = substring to pick a
    video key (default first). Uses the real recorded `timestamp` column.
    """
    import cv2, glob as _glob
    import pyarrow.parquet as pq

    info = json.load(open(os.path.join(path, "meta", "info.json")))
    hz = float(info["fps"])
    vid_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not vid_keys:
        raise ValueError(f"no video feature in {path}")
    cam = next((k for k in vid_keys if camera and camera in k), vid_keys[0])
    if "file_index" not in info["data_path"]:
        raise ValueError(f"{path}: only LeRobot v3.0 layout supported "
                         f"(data_path={info['data_path']})")

    em = pq.read_table(_glob.glob(os.path.join(path, "meta", "episodes", "**", "*.parquet"),
                                  recursive=True)[0]).to_pandas()
    em = em.sort_values("episode_index")
    if limit:
        em = em.head(limit)
    vck, vfk = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
    fts = f"videos/{cam}/from_timestamp"
    em = em.sort_values([vck, vfk, fts])       # decode order = shard order

    cap, cur, fpos = None, None, 0
    for _, r in em.iterrows():
        i = int(r["episode_index"])
        pf = os.path.join(path, info["data_path"].format(
            chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
        a, b = int(r["dataset_from_index"]), int(r["dataset_to_index"])
        t = pq.read_table(pf, columns=["observation.state", "action", "timestamp"]).slice(a, b - a)
        joints = np.stack(t["observation.state"].to_pylist()).astype(np.float64)
        actions = np.stack(t["action"].to_pylist()).astype(np.float64)
        ts = np.asarray(t["timestamp"].to_pylist(), np.float64)

        shard = (int(r[vck]), int(r[vfk]))
        if shard != cur:
            if cap is not None:
                cap.release()
            vf = os.path.join(path, info["video_path"].format(
                video_key=cam, chunk_index=shard[0], file_index=shard[1]))
            cap = cv2.VideoCapture(vf)
            cur, fpos = shard, 0
        start = int(round(float(r[fts]) * hz))
        while fpos < start:                    # skip to episode's first frame
            if not cap.grab():
                break
            fpos += 1
        rgb = _read_frames(cap, len(joints), size)
        fpos += len(rgb)

        T = min(len(rgb), len(joints), len(actions), len(ts))
        if T == 0:
            continue
        yield Episode(rgb=rgb[:T], joints=joints[:T], actions=actions[:T],
                      ts=ts[:T], name=f"ep{i:06d}", hz=hz)
    if cap is not None:
        cap.release()
