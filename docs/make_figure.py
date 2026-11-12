"""Make docs/pose2sim_demo.png from the trial in tests/data. Run from the repository root:

python docs/make_figure.py
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

from pose2sim_to_nwb.converter import convert_trial
from pose2sim_to_nwb.skeleton import KEYPOINT_NAMES, SKELETON_EDGES

out = Path("docs")
p = convert_trial(
    Path("tests/data/S01_Demo_SingleTrial"),
    Path(tempfile.mkdtemp()),
    datetime(2024, 1, 1, tzinfo=timezone.utc),
    overwrite=True,
)
io = NWBHDF5IO(str(p), "r", load_namespaces=True)
nwb = io.read()
beh = nwb.processing["behavior"]
mc = beh["MultiCameraPoseEstimation"]


def arr(pe, dims):
    return np.stack([np.asarray(pe.pose_estimation_series[n].data[:])[:, :dims] for n in KEYPOINT_NAMES], axis=1)


P3 = arr(mc, 3)  # T,K,3
T = P3.shape[0]
_ch = {c.name: c for c in mc.children}
cams = [arr(_ch[f"PoseEstimation_cam_0{i}"], 2) for i in range(1, 5)]
f = 50
blue, grey, orange = "#2f6db5", "#9aa5b1", "#d9822b"
fig = plt.figure(figsize=(13, 5.2), dpi=130)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 2.3, 0.05], wspace=0.25, hspace=0.3)
for i, c in enumerate(cams):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    xy = c[f]
    for a, b in SKELETON_EDGES:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=blue, lw=1.6)
    ax.scatter(xy[:, 0], xy[:, 1], s=7, color=blue, zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"cam_0{i + 1}: OpenPose 2D", fontsize=9, color="#333")
    m = np.nanmean(xy, 0)
    r = np.nanmax(np.nanmax(xy, 0) - np.nanmin(xy, 0)) * 0.65
    ax.set_xlim(m[0] - r, m[0] + r)
    ax.set_ylim(m[1] + r, m[1] - r)
    for s_ in ax.spines.values():
        s_.set_color("#ccc")
ax = fig.add_subplot(gs[:, 2], projection="3d")
frames = list(range(0, T, 25))
for fr in frames:
    p = P3[fr]
    col = orange if fr == f else grey
    a_ = 1.0 if fr == f else 0.35
    for a, b in SKELETON_EDGES:
        ax.plot(
            p[[a, b], 0], p[[a, b], 1], p[[a, b], 2], color=col, lw=2 if fr == f else 1, alpha=a_ if fr == f else 0.5
        )
if f not in frames:
    p = P3[f]
    for a, b in SKELETON_EDGES:
        ax.plot(p[[a, b], 0], p[[a, b], 1], p[[a, b], 2], color=orange, lw=2)
pts = P3.reshape(-1, 3)
lo = np.nanmin(pts, 0)
hi = np.nanmax(pts, 0)
c0 = (lo + hi) / 2
r = (hi - lo).max() / 2
ax.set_xlim(c0[0] - r, c0[0] + r)
ax.set_ylim(c0[1] - r, c0[1] + r)
ax.set_zlim(c0[2] - r, c0[2] + r)
ax.set_xlabel("x (m)", fontsize=8)
ax.set_ylabel("y (m)", fontsize=8)
ax.set_zlabel("z (m)", fontsize=8)
ax.tick_params(labelsize=7)
ax.view_init(elev=12, azim=-60)
ax.set_title(
    f"Triangulated 3D keypoints in metres (frame {f} in orange, every 25th frame in grey)", fontsize=9, color="#333"
)
fig.suptitle(
    "One Pose2Sim demo trial after conversion: per-camera 2D detections and the 3D points stored in NWB",
    fontsize=11,
    x=0.5,
    y=0.99,
)
fig.savefig(out / "pose2sim_demo.png", bbox_inches="tight", facecolor="white")
print("Saved", out / "pose2sim_demo.png")
