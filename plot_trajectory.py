import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "positions" / "mouse_tracks_night.csv"

parser = argparse.ArgumentParser(description="Plot trajectories from SAM2 centroid CSV.")
parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to centroid CSV file.")
parser.add_argument("--fps", type=float, default=25.0, help="Frame rate used when converting frame index to seconds.")
args = parser.parse_args()

if not args.csv.exists():
    raise FileNotFoundError(f"CSV not found: {args.csv}")

df = pd.read_csv(args.csv)

if "frame_idx" not in df.columns or "cx" not in df.columns or "cy" not in df.columns:
    raise ValueError("CSV must contain frame_idx, cx, cy columns.")

# フレーム → 時間（秒）に変換
df["time"] = df["frame_idx"] / args.fps

# ============================================================
# 1. 横軸が時間のプロット（cx(t), cy(t)）
# ============================================================

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

palette = ["orange", "green", "blue", "purple", "red", "brown", "magenta", "cyan"]

for idx, (obj_id, g) in enumerate(df.groupby("object_id")):
    color = palette[idx % len(palette)]
    # x座標
    axes[0].plot(g["time"], g["cx"], label=f"id {obj_id}", color=color)
    # y座標
    axes[1].plot(g["time"], g["cy"], label=f"id {obj_id}", color=color)

axes[0].set_ylabel("x (pixels)")
axes[1].set_ylabel("y (pixels)")
axes[1].set_xlabel("time (s)")

axes[0].set_title("cx vs time")
axes[1].set_title("cy vs time")

axes[0].set_ylim(0,1920)
axes[1].set_ylim(0,1080)

axes[0].legend()
axes[0].grid(True)
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 2. 2D 平面上での移動軌跡（cx vs cy）
# ============================================================

plt.figure(figsize=(6, 6))

for idx, (obj_id, g) in enumerate(df.groupby("object_id")):
    color = palette[idx % len(palette)]
    plt.plot(g["cx"], g["cy"], marker=".", linestyle="-", label=f"id {obj_id}", color=color)

plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("2D trajectories")

plt.grid(True)
plt.legend()

plt.ylim(1080,0)
plt.xlim(0,1920)

plt.show()