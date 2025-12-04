import argparse
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "positions" / "mouse_tracks_night.csv"

parser = argparse.ArgumentParser(description="Plot trajectories from SAM2 centroid CSV.")
parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to centroid CSV file.")
parser.add_argument("--fps", type=float, default=25.0, help="Frame rate used when converting frame index to seconds.")
parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs", help="Directory to write plots/animations.")
parser.add_argument("--save-png", action="store_true", help="Save static PNG plots instead of showing.")
parser.add_argument("--save-anim", action="store_true", help="Save animated MP4 of trajectories.")
parser.add_argument("--frames-dir", type=Path, default=None, help="Optional frames directory to overlay trajectories on original frames.")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

if not args.csv.exists():
    raise FileNotFoundError(f"CSV not found: {args.csv}")

df = pd.read_csv(args.csv)

if "frame_idx" not in df.columns or "cx" not in df.columns or "cy" not in df.columns:
    raise ValueError("CSV must contain frame_idx, cx, cy columns.")

# フレーム → 時間（秒）に変換
df["time"] = df["frame_idx"] / args.fps

palette = ["orange", "green", "blue", "purple", "red", "brown", "magenta", "cyan"]

def save_static_plots(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    for idx, (obj_id, g) in enumerate(df.groupby("object_id")):
        color = palette[idx % len(palette)]
        axes[0].plot(g["time"], g["cx"], label=f"id {obj_id}", color=color)
        axes[1].plot(g["time"], g["cy"], label=f"id {obj_id}", color=color)

    axes[0].set_ylabel("x (pixels)")
    axes[1].set_ylabel("y (pixels)")
    axes[1].set_xlabel("time (s)")

    axes[0].set_title("cx vs time")
    axes[1].set_title("cy vs time")

    axes[0].set_ylim(0, 1920)
    axes[1].set_ylim(0, 1080)

    axes[0].legend()
    axes[0].grid(True)
    axes[1].grid(True)

    plt.tight_layout()
    out_path = out_dir / (args.csv.stem + "_time_series.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # 2D trajectory
    fig2 = plt.figure(figsize=(6, 6))
    for idx, (obj_id, g) in enumerate(df.groupby("object_id")):
        color = palette[idx % len(palette)]
        plt.plot(g["cx"], g["cy"], marker=".", linestyle="-", label=f"id {obj_id}", color=color)

    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("2D trajectories")
    plt.grid(True)
    plt.legend()
    plt.ylim(1080, 0)
    plt.xlim(0, 1920)
    out2 = out_dir / (args.csv.stem + "_2d.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    print(f"Saved static plots: {out_path}, {out2}")


def save_animation(df: pd.DataFrame, out_dir: Path, frames_dir: Path | None):
    max_frame = int(df["frame_idx"].max())
    nframes = max_frame + 1

    # build per-object position arrays indexed by frame
    objects = sorted(df["object_id"].unique())
    pos = {obj: {int(r.frame_idx): (r.cx, r.cy) for _, r in df[df["object_id"] == obj].iterrows()} for obj in objects}

    # background images if frames_dir provided
    bg_images = None
    if frames_dir and frames_dir.exists():
        # collect frame files sorted by filename
        files = sorted(frames_dir.iterdir())
        if len(files) >= nframes:
            bg_images = files
        else:
            bg_images = files  # still use what we have

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    scatters = {}
    trails = {}
    for idx, obj in enumerate(objects):
        color = palette[idx % len(palette)]
        scatters[obj] = ax.plot([], [], marker="o", color=color, markersize=6, label=f"id {obj}")[0]
        trails[obj] = ax.plot([], [], linestyle="-", color=color, alpha=0.6)[0]

    ax.legend()

    bg_im = None

    def init():
        for obj in objects:
            scatters[obj].set_data([], [])
            trails[obj].set_data([], [])
        return list(scatters.values()) + list(trails.values())

    # prepare trail data storage
    trail_data = {obj: ([], []) for obj in objects}

    def update(frame):
        nonlocal bg_im
        if bg_images and frame < len(bg_images):
            img = plt.imread(str(bg_images[frame]))
            if bg_im is None:
                bg_im = ax.imshow(img, extent=[0, 1920, 1080, 0])
            else:
                bg_im.set_data(img)

        for obj in objects:
            if frame in pos[obj]:
                x, y = pos[obj][frame]
                trail_data[obj][0].append(x)
                trail_data[obj][1].append(y)
                scatters[obj].set_data([x], [y])
                trails[obj].set_data(trail_data[obj][0], trail_data[obj][1])
            else:
                # keep last known position
                scatters[obj].set_data([], [])
        return list(scatters.values()) + list(trails.values())

    interval = 1000.0 / args.fps if args.fps > 0 else 40
    ani = animation.FuncAnimation(fig, update, frames=nframes, init_func=init, blit=False, interval=interval)

    out_path = out_dir / (args.csv.stem + "_traj.mp4")
    print(f"Saving animation to: {out_path} (this may take a while)")
    writer = animation.FFMpegWriter(fps=args.fps)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Saved animation: {out_path}")


if args.save_png:
    save_static_plots(df, args.out_dir)
else:
    # interactive: show as before
    save_static_plots(df, args.out_dir)
    plt.show()

if args.save_anim:
    save_animation(df, args.out_dir, args.frames_dir)
