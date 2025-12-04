# ============================================================
# PREPERATION
# ============================================================
import argparse
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

# select the device for computation
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# ============================================================
# INITIALIZING PREDICTOR
# ============================================================

from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parent
PROJECT_CONFIG_ROOT = ROOT / "sam2"
DEFAULT_VIDEO_MP4 = ROOT / "videos" / "D1AT_20251029_Trim_night_30sec.mp4"
DEFAULT_FRAMES_DIR = ROOT / "videos" / "night_30sec"
DEFAULT_CHECKPOINT = ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_OUTPUT_CSV = ROOT / "positions" / "mouse_tracks_night.csv"

parser = argparse.ArgumentParser(
    description="Track multiple objects in a video using SAM2 and save centroid trajectories."
)
parser.add_argument(
    "--video-mp4",
    type=Path,
    default=DEFAULT_VIDEO_MP4,
    help="Path to the source MP4 video. Used to extract frames when --frames-dir is empty.",
)
parser.add_argument(
    "--frames-dir",
    type=Path,
    default=DEFAULT_FRAMES_DIR,
    help="Directory containing JPEG frames (will be created from the video if missing).",
)
parser.add_argument(
    "--num-objects",
    type=int,
    default=2,
    help="Number of objects to click in the first frame.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=DEFAULT_CHECKPOINT,
    help="Path to the SAM2 checkpoint (.pt).",
)
parser.add_argument(
    "--config",
    type=str,
    default=DEFAULT_CONFIG,
    help="Hydra path or file path to the SAM2 config (.yaml).",
)
parser.add_argument(
    "--output-csv",
    type=Path,
    default=DEFAULT_OUTPUT_CSV,
    help="Path of the CSV file to write centroid tracks.",
)
parser.add_argument(
    "--ann-frame",
    type=int,
    default=0,
    help="Frame index used for manual clicks (default: 0).",
)

args = parser.parse_args()

if not args.checkpoint.exists():
    raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")
if args.num_objects < 1:
    raise ValueError("--num-objects must be >= 1")


def resolve_config_name(raw_config: str) -> str:
    """Convert CLI input into a Hydra-compatible config path."""

    config_path = Path(raw_config)

    if not config_path.is_absolute():
        return raw_config

    try:
        return str(config_path.resolve().relative_to(PROJECT_CONFIG_ROOT))
    except ValueError:
        pass

    try:
        package_root = resources.files("sam2")
        return str(config_path.resolve().relative_to(package_root))
    except (ModuleNotFoundError, ValueError) as exc:
        raise FileNotFoundError(
            f"SAM2 config not found via Hydra search paths: {raw_config}"
        ) from exc


def ensure_frames(frames_dir: Path, video_path: Path) -> None:
    """Create JPEG frames from video if none are present."""

    if frames_dir.exists():
        if any(frames_dir.glob("*.jpg")) or any(frames_dir.glob("*.jpeg")):
            return
    if not video_path.exists():
        raise FileNotFoundError(
            f"Frame directory {frames_dir} is empty and source video not found: {video_path}"
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    index = 0
    print(f"Extracting frames from {video_path} to {frames_dir} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = frames_dir / f"{index:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        index += 1
    cap.release()
    if index == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")
    print(f"Extracted {index} frames.")


ensure_frames(args.frames_dir, args.video_mp4)


def _frame_sort_key(path: Path):
    stem = path.stem
    return int(stem) if stem.isdigit() else stem


frame_paths = sorted(
    [
        p
        for p in args.frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    ],
    key=_frame_sort_key,
)

if not frame_paths:
    raise RuntimeError(f"No JPEG frames found under {args.frames_dir}")

print(f"Using {len(frame_paths)} frames from {args.frames_dir}")

config_name = resolve_config_name(args.config)
checkpoint_path = args.checkpoint.resolve()

predictor = build_sam2_video_predictor(config_name, str(checkpoint_path), device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# For your custom videos, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
#  ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
#  example: module load ffmpeg && ffmpeg -i /home/users/yuta.inaba/DEV/test/videos/D1AT_20251029_Trim_day_10min.mp4 -q:v 2 -start_number 0 videos/day_10min/'%05d.jpg'
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# initializing predictor
inference_state = predictor.init_state(video_path=str(args.frames_dir))

#Note: if you have run any previous tracking using this inference_state, please reset it first via reset_state.
#(The cell below is just for illustration; it's not needed to call reset_state here as this inference_state is just freshly initialized above.)

#predictor.reset_state(inference_state)

# ============================================================
# LABELING
# ============================================================

class Pointer:

    def __init__(self, img_path: Path):
        self.img_path = img_path
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise FileNotFoundError(f"Failed to load image for annotation: {img_path}")
        self.x = 0
        self.y = 0
        self._clicked = False

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.x = x
            self.y = y
            self._clicked = True

            self.img_marked = self.img.copy()
            cv2.circle(self.img_marked, (self.x, self.y), 8, (0, 0, 255), -1)
            cv2.imshow(str(self.img_path), self.img_marked)

    def point_gui(self):
        cv2.imshow(str(self.img_path), self.img)
        cv2.setMouseCallback(str(self.img_path), self.onMouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if not self._clicked:
            raise RuntimeError("No click detected. Please click on the object before closing the window.")

prompts = {} # hold all the clicks we add for visualization


## Pointing first mouse ##
ann_frame_idx = args.ann_frame  # the frame index we interact with

if ann_frame_idx < 0 or ann_frame_idx >= len(frame_paths):
    raise IndexError(
        f"Annotation frame index {ann_frame_idx} is outside available range [0, {len(frame_paths) - 1}]"
    )

for object_offset in range(args.num_objects):
    ann_obj_id = object_offset + 1
    pointer = Pointer(frame_paths[ann_frame_idx])
    pointer.point_gui()
    points = np.array([[pointer.x, pointer.y]], dtype=np.float32)

    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(frame_paths[ann_frame_idx]))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    plt.show()


# ============================================================
# SEGMENTATION
# ============================================================

# create path to save images
save_path = args.frames_dir.parent / f"{args.frames_dir.name}_segmented"
save_path.mkdir(parents=True, exist_ok=True)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# output segmented images
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in tqdm(range(0, len(frame_paths), vis_frame_stride), desc="output segmented image"):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(frame_paths[out_frame_idx]))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(save_path / f"s_{frame_paths[out_frame_idx].name}")
    plt.clf()
    plt.close()



# ============================================================
# CALCULATING COORDINATE OF MOUSE (not SAM2 function)
# ============================================================

from collections import defaultdict
import csv

# 各オブジェクトのトラック用：tracks[obj_id] = [(frame_idx, cx, cy), ...]
tracks = defaultdict(list)

for frame_idx, obj_dict in video_segments.items():
    for obj_id, mask in obj_dict.items():
        m = mask.squeeze()          # shape: (H, W)

        # マスクが空（検出なし）の場合はスキップ
        if m.sum() == 0:
            continue

        # マスク内の画素座標を取得
        ys, xs = np.nonzero(m)      # ys: 行(高さ), xs: 列(幅)

        # 重心を計算（float）
        cx = xs.mean()
        cy = ys.mean()

        # トラックに追加
        tracks[obj_id].append((frame_idx, float(cx), float(cy)))

args.output_csv.parent.mkdir(parents=True, exist_ok=True)

with args.output_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["object_id", "frame_idx", "cx", "cy"])
    for obj_id, traj in tracks.items():
        for frame_idx, cx, cy in traj:
            writer.writerow([obj_id, frame_idx, cx, cy])
    print(f"Saved centroid CSV to {args.output_csv}")

