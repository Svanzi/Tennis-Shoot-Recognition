import cv2
import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from tqdm import tqdm

##########################################################################################################
# Script to annotate a video with body-only landmarks using MediaPipe Pose,
# and export a CSV file with the landmark coordinates per frame.
# Usage:
#   python Landmarker.py input_video.mp4
# Outputs:
#   - Annotated video in ./Landmarked/input_video_Landmarked.mp4 
#   - CSV file with landmark coordinates in ./Landmarked/input_video_Landmarked.csv
##########################################################################################################


# --------- Constants & Types ---------
# OpenCV uses BGR color order (NOT RGB).
ColorBGR = Tuple[int, int, int]
Connection = Tuple[PoseLandmark, PoseLandmark, ColorBGR]

# Body landmark indices (no face)
BODY_LANDMARKS: List[PoseLandmark] = [
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,      PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,     PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_HEEL,     PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.RIGHT_FOOT_INDEX,
]
BODY_IDX: List[int] = [lm.value for lm in BODY_LANDMARKS]

# Body-only connections (no face), with OpenCV BGR colors
BODY_CONNECTIONS: List[Connection] = [
    # Torso / shoulders / hips -> white
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.RIGHT_SHOULDER,  (255, 255, 255)),
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.LEFT_HIP,        (255, 255, 255)),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP,       (255, 255, 255)),
    (PoseLandmark.LEFT_HIP,       PoseLandmark.RIGHT_HIP,       (255, 255, 255)),

    # Left arm -> blue
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, (255, 0, 0)),
    (PoseLandmark.LEFT_ELBOW,    PoseLandmark.LEFT_WRIST, (255, 0, 0)),

    # Right arm -> red
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, (0, 0, 255)),
    (PoseLandmark.RIGHT_ELBOW,    PoseLandmark.RIGHT_WRIST, (0, 0, 255)),

    # Left leg -> green
    (PoseLandmark.LEFT_HIP,   PoseLandmark.LEFT_KNEE,       (0, 255, 0)),
    (PoseLandmark.LEFT_KNEE,  PoseLandmark.LEFT_ANKLE,      (0, 255, 0)),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,       (0, 255, 0)),
    (PoseLandmark.LEFT_HEEL,  PoseLandmark.LEFT_FOOT_INDEX, (0, 255, 0)),

    # Right leg -> yellow (BGR)
    (PoseLandmark.RIGHT_HIP,   PoseLandmark.RIGHT_KNEE,       (0, 255, 255)),
    (PoseLandmark.RIGHT_KNEE,  PoseLandmark.RIGHT_ANKLE,      (0, 255, 255)),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,       (0, 255, 255)),
    (PoseLandmark.RIGHT_HEEL,  PoseLandmark.RIGHT_FOOT_INDEX, (0, 255, 255)),
]


# --------- Core helpers ---------
def draw_body_only(
    frame,
    landmarks: Sequence,
    img_w: int,
    img_h: int,
    visibility_thr: float = 0.5,
    point_radius: int = 3,
    point_color: ColorBGR = (0, 255, 0),  # green dots
    point_thickness: int = -1,
    line_thickness: int = 2,
) -> None:
    """
    Draw body-only landmarks and connections using OpenCV.
    - `landmarks` is the list from MediaPipe (result.pose_landmarks.landmark)
    - Silent no-op if visibility is low or points are off-frame.
    """
    # Points
    for idx in BODY_IDX:
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < visibility_thr:
            continue
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(frame, (x, y), point_radius, point_color, point_thickness)

    # Lines
    for a, b, color in BODY_CONNECTIONS:
        la, lb = landmarks[a.value], landmarks[b.value]
        if getattr(la, "visibility", 1.0) < visibility_thr:
            continue
        if getattr(lb, "visibility", 1.0) < visibility_thr:
            continue
        ax, ay = int(la.x * img_w), int(la.y * img_h)
        bx, by = int(lb.x * img_w), int(lb.y * img_h)
        if (0 <= ax < img_w and 0 <= ay < img_h and
            0 <= bx < img_w and 0 <= by < img_h):
            cv2.line(frame, (ax, ay), (bx, by), color, line_thickness)


def dump_body_to_csv(
    landmarks: Sequence,
    frame_idx: int,
    csv_writer: csv.writer
) -> None:
    """
    Write only body landmarks to CSV:
    columns: Frame, LandmarkIndex, BodyPartName, x, y, z, Visibility
    """
    for idx in BODY_IDX:
        lm = landmarks[idx]
        lm_name = PoseLandmark(idx).name
        visibility = getattr(lm, "visibility", 1.0)
        csv_writer.writerow([frame_idx, idx, lm_name, lm.x, lm.y, lm.z, visibility])


def open_video_writer(
    out_path: Path,
    fps: float,
    size: Tuple[int, int],
    fourcc: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create a cv2.VideoWriter and validate it's open. Raises RuntimeError if it fails.
    """
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc_code, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer


# --------- Main pipeline ---------
def main() -> None:
    parser = ArgumentParser(description="Annotate a video and export a CSV with body-only landmarks.")
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument(
        "--outdir", type=Path, default=Path("Landmarked"),
        help="Output directory for annotated video and CSV (default: ./Landmarked)"
    )
    parser.add_argument(
        "--visibility-thr", type=float, default=0.5,
        help="Visibility threshold for drawing/writing landmarks (default: 0.5)"
    )
    parser.add_argument(
        "--model-complexity", type=int, default=1, choices=[0, 1, 2],
        help="MediaPipe Pose model complexity (0,1,2). Higher = more accurate/slower. Default: 1"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Disable CSV export (only write annotated video)."
    )
    args = parser.parse_args()

    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    # Ensure output directory exists
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Derive output paths
    annotated_path: Path = outdir / f"{in_path.stem}_Landmarked.mp4"
    csv_path: Path = outdir / f"{in_path.stem}_Landmarks.csv"

    # Open input video
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")

    # Probe stream info
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        # Fallback if missing/invalid FPS metadata
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Open output writer (raises if fails)
    writer = open_video_writer(annotated_path, fps, (width, height), fourcc='mp4v')

    # Optional CSV (use context manager for auto-close)
    write_csv = not args.no_csv
    csv_file = None
    csv_writer = None
    if write_csv:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Landmark", "Body Part", "x Pos", "y Pos", "z Pos", "Visibility"])

    mp_pose = mp.solutions.pose

    # MediaPipe Pose context; use a single instance for the whole video
    with mp_pose.Pose(static_image_mode=False, model_complexity=args.model_complexity) as pose:
        frame_idx = 0
        # Use tqdm only if total_frames known (>0); otherwise, it still works but won’t show ETA.
        with tqdm(total=total_frames, desc="Video elaboration", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR->RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark

                    # Draw body landmarks (in-place)
                    draw_body_only(
                        frame, lms, width, height,
                        visibility_thr=args.visibility_thr
                    )

                    # Dump to CSV
                    if write_csv and csv_writer is not None:
                        dump_body_to_csv(lms, frame_idx, csv_writer)

                # Write annotated frame
                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

    # Clean-up
    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    print(f"\nVideo saved in: {annotated_path}")
    if write_csv:
        print(f"CSV saved in: {csv_path}")


if __name__ == "__main__":
    main()