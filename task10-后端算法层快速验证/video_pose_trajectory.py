#!/usr/bin/env python3
"""Extract MediaPipe Pose trajectories from videos.

Outputs per video:
  - pose_sequence.jsonl: frame-by-frame landmarks and features
  - features.csv: compact numeric feature trajectories
  - summary.json: video metadata, quality metrics, feature stats, motion hints
  - snapshots/: sampled frames, optionally with skeleton overlay
  - overlay.mp4: optional skeleton overlay video

Example:
  python video_pose_trajectory.py datasets/raw/self_recorded/videos/squat_positive_front_001.mp4 --draw-overlay
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

CORE_LANDMARKS = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

CSV_FEATURES = [
    "mean_visibility",
    "visible_landmark_count",
    "core_visible_ratio",
    "motion_energy",
    "shoulder_width",
    "hip_width",
    "torso_length",
    "hip_center_y",
    "hip_height_above_ankles_torso",
    "shoulder_height_above_ankles_torso",
    "foot_spread_shoulder_ratio",
    "wrist_spread_shoulder_ratio",
    "left_wrist_height_vs_shoulder",
    "right_wrist_height_vs_shoulder",
    "hands_above_shoulders_ratio",
    "left_knee_height_vs_hip",
    "right_knee_height_vs_hip",
    "left_knee_angle",
    "right_knee_angle",
    "mean_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
]


def import_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        missing = getattr(exc, "name", "unknown")
        print(
            "Missing dependency: "
            f"{missing}\n\n"
            "Install in your conda environment, for example:\n"
            "  pip install opencv-python mediapipe numpy\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return cv2, mp, np


def get_solutions_pose_module(mp: Any) -> Any | None:
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        pose_module = getattr(solutions, "pose", None)
        if pose_module is not None:
            return pose_module

    try:
        from mediapipe.python.solutions import pose as fallback_pose_module  # type: ignore
    except ImportError:
        return None

    return fallback_pose_module


class SolutionsPoseEstimator:
    def __init__(self, pose_module: Any, args: argparse.Namespace) -> None:
        self.pose_module = pose_module
        self.args = args
        self.pose = None

    def __enter__(self) -> "SolutionsPoseEstimator":
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=self.args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.args.min_detection_confidence,
            min_tracking_confidence=self.args.min_tracking_confidence,
        )
        self.pose.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.pose is not None:
            self.pose.__exit__(exc_type, exc, tb)

    def process(self, rgb_frame: Any, timestamp_ms: int) -> list[Any] | None:
        if self.pose is None:
            raise RuntimeError("Pose estimator is not initialized")
        result = self.pose.process(rgb_frame)
        if not result.pose_landmarks:
            return None
        return list(result.pose_landmarks.landmark)


class TasksPoseEstimator:
    def __init__(self, mp: Any, args: argparse.Namespace) -> None:
        self.mp = mp
        self.args = args
        self.landmarker = None

    def __enter__(self) -> "TasksPoseEstimator":
        model_path = Path(self.args.model_asset_path).expanduser().resolve()
        if not model_path.exists():
            print(
                "MediaPipe Tasks backend requires a Pose Landmarker model file.\n\n"
                f"Expected model path:\n  {model_path}\n\n"
                "Download the lite model with:\n"
                f"  mkdir -p {model_path.parent}\n"
                f"  wget -O {model_path} {DEFAULT_TASK_MODEL_URL}\n\n"
                "Then rerun the same command.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        try:
            BaseOptions = self.mp.tasks.BaseOptions
            PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = self.mp.tasks.vision.RunningMode
        except AttributeError as exc:
            try:
                from mediapipe.tasks import python  # type: ignore
                from mediapipe.tasks.python import vision  # type: ignore

                BaseOptions = python.BaseOptions
                PoseLandmarker = vision.PoseLandmarker
                PoseLandmarkerOptions = vision.PoseLandmarkerOptions
                VisionRunningMode = vision.RunningMode
            except ImportError as import_exc:
                print(
                    "Your mediapipe package exposes neither legacy Pose solutions nor "
                    "the current Tasks PoseLandmarker API.\n\n"
                    "Try reinstalling MediaPipe in this conda environment:\n"
                    "  pip uninstall mediapipe\n"
                    "  pip install mediapipe\n",
                    file=sys.stderr,
                )
                raise SystemExit(2) from import_exc

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.args.min_detection_confidence,
            min_pose_presence_confidence=self.args.min_detection_confidence,
            min_tracking_confidence=self.args.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.landmarker.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.landmarker is not None:
            self.landmarker.__exit__(exc_type, exc, tb)

    def process(self, rgb_frame: Any, timestamp_ms: int) -> list[Any] | None:
        if self.landmarker is None:
            raise RuntimeError("Pose estimator is not initialized")
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not getattr(result, "pose_landmarks", None):
            return None
        if not result.pose_landmarks:
            return None
        return list(result.pose_landmarks[0])


def create_pose_estimator(mp: Any, args: argparse.Namespace) -> tuple[Any, str]:
    if args.backend in {"auto", "solutions"}:
        pose_module = get_solutions_pose_module(mp)
        if pose_module is not None:
            return SolutionsPoseEstimator(pose_module, args), "mediapipe_solutions_pose"
        if args.backend == "solutions":
            raise SystemExit("MediaPipe legacy solutions Pose API is not available in this environment.")

    if args.backend in {"auto", "tasks"}:
        return TasksPoseEstimator(mp, args), "mediapipe_tasks_pose_landmarker"

    raise SystemExit(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks, trajectory features, snapshots, and optional overlay videos."
    )
    parser.add_argument("input", type=Path, help="Video file or directory containing videos.")
    parser.add_argument(
        "--backend",
        choices=["auto", "solutions", "tasks"],
        default="auto",
        help="Pose backend. auto uses legacy solutions if available, otherwise MediaPipe Tasks.",
    )
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task",
        help="MediaPipe Tasks Pose Landmarker .task model path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "pose" / "videos",
        help="Directory for extracted pose outputs.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end-sec", type=float, default=None, help="End time in seconds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames per video.")
    parser.add_argument("--label", default=None, help="Optional action label stored in summary metadata.")
    parser.add_argument("--view", default=None, help="Optional view label, e.g. front/side/diagonal.")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--resize-width", type=int, default=None, help="Resize frames before pose extraction.")
    parser.add_argument("--draw-overlay", action="store_true", help="Write overlay.mp4 with pose skeleton.")
    parser.add_argument("--no-snapshots", action="store_true", help="Disable sampled frame snapshots.")
    parser.add_argument(
        "--snapshot-interval-sec",
        type=float,
        default=1.0,
        help="Save one snapshot every N seconds. Use 0 to disable interval snapshots.",
    )
    parser.add_argument(
        "--snapshot-frames",
        default="",
        help="Comma separated original frame indexes to save, e.g. 0,45,90.",
    )
    return parser.parse_args()


def list_videos(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise SystemExit(f"Input file is not a supported video: {path}")
        return [path]

    if not path.is_dir():
        raise SystemExit(f"Input path does not exist: {path}")

    pattern = "**/*" if recursive else "*"
    videos = [p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def point(landmarks: dict[str, dict[str, float]], name: str, min_vis: float) -> tuple[float, float, float] | None:
    item = landmarks.get(name)
    if not item:
        return None
    if item.get("visibility", 0.0) < min_vis:
        return None
    return (float(item["x"]), float(item["y"]), float(item.get("z", 0.0)))


def distance_2d(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def angle_degrees(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
    c: tuple[float, float, float] | None,
) -> float | None:
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos_value = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def feature_mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_features(
    landmarks: dict[str, dict[str, float]],
    prev_landmarks: dict[str, dict[str, float]] | None,
    min_vis: float,
) -> dict[str, float | int | bool | None]:
    pts = {name: point(landmarks, name, min_vis) for name in LANDMARK_NAMES}

    left_shoulder = pts["LEFT_SHOULDER"]
    right_shoulder = pts["RIGHT_SHOULDER"]
    left_hip = pts["LEFT_HIP"]
    right_hip = pts["RIGHT_HIP"]
    left_knee = pts["LEFT_KNEE"]
    right_knee = pts["RIGHT_KNEE"]
    left_ankle = pts["LEFT_ANKLE"]
    right_ankle = pts["RIGHT_ANKLE"]
    left_wrist = pts["LEFT_WRIST"]
    right_wrist = pts["RIGHT_WRIST"]
    left_elbow = pts["LEFT_ELBOW"]
    right_elbow = pts["RIGHT_ELBOW"]

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    ankle_center = midpoint(left_ankle, right_ankle)

    shoulder_width = distance_2d(left_shoulder, right_shoulder)
    hip_width = distance_2d(left_hip, right_hip)
    torso_length = distance_2d(shoulder_center, hip_center)
    scale = torso_length or shoulder_width or hip_width

    foot_spread = None
    if left_ankle is not None and right_ankle is not None:
        foot_spread = abs(left_ankle[0] - right_ankle[0])

    wrist_spread = None
    if left_wrist is not None and right_wrist is not None:
        wrist_spread = abs(left_wrist[0] - right_wrist[0])

    left_wrist_height_vs_shoulder = None
    right_wrist_height_vs_shoulder = None
    if left_wrist is not None and left_shoulder is not None:
        left_wrist_height_vs_shoulder = ratio(left_shoulder[1] - left_wrist[1], scale)
    if right_wrist is not None and right_shoulder is not None:
        right_wrist_height_vs_shoulder = ratio(right_shoulder[1] - right_wrist[1], scale)

    left_knee_height_vs_hip = None
    right_knee_height_vs_hip = None
    if left_knee is not None and left_hip is not None:
        left_knee_height_vs_hip = ratio(left_hip[1] - left_knee[1], scale)
    if right_knee is not None and right_hip is not None:
        right_knee_height_vs_hip = ratio(right_hip[1] - right_knee[1], scale)

    motion_energy = None
    if prev_landmarks:
        motions = []
        for name in CORE_LANDMARKS:
            current = point(landmarks, name, min_vis)
            previous = point(prev_landmarks, name, min_vis)
            d = distance_2d(current, previous)
            if d is not None and scale:
                motions.append(d / scale)
        if motions:
            motion_energy = sum(motions) / len(motions)

    left_knee_angle = angle_degrees(left_hip, left_knee, left_ankle)
    right_knee_angle = angle_degrees(right_hip, right_knee, right_ankle)
    left_hip_angle = angle_degrees(left_shoulder, left_hip, left_knee)
    right_hip_angle = angle_degrees(right_shoulder, right_hip, right_knee)
    left_elbow_angle = angle_degrees(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_degrees(right_shoulder, right_elbow, right_wrist)

    vis_values = [item.get("visibility", 0.0) for item in landmarks.values()]
    visible_count = sum(1 for v in vis_values if v >= min_vis)
    core_visible_count = sum(
        1 for name in CORE_LANDMARKS if landmarks.get(name, {}).get("visibility", 0.0) >= min_vis
    )

    hands_above = []
    if left_wrist_height_vs_shoulder is not None:
        hands_above.append(1.0 if left_wrist_height_vs_shoulder > 0.0 else 0.0)
    if right_wrist_height_vs_shoulder is not None:
        hands_above.append(1.0 if right_wrist_height_vs_shoulder > 0.0 else 0.0)

    return {
        "mean_visibility": feature_mean(vis_values),
        "visible_landmark_count": visible_count,
        "core_visible_ratio": core_visible_count / len(CORE_LANDMARKS),
        "motion_energy": motion_energy,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "hip_center_y": hip_center[1] if hip_center else None,
        "hip_height_above_ankles_torso": ratio(ankle_center[1] - hip_center[1], scale)
        if ankle_center and hip_center
        else None,
        "shoulder_height_above_ankles_torso": ratio(ankle_center[1] - shoulder_center[1], scale)
        if ankle_center and shoulder_center
        else None,
        "foot_spread_shoulder_ratio": ratio(foot_spread, shoulder_width),
        "wrist_spread_shoulder_ratio": ratio(wrist_spread, shoulder_width),
        "left_wrist_height_vs_shoulder": left_wrist_height_vs_shoulder,
        "right_wrist_height_vs_shoulder": right_wrist_height_vs_shoulder,
        "hands_above_shoulders_ratio": feature_mean(hands_above),
        "left_knee_height_vs_hip": left_knee_height_vs_hip,
        "right_knee_height_vs_hip": right_knee_height_vs_hip,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "mean_knee_angle": feature_mean([left_knee_angle, right_knee_angle]),
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
    }


def feature_stats(records: list[dict[str, Any]], feature_name: str) -> dict[str, float | int | None]:
    values = [safe_float(r["features"].get(feature_name)) for r in records if r.get("pose_detected")]
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "range": None, "first": None, "last": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "first": values[0],
        "last": values[-1],
    }


def build_motion_hints(stats: dict[str, dict[str, float | int | None]]) -> dict[str, Any]:
    def value(feature: str, key: str) -> float | None:
        raw = stats.get(feature, {}).get(key)
        return safe_float(raw)

    hip_height_range = value("hip_height_above_ankles_torso", "range")
    knee_min = value("mean_knee_angle", "min")
    knee_range = value("mean_knee_angle", "range")
    foot_range = value("foot_spread_shoulder_ratio", "range")
    wrist_range = value("wrist_spread_shoulder_ratio", "range")
    hands_above_mean = value("hands_above_shoulders_ratio", "mean")
    motion_mean = value("motion_energy", "mean")

    squat_score = 0
    if hip_height_range is not None and hip_height_range >= 0.25:
        squat_score += 1
    if knee_min is not None and knee_min <= 125:
        squat_score += 1
    if knee_range is not None and knee_range >= 35:
        squat_score += 1

    jumping_jack_score = 0
    if foot_range is not None and foot_range >= 0.45:
        jumping_jack_score += 1
    if wrist_range is not None and wrist_range >= 0.70:
        jumping_jack_score += 1
    if hands_above_mean is not None and hands_above_mean >= 0.25:
        jumping_jack_score += 1
    if motion_mean is not None and motion_mean >= 0.015:
        jumping_jack_score += 1

    return {
        "not_a_classifier": True,
        "squat_like": {
            "score_0_to_3": squat_score,
            "hip_height_above_ankles_torso_range": hip_height_range,
            "mean_knee_angle_min": knee_min,
            "mean_knee_angle_range": knee_range,
            "interpretation": "higher score means the trajectory contains hip drop and knee flexion patterns",
        },
        "jumping_jack_like": {
            "score_0_to_4": jumping_jack_score,
            "foot_spread_shoulder_ratio_range": foot_range,
            "wrist_spread_shoulder_ratio_range": wrist_range,
            "hands_above_shoulders_ratio_mean": hands_above_mean,
            "motion_energy_mean": motion_mean,
            "interpretation": "higher score means the trajectory contains hand/foot opening patterns",
        },
    }


def resize_frame(cv2: Any, frame: Any, resize_width: int | None) -> Any:
    if not resize_width:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    scale = resize_width / float(width)
    return cv2.resize(frame, (resize_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def should_save_snapshot(
    frame_index: int,
    timestamp_sec: float,
    explicit_frames: set[int],
    interval_sec: float,
    next_interval: float,
) -> tuple[bool, float]:
    if frame_index in explicit_frames:
        return True, next_interval
    if interval_sec > 0 and timestamp_sec + 1e-9 >= next_interval:
        while timestamp_sec + 1e-9 >= next_interval:
            next_interval += interval_sec
        return True, next_interval
    return False, next_interval


def draw_landmarks(cv2: Any, frame: Any, landmarks: dict[str, dict[str, float]], min_vis: float) -> Any:
    if not landmarks:
        return frame

    points: dict[int, tuple[int, int]] = {}
    for name, item in landmarks.items():
        if item.get("visibility", 0.0) < min_vis:
            continue
        idx = int(item["index"])
        points[idx] = (int(round(item["px"])), int(round(item["py"])))

    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)

    for idx, pt in points.items():
        color = (0, 255, 0)
        if idx in {15, 16, 23, 24, 25, 26, 27, 28}:
            color = (0, 220, 255)
        cv2.circle(frame, pt, 3, color, -1)

    return frame


def make_output_dir(output_root: Path, video_path: Path) -> Path:
    stem = video_path.stem
    out = output_root / stem
    if out.exists():
        suffix = time.strftime("%Y%m%d_%H%M%S")
        out = output_root / f"{stem}_{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_video(args: argparse.Namespace, video_path: Path) -> Path:
    cv2, mp, _np = import_runtime_deps()
    estimator, backend_name = create_pose_estimator(mp, args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if args.start_sec and source_fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * source_fps))

    out_dir = make_output_dir(args.output_root, video_path)
    snapshots_dir = out_dir / "snapshots"
    if not args.no_snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "pose_sequence.jsonl"
    csv_path = out_dir / "features.csv"
    overlay_path = out_dir / "overlay.mp4"

    explicit_snapshot_frames = {
        int(x.strip()) for x in args.snapshot_frames.split(",") if x.strip().isdigit()
    }
    next_snapshot_sec = max(args.start_sec, 0.0)
    if args.snapshot_interval_sec > 0:
        next_snapshot_sec = math.floor(max(args.start_sec, 0.0) / args.snapshot_interval_sec) * args.snapshot_interval_sec

    records: list[dict[str, Any]] = []
    prev_landmarks: dict[str, dict[str, float]] | None = None
    processed_count = 0
    overlay_writer = None

    with estimator as pose_estimator, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            timestamp_sec = frame_index / source_fps if source_fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if args.end_sec is not None and timestamp_sec > args.end_sec:
                break
            if frame_index % max(1, args.frame_stride) != 0:
                continue
            if args.max_frames is not None and processed_count >= args.max_frames:
                break

            frame = resize_frame(cv2, frame, args.resize_width)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round(timestamp_sec * 1000.0))
            pose_landmark_list = pose_estimator.process(rgb, timestamp_ms)

            landmarks: dict[str, dict[str, float]] = {}
            if pose_landmark_list:
                for i, lm in enumerate(pose_landmark_list[: len(LANDMARK_NAMES)]):
                    name = LANDMARK_NAMES[i]
                    visibility = getattr(lm, "visibility", None)
                    if visibility is None:
                        visibility = getattr(lm, "presence", 1.0)
                    landmarks[name] = {
                        "index": i,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(visibility),
                        "presence": float(getattr(lm, "presence", visibility)),
                        "px": float(lm.x * width),
                        "py": float(lm.y * height),
                    }

            features = compute_features(landmarks, prev_landmarks, args.visibility_threshold) if landmarks else {}
            mean_visibility = safe_float(features.get("mean_visibility")) if features else None
            core_visible_ratio = safe_float(features.get("core_visible_ratio")) if features else None
            pose_detected = bool(landmarks)

            record = {
                "video": str(video_path),
                "frame_index": frame_index,
                "processed_index": processed_count,
                "timestamp_sec": timestamp_sec,
                "image_width": width,
                "image_height": height,
                "pose_detected": pose_detected,
                "pose_quality": {
                    "mean_visibility": mean_visibility,
                    "visible_landmark_count": features.get("visible_landmark_count") if features else 0,
                    "core_visible_ratio": core_visible_ratio,
                },
                "landmarks": landmarks,
                "features": features,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

            overlay_frame = frame.copy()
            if args.draw_overlay or not args.no_snapshots:
                overlay_frame = draw_landmarks(cv2, overlay_frame, landmarks, args.visibility_threshold)

            if args.draw_overlay:
                if overlay_writer is None:
                    output_fps = source_fps / max(1, args.frame_stride) if source_fps > 0 else 30.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, output_fps, (width, height))
                overlay_writer.write(overlay_frame)

            if not args.no_snapshots:
                save_snap, next_snapshot_sec = should_save_snapshot(
                    frame_index,
                    timestamp_sec,
                    explicit_snapshot_frames,
                    args.snapshot_interval_sec,
                    next_snapshot_sec,
                )
                if save_snap:
                    snapshot_name = f"frame_{frame_index:06d}_{timestamp_sec:.2f}s.jpg"
                    cv2.imwrite(str(snapshots_dir / snapshot_name), overlay_frame)

            prev_landmarks = landmarks if landmarks else prev_landmarks
            processed_count += 1

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["frame_index", "processed_index", "timestamp_sec", "pose_detected"] + CSV_FEATURES
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "frame_index": record["frame_index"],
                "processed_index": record["processed_index"],
                "timestamp_sec": f"{record['timestamp_sec']:.6f}",
                "pose_detected": record["pose_detected"],
            }
            for name in CSV_FEATURES:
                value = record.get("features", {}).get(name)
                row[name] = "" if value is None else value
            writer.writerow(row)

    stats = {name: feature_stats(records, name) for name in CSV_FEATURES}
    detected_frames = sum(1 for r in records if r.get("pose_detected"))
    duration_sec = 0.0
    if records:
        duration_sec = records[-1]["timestamp_sec"] - records[0]["timestamp_sec"]

    summary = {
        "source_video": str(video_path),
        "label": args.label,
        "view": args.view,
        "extractor": backend_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_video": {
            "fps": source_fps,
            "frame_count": source_frame_count,
            "width": source_width,
            "height": source_height,
        },
        "processing": {
            "frame_stride": args.frame_stride,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "max_frames": args.max_frames,
            "resize_width": args.resize_width,
            "processed_frames": len(records),
            "processed_duration_sec": duration_sec,
        },
        "pose_quality": {
            "detected_frames": detected_frames,
            "detected_ratio": detected_frames / len(records) if records else 0.0,
            "mean_visibility": stats["mean_visibility"]["mean"],
            "mean_core_visible_ratio": stats["core_visible_ratio"]["mean"],
        },
        "landmark_names": LANDMARK_NAMES,
        "feature_stats": stats,
        "motion_hints": build_motion_hints(stats),
        "outputs": {
            "pose_sequence_jsonl": str(jsonl_path),
            "features_csv": str(csv_path),
            "summary_json": str(out_dir / "summary.json"),
            "snapshots_dir": str(snapshots_dir) if not args.no_snapshots else None,
            "overlay_video": str(overlay_path) if args.draw_overlay else None,
        },
    }
    write_json(out_dir / "summary.json", summary)
    return out_dir


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    videos = list_videos(args.input, args.recursive)
    if not videos:
        raise SystemExit(f"No videos found: {args.input}")

    outputs = []
    for video in videos:
        print(f"Processing: {video}")
        out_dir = process_video(args, video)
        outputs.append(out_dir)
        print(f"  output: {out_dir}")

    print("Done.")
    for out in outputs:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
