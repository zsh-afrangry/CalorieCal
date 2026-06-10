#!/usr/bin/env python
"""
Offline probe for MMPose whole-body inference.

The script is intentionally independent from the frontend. It validates whether
the Python whole-body chain can output stable 133-point JSON before any browser
integration is considered.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


KEYPOINT_GROUPS: Tuple[Tuple[str, int, int], ...] = (
    ("body", 0, 17),
    ("left_foot", 17, 20),
    ("right_foot", 20, 23),
    ("face", 23, 91),
    ("left_hand", 91, 112),
    ("right_hand", 112, 133),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMPose whole-body inference on an image/video and write compact JSON."
    )
    parser.add_argument(
        "--input",
        help="Path to an image or video. Omit with --self-test to run a blank-frame dependency check.",
    )
    parser.add_argument(
        "--output",
        default="task8-更详细人体识别方案研究/wholebody_probe_output.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="MMPose device, for example cpu, cuda, or cuda:0. Use cpu first if RTX sm_120 is unsupported.",
    )
    parser.add_argument(
        "--pose2d",
        default="wholebody",
        help="MMPose Inferencer pose2d alias or config. Default: wholebody.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum frames to process for videos.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="Read every Nth frame for videos.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint score threshold used only for summary statistics.",
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.3,
        help="Detector threshold passed to MMPose Inferencer.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save MMPose visualization images into <output stem>_vis.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a single blank frame to verify the dependency chain.",
    )
    return parser.parse_args()


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def read_image(path: Path) -> np.ndarray:
    # cv2.imread can fail on non-ASCII Windows paths. imdecode + fromfile is safer.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def has_non_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def open_video_capture(path: Path) -> Tuple[cv2.VideoCapture, Optional[Path]]:
    """Open video, copying to an ASCII temp path if OpenCV dislikes Unicode paths."""
    temp_path: Optional[Path] = None
    source = path

    def make_temp_path() -> Path:
        import os

        fd, temp_name = tempfile.mkstemp(suffix=path.suffix or ".mp4")
        os.close(fd)
        temp = Path(temp_name)
        temp.unlink(missing_ok=True)
        return temp

    if has_non_ascii(str(path)):
        temp_path = make_temp_path()
        shutil.copyfile(path, temp_path)
        source = temp_path

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened() and temp_path is None:
        temp_path = make_temp_path()
        shutil.copyfile(path, temp_path)
        cap = cv2.VideoCapture(str(temp_path))

    return cap, temp_path


def load_frames(input_path: Optional[Path], self_test: bool, max_frames: int, frame_stride: int) -> List[Dict[str, Any]]:
    if self_test:
        return [
            {
                "frameIndex": 0,
                "sourceTimeMs": None,
                "image": np.zeros((240, 320, 3), dtype=np.uint8),
            }
        ]

    if input_path is None:
        raise ValueError("--input is required unless --self-test is set")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if is_image_path(input_path):
        image = read_image(input_path)
        return [{"frameIndex": 0, "sourceTimeMs": None, "image": image}]

    if not is_video_path(input_path):
        raise ValueError(f"Unsupported input type: {input_path.suffix}")

    cap, temp_path = open_video_capture(input_path)
    if not cap.isOpened():
        if temp_path:
            temp_path.unlink(missing_ok=True)
        raise ValueError(f"Failed to open video: {input_path}")

    frames: List[Dict[str, Any]] = []
    try:
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        stride = max(1, frame_stride)

        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frames.append(
                    {
                        "frameIndex": frame_index,
                        "sourceTimeMs": (frame_index / fps * 1000) if fps > 0 else None,
                        "image": frame,
                    }
                )
            frame_index += 1
    finally:
        cap.release()
        if temp_path:
            temp_path.unlink(missing_ok=True)

    if not frames:
        raise ValueError(f"No frames read from video: {input_path}")
    return frames


def score_stats(scores: Sequence[float], start: int, end: int, threshold: float) -> Dict[str, Any]:
    group_scores = [float(v) for v in scores[start:end] if v is not None]
    if not group_scores:
        return {
            "count": max(0, end - start),
            "valid": 0,
            "mean": None,
            "max": None,
        }
    valid = sum(1 for value in group_scores if value >= threshold)
    return {
        "count": len(group_scores),
        "valid": valid,
        "mean": round(float(sum(group_scores) / len(group_scores)), 4),
        "max": round(float(max(group_scores)), 4),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def summarize_instance(instance: Dict[str, Any], kpt_thr: float) -> Dict[str, Any]:
    keypoints = instance.get("keypoints") or []
    scores = instance.get("keypoint_scores") or []
    total = len(keypoints)
    score_values = [float(v) for v in scores if v is not None]
    valid = sum(1 for value in score_values if value >= kpt_thr)
    groups = {
        name: score_stats(score_values, start, end, kpt_thr)
        for name, start, end in KEYPOINT_GROUPS
    }

    return {
        "bbox": to_jsonable(instance.get("bbox")),
        "bboxScore": to_jsonable(instance.get("bbox_score")),
        "keypointCount": total,
        "validKeypoints": valid,
        "keypointScoreMean": round(float(sum(score_values) / len(score_values)), 4) if score_values else None,
        "keypointScoreMax": round(float(max(score_values)), 4) if score_values else None,
        "groups": groups,
        "keypoints": to_jsonable(keypoints),
        "keypointScores": to_jsonable(scores),
    }


def compact_result(
    result: Dict[str, Any],
    frame_meta: Dict[str, Any],
    elapsed_ms: float,
    kpt_thr: float,
) -> Dict[str, Any]:
    predictions = result.get("predictions") or []
    frame_predictions: Iterable[Dict[str, Any]]
    if predictions and isinstance(predictions[0], list):
        frame_predictions = predictions[0]
    else:
        frame_predictions = predictions

    instances = [summarize_instance(instance, kpt_thr) for instance in frame_predictions]
    image = frame_meta["image"]
    return {
        "frameIndex": frame_meta["frameIndex"],
        "sourceTimeMs": frame_meta["sourceTimeMs"],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "elapsedMs": round(elapsed_ms, 2),
        "instanceCount": len(instances),
        "instances": instances,
    }


def draw_overlay(image: np.ndarray, frame: Dict[str, Any], kpt_thr: float) -> np.ndarray:
    colors = {
        "body": (60, 220, 255),
        "left_foot": (255, 180, 60),
        "right_foot": (255, 140, 90),
        "face": (120, 255, 120),
        "left_hand": (255, 120, 220),
        "right_hand": (180, 120, 255),
    }
    output = image.copy()

    for instance in frame["instances"]:
        bbox = instance.get("bbox")
        if isinstance(bbox, list) and bbox:
            box = bbox[0] if isinstance(bbox[0], list) else bbox
            if len(box) >= 4:
                x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
                cv2.rectangle(output, (x1, y1), (x2, y2), (80, 240, 80), 2)

        keypoints = instance.get("keypoints") or []
        scores = instance.get("keypointScores") or []
        for group_name, start, end in KEYPOINT_GROUPS:
            color = colors[group_name]
            for index in range(start, min(end, len(keypoints))):
                score = float(scores[index]) if index < len(scores) else 0.0
                if score < kpt_thr:
                    continue
                point = keypoints[index]
                if not isinstance(point, list) or len(point) < 2:
                    continue
                x, y = int(round(float(point[0]))), int(round(float(point[1])))
                cv2.circle(output, (x, y), 2, color, -1, lineType=cv2.LINE_AA)

    return output


def build_output_summary(frames: List[Dict[str, Any]], started_at: float, finished_at: float) -> Dict[str, Any]:
    elapsed_values = [frame["elapsedMs"] for frame in frames]
    instance_counts = [frame["instanceCount"] for frame in frames]
    keypoint_counts = [
        instance["keypointCount"]
        for frame in frames
        for instance in frame["instances"]
    ]
    mean_scores = [
        instance["keypointScoreMean"]
        for frame in frames
        for instance in frame["instances"]
        if instance["keypointScoreMean"] is not None
    ]
    return {
        "frameCount": len(frames),
        "totalElapsedMs": round((finished_at - started_at) * 1000, 2),
        "meanFrameInferMs": round(float(sum(elapsed_values) / len(elapsed_values)), 2) if elapsed_values else None,
        "maxFrameInferMs": round(float(max(elapsed_values)), 2) if elapsed_values else None,
        "meanInstances": round(float(sum(instance_counts) / len(instance_counts)), 2) if instance_counts else 0,
        "maxKeypointCount": max(keypoint_counts) if keypoint_counts else 0,
        "meanKeypointScore": round(float(sum(mean_scores) / len(mean_scores)), 4) if mean_scores else None,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    input_path = Path(args.input) if args.input else None

    try:
        from mmpose.apis import MMPoseInferencer
        import torch
        import mmpose
        import mmcv
        import mmengine
        import mmdet
        import numpy
    except Exception as exc:  # pragma: no cover - this is a probe script
        raise SystemExit(f"Failed to import MMPose stack: {exc}") from exc

    frames = load_frames(input_path, args.self_test, args.max_frames, args.frame_stride)
    vis_dir = output_path.with_name(f"{output_path.stem}_vis") if args.save_vis else None

    inferencer = MMPoseInferencer(args.pose2d, device=args.device, show_progress=False)

    compact_frames: List[Dict[str, Any]] = []
    started_at = time.perf_counter()

    for frame_meta in frames:
        frame_started = time.perf_counter()
        result = next(
            inferencer(
                frame_meta["image"],
                return_datasamples=False,
                return_vis=False,
                draw_bbox=True,
                bbox_thr=args.bbox_thr,
                kpt_thr=args.kpt_thr,
            )
        )
        elapsed_ms = (time.perf_counter() - frame_started) * 1000
        compact_frame = compact_result(result, frame_meta, elapsed_ms, args.kpt_thr)
        compact_frames.append(compact_frame)

        if vis_dir:
            overlay = draw_overlay(frame_meta["image"], compact_frame, args.kpt_thr)
            write_image(vis_dir / f"frame_{frame_meta['frameIndex']:06d}.jpg", overlay)

    finished_at = time.perf_counter()

    payload = {
        "ok": True,
        "input": str(input_path) if input_path else None,
        "selfTest": bool(args.self_test),
        "device": args.device,
        "pose2d": args.pose2d,
        "thresholds": {
            "bboxThr": args.bbox_thr,
            "keypointThr": args.kpt_thr,
        },
        "environment": {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "torchCudaAvailable": bool(torch.cuda.is_available()),
            "torchDeviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torchDeviceCapability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "mmcv": mmcv.__version__,
            "mmengine": mmengine.__version__,
            "mmdet": mmdet.__version__,
            "mmpose": mmpose.__version__,
        },
        "keypointLayout": {
            "totalExpected": 133,
            "groups": [
                {"name": name, "start": start, "endExclusive": end, "count": end - start}
                for name, start, end in KEYPOINT_GROUPS
            ],
        },
        "summary": build_output_summary(compact_frames, started_at, finished_at),
        "frames": compact_frames,
    }

    write_json(output_path, payload)
    print(f"ok=True output={output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
