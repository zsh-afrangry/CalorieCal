#!/usr/bin/env python
"""
Webcam probe for MMPose whole-body inference.

This closes the Task8 Phase 7 whole-body chain without integrating it into the
browser demo. It measures live webcam latency/FPS and can optionally show a
133-keypoint overlay in an OpenCV window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMPose whole-body inference on a webcam and write metrics JSON."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--device", default="cuda", help="MMPose device, for example cuda or cpu.")
    parser.add_argument("--pose2d", default="wholebody", help="MMPose pose2d alias or config.")
    parser.add_argument("--duration", type=float, default=10.0, help="Maximum capture duration in seconds.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum frames to process.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Frames ignored in post-warmup metrics.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint score threshold.")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Detector bbox threshold.")
    parser.add_argument("--show", action="store_true", help="Show live OpenCV overlay. Press q to stop.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam frames before inference/display.")
    parser.add_argument(
        "--output",
        default="task8-更详细人体识别方案研究/wholebody_webcam_cuda_rtx3060.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--save-snapshot",
        help="Optional path for one post-warmup overlay snapshot.",
    )
    return parser.parse_args()


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def summarize_frames(
    frames: List[Dict[str, Any]],
    started_at: float,
    finished_at: float,
    warmup_frames: int,
) -> Dict[str, Any]:
    warmup_count = min(max(0, warmup_frames), len(frames))
    measured = frames[warmup_count:] or frames
    elapsed_values = [float(frame["elapsedMs"]) for frame in measured]
    all_elapsed_values = [float(frame["elapsedMs"]) for frame in frames]
    instance_counts = [int(frame["instanceCount"]) for frame in measured]
    keypoint_counts = [
        int(instance["keypointCount"])
        for frame in measured
        for instance in frame["instances"]
    ]
    score_values = [
        float(instance["keypointScoreMean"])
        for frame in measured
        for instance in frame["instances"]
        if instance["keypointScoreMean"] is not None
    ]
    wall_seconds = max(finished_at - started_at, 1e-6)
    mean_infer = mean(elapsed_values)

    return {
        "frameCount": len(frames),
        "measuredFrameCount": len(measured),
        "warmupFrames": warmup_count,
        "totalWallMs": round(wall_seconds * 1000, 2),
        "wallFps": round(len(frames) / wall_seconds, 2),
        "meanInferMs": round_or_none(mean_infer),
        "p50InferMs": round_or_none(percentile(elapsed_values, 50)),
        "p95InferMs": round_or_none(percentile(elapsed_values, 95)),
        "maxInferMs": round_or_none(max(elapsed_values) if elapsed_values else None),
        "firstFrameInferMs": round_or_none(all_elapsed_values[0] if all_elapsed_values else None),
        "modelOnlyFpsEstimate": round_or_none(1000 / mean_infer if mean_infer and mean_infer > 0 else None),
        "meanInstances": round_or_none(mean(instance_counts), 2) or 0,
        "maxKeypointCount": max(keypoint_counts) if keypoint_counts else 0,
        "meanKeypointScore": round_or_none(mean(score_values), 4),
    }


def put_hud(
    image: Any,
    frame: Dict[str, Any],
    recent_fps: float,
    kpt_thr: float,
    cv2_module: Any,
) -> Any:
    output = image.copy()
    instance = frame["instances"][0] if frame["instances"] else None
    keypoints = instance["validKeypoints"] if instance else 0
    score = instance["keypointScoreMean"] if instance else None
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    lines = [
        f"infer {frame['elapsedMs']:.1f} ms  fps {recent_fps:.1f}",
        f"instances {frame['instanceCount']}  valid kpts {keypoints}/133  score {score_text}",
        f"kpt_thr {kpt_thr:.2f}",
    ]
    y = 24
    for line in lines:
        cv2_module.putText(
            output,
            line,
            (12, y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            4,
            cv2_module.LINE_AA,
        )
        cv2_module.putText(
            output,
            line,
            (12, y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
        y += 24
    return output


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    try:
        import cv2
        from mmpose.apis import MMPoseInferencer
        import mmcv
        import mmengine
        import mmdet
        import mmpose
        import numpy
        import torch
        from wholebody_offline_probe import compact_result, draw_overlay, write_image, write_json
    except Exception as exc:  # pragma: no cover - probe script
        raise SystemExit(f"Failed to import MMPose stack: {exc}") from exc

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    inferencer = MMPoseInferencer(args.pose2d, device=args.device, show_progress=False)
    compact_frames: List[Dict[str, Any]] = []
    snapshot_written = False
    started_at = time.perf_counter()
    recent_times: List[float] = []

    try:
        while len(compact_frames) < args.max_frames:
            now = time.perf_counter()
            if now - started_at >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_started = time.perf_counter()
            result = next(
                inferencer(
                    frame,
                    return_datasamples=False,
                    return_vis=False,
                    draw_bbox=True,
                    bbox_thr=args.bbox_thr,
                    kpt_thr=args.kpt_thr,
                )
            )
            elapsed_ms = (time.perf_counter() - frame_started) * 1000
            compact_frame = compact_result(
                result,
                {
                    "frameIndex": len(compact_frames),
                    "sourceTimeMs": (time.perf_counter() - started_at) * 1000,
                    "image": frame,
                },
                elapsed_ms,
                args.kpt_thr,
            )
            compact_frames.append(compact_frame)

            recent_times.append(time.perf_counter())
            recent_times = [value for value in recent_times if recent_times[-1] - value <= 1.0]
            recent_fps = float(len(recent_times))

            overlay = draw_overlay(frame, compact_frame, args.kpt_thr)
            overlay = put_hud(overlay, compact_frame, recent_fps, args.kpt_thr, cv2)

            if (
                args.save_snapshot
                and not snapshot_written
                and len(compact_frames) > max(0, args.warmup_frames)
            ):
                write_image(Path(args.save_snapshot), overlay)
                snapshot_written = True

            if args.show:
                cv2.imshow("Task8 WholeBody Webcam Probe", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        finished_at = time.perf_counter()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    payload = {
        "ok": True,
        "camera": args.camera,
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
        "summary": summarize_frames(compact_frames, started_at, finished_at, args.warmup_frames),
        "frames": compact_frames,
    }
    write_json(output_path, payload)
    print(f"ok=True output={output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
