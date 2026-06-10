#!/usr/bin/env python3
"""Render action-state overlays from existing Task10 pose/state outputs.

This script does not run pose detection again. It reads:
  - summary.json
  - pose_sequence.jsonl
  - state_summary_<action>.json
  - state_sequence_<action>.jsonl

and writes action_overlay_<action>.mp4 beside each sample output. Existing
videos are preserved by default; repeated renders create timestamped files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "datasets" / "pose" / "videos"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

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

REPORT_COLUMNS = [
    "sample_id",
    "action",
    "expected_positive",
    "predicted_positive",
    "status",
    "supported",
    "summary_rules_passed",
    "cycle_count",
    "max_score",
    "output_video",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render action-state overlays for Task10 samples.")
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--action",
        default="auto",
        help="Action to render, or auto. Known current actions: squat, jumping_jack.",
    )
    parser.add_argument("--all-actions", action="store_true", help="Render every available state summary per sample.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated no-op kept for old commands. Use --force-overwrite only when replacement is intentional.",
    )
    parser.add_argument("--force-overwrite", action="store_true", help="Replace existing overlay videos and reports.")
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--reason-max-chars", type=int, default=96)
    parser.add_argument("--sample-contains", default="", help="Only render samples whose output directory name contains this text.")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--name", default="action_overlay_report")
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        print(
            "Missing dependency: opencv-python\n\n"
            "Install it in your conda environment:\n"
            "  pip install opencv-python\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return cv2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_output_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root does not exist: {root}")
    return sorted({p.parent for p in root.glob("*/pose_sequence.jsonl")})


def resolve_source_video(summary: dict[str, Any]) -> Path:
    raw = str(summary.get("source_video") or "")
    if not raw:
        raise RuntimeError("summary.json does not contain source_video")
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def available_actions(output_dir: Path) -> dict[str, Path]:
    actions = {}
    for path in sorted(output_dir.glob("state_summary_*.json")):
        action = path.stem.removeprefix("state_summary_")
        actions[action] = path
    return actions


def choose_actions(output_dir: Path, summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    actions = available_actions(output_dir)
    if not actions:
        return []
    if args.all_actions:
        return sorted(actions)
    if args.action != "auto":
        return [args.action] if args.action in actions else []

    label = str(summary.get("label") or "")
    if label in actions:
        return [label]

    predicted = []
    for action, path in actions.items():
        state_summary = read_json(path)
        if state_summary.get("predicted_positive"):
            predicted.append(action)
    if predicted:
        return sorted(predicted)
    return sorted(actions)


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


def output_fps(pose_rows: list[dict[str, Any]], summary: dict[str, Any]) -> float:
    timestamps = [safe_float(row.get("timestamp_sec")) for row in pose_rows]
    timestamps = [x for x in timestamps if x is not None]
    deltas = [
        b - a
        for a, b in zip(timestamps, timestamps[1:])
        if b is not None and a is not None and b > a
    ]
    if deltas:
        return max(1.0, min(120.0, 1.0 / statistics.median(deltas)))

    fps = safe_float(summary.get("input_video", {}).get("fps"))
    stride = int(summary.get("processing", {}).get("frame_stride") or 1)
    if fps and fps > 0:
        return fps / max(1, stride)
    return 30.0


def stage_color(stage: str, predicted: bool, supported: bool, summary_passed: bool) -> tuple[int, int, int]:
    if not supported:
        return (180, 180, 180)
    if not summary_passed:
        return (70, 130, 255)
    if predicted:
        return (70, 220, 70)
    if stage in {"down", "open", "extended", "left_up", "right_up"}:
        return (0, 220, 255)
    return (255, 210, 70)


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"_{stamp}" if index == 0 else f"_{stamp}_{index:03d}"
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to create unique output path for {path}")


def report_paths(report_dir: Path, name: str, force_overwrite: bool) -> tuple[Path, Path]:
    csv_path = report_dir / f"{name}.csv"
    md_path = report_dir / f"{name}.md"
    if force_overwrite or (not csv_path.exists() and not md_path.exists()):
        return csv_path, md_path

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"_{stamp}" if index == 0 else f"_{stamp}_{index:03d}"
        candidate_csv = report_dir / f"{name}{suffix}.csv"
        candidate_md = report_dir / f"{name}{suffix}.md"
        if not candidate_csv.exists() and not candidate_md.exists():
            return candidate_csv, candidate_md
    raise RuntimeError(f"Failed to create unique report path in {report_dir}")


def draw_landmarks(cv2: Any, frame: Any, landmarks: dict[str, Any], min_visibility: float) -> None:
    points: dict[int, tuple[int, int]] = {}
    for item in landmarks.values():
        if safe_float(item.get("visibility")) is not None and float(item.get("visibility")) < min_visibility:
            continue
        idx = int(item["index"])
        points[idx] = (int(round(float(item["px"]))), int(round(float(item["py"]))))

    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)

    for idx, pt in points.items():
        color = (0, 255, 0)
        if idx in {15, 16, 23, 24, 25, 26, 27, 28}:
            color = (0, 220, 255)
        cv2.circle(frame, pt, 3, color, -1)


def draw_text_fit(
    cv2: Any,
    frame: Any,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    color: tuple[int, int, int],
    font_scale: float = 0.58,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = font_scale
    while scale > 0.34:
        size, _baseline = cv2.getTextSize(text, font, scale, thickness)
        if size[0] <= max_width:
            break
        scale -= 0.04
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(
    cv2: Any,
    frame: Any,
    state: dict[str, Any],
    state_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    h, w = frame.shape[:2]
    panel_h = 118 if h >= 420 else 96
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    action = str(state_summary.get("spec_action") or state.get("action") or "")
    supported = bool(state_summary.get("supported"))
    summary_passed = bool(state_summary.get("summary_rules_passed"))
    predicted = bool(state_summary.get("predicted_positive"))
    expected = bool(state_summary.get("expected_positive"))
    cycle_count = int(state_summary.get("cycle_count") or 0)
    max_score = safe_float(state_summary.get("max_score"))
    score = safe_float(state.get("score"))
    stage = str(state.get("stage") or "unknown")

    if not supported:
        verdict = "UNSUPPORTED VIEW"
    elif predicted:
        verdict = "DETECTED"
    elif not summary_passed:
        verdict = "REJECTED BY RULES"
    else:
        verdict = "NO COMPLETE CYCLE"

    color = stage_color(stage, predicted, supported, summary_passed)
    draw_text_fit(
        cv2,
        frame,
        f"{action} | {verdict} | expected={expected} predicted={predicted}",
        (14, 28),
        w - 28,
        color,
        0.68,
        2,
    )
    max_score_text = "n/a" if max_score is None else f"{max_score:.2f}"
    score_text = "n/a" if score is None else f"{score:.2f}"
    draw_text_fit(
        cv2,
        frame,
        f"stage={stage} score={score_text} max={max_score_text} cycles={cycle_count}",
        (14, 56),
        w - 28,
        (235, 235, 235),
    )

    reason = truncate(str(state_summary.get("summary_reason") or ""), args.reason_max_chars)
    if reason:
        draw_text_fit(cv2, frame, f"rule: {reason}", (14, 84), w - 28, (210, 210, 210), 0.50)

    bar_x, bar_y = 14, min(panel_h - 18, h - 18)
    bar_w = min(360, max(80, w - 28))
    bar_h = 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * clamp01(score)), bar_y + bar_h),
        color,
        -1,
    )


def blank_frame(cv2: Any, width: int, height: int) -> Any:
    import numpy as np  # type: ignore

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "source frame unavailable",
        (20, max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return frame


def render_one(
    cv2: Any,
    output_dir: Path,
    action: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    state_summary_path = output_dir / f"state_summary_{action}.json"
    state_sequence_path = output_dir / f"state_sequence_{action}.jsonl"
    if not state_summary_path.exists() or not state_sequence_path.exists():
        return {
            "sample_id": output_dir.name,
            "action": action,
            "note": "missing state output",
        }

    base_out_path = output_dir / f"action_overlay_{action}.mp4"
    out_path = base_out_path if args.force_overwrite else unique_path(base_out_path)
    note = "rendered"
    if out_path != base_out_path:
        note = "rendered versioned"

    pose_rows = iter_jsonl(output_dir / "pose_sequence.jsonl")
    state_rows = {
        int(row["frame_index"]): row
        for row in iter_jsonl(state_sequence_path)
    }
    state_summary = read_json(state_summary_path)
    source_video = resolve_source_video(summary)
    cap = cv2.VideoCapture(str(source_video))
    source_available = cap.isOpened()

    first_row = pose_rows[0] if pose_rows else {}
    width = int(first_row.get("image_width") or summary.get("input_video", {}).get("width") or 640)
    height = int(first_row.get("image_height") or summary.get("input_video", {}).get("height") or 480)
    fps = output_fps(pose_rows, summary)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        if source_available:
            cap.release()
        raise RuntimeError(f"Failed to create video writer: {out_path}")

    for row in pose_rows:
        frame_index = int(row["frame_index"])
        if source_available:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
        else:
            ok, frame = False, None

        if not ok or frame is None:
            frame = blank_frame(cv2, width, height)
        elif frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        landmarks = row.get("landmarks") or {}
        draw_landmarks(cv2, frame, landmarks, args.min_visibility)
        state = state_rows.get(frame_index, {"stage": "unknown", "score": None, "action": action})
        draw_panel(cv2, frame, state, state_summary, args)
        writer.write(frame)

    writer.release()
    if source_available:
        cap.release()

    return {
        "sample_id": output_dir.name,
        "action": action,
        "expected_positive": state_summary.get("expected_positive"),
        "predicted_positive": state_summary.get("predicted_positive"),
        "status": state_summary.get("status"),
        "supported": state_summary.get("supported"),
        "summary_rules_passed": state_summary.get("summary_rules_passed"),
        "cycle_count": state_summary.get("cycle_count"),
        "max_score": state_summary.get("max_score"),
        "output_video": str(out_path),
        "note": note,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Task10 Action Overlay Report",
        "",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Rows: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('sample_id')}` action={row.get('action')} "
            f"expected={row.get('expected_positive')} predicted={row.get('predicted_positive')} "
            f"cycles={row.get('cycle_count')} note={row.get('note')} "
            f"video=`{row.get('output_video', '')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cv2 = import_cv2()
    rows: list[dict[str, Any]] = []
    rendered_count = 0
    for output_dir in find_output_dirs(args.input_root):
        if args.sample_contains and args.sample_contains not in output_dir.name:
            continue
        if args.max_videos is not None and rendered_count >= args.max_videos:
            break
        summary = read_json(output_dir / "summary.json")
        actions = choose_actions(output_dir, summary, args)
        for action in actions:
            if args.max_videos is not None and rendered_count >= args.max_videos:
                break
            print(f"Rendering: {output_dir.name} [{action}]")
            row = render_one(cv2, output_dir, action, args)
            rows.append(row)
            rendered_count += 1

    csv_path, md_path = report_paths(args.report_dir, args.name, args.force_overwrite)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Rendered rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
