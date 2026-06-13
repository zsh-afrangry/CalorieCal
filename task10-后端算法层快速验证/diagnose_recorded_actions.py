#!/usr/bin/env python3
"""Create diagnostics for a recorded mixed-action recognition output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "datasets" / "pose" / "recorded_actions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose recorded action recognition output.")
    parser.add_argument("output_dir", type=Path, help="Pose output dir containing features.csv and mixed_action_sequence.jsonl.")
    parser.add_argument("--name", default="mixed_action_diagnostics")
    parser.add_argument("--min-gap-sec", type=float, default=0.75)
    parser.add_argument("--squat-start-sec", type=float, default=13.0)
    parser.add_argument("--jumping-jack-end-sec", type=float, default=13.0)
    return parser.parse_args()


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


def read_features(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"frame_index", "processed_index"}:
                    item[key] = int(value)
                elif key == "pose_detected":
                    item[key] = str(value).lower() == "true"
                else:
                    item[key] = safe_float(value)
            rows.append(item)
    return rows


def read_sequence(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def result_for(sequence_row: dict[str, Any], action: str) -> dict[str, Any]:
    return next((item for item in sequence_row.get("results", []) if item.get("action") == action), {})


def local_minima(
    rows: list[dict[str, Any]],
    feature: str,
    start_sec: float,
    max_value: float,
    min_gap_sec: float,
) -> list[dict[str, Any]]:
    candidates = []
    for i in range(2, len(rows) - 2):
        row = rows[i]
        t = safe_float(row.get("timestamp_sec"))
        value = safe_float(row.get(feature))
        if t is None or value is None or t < start_sec or value > max_value:
            continue
        window = [safe_float(rows[j].get(feature)) for j in range(i - 2, i + 3)]
        values = [v for v in window if v is not None]
        if len(values) < 5 or value != min(values):
            continue
        candidates.append(row)

    selected: list[dict[str, Any]] = []
    for row in candidates:
        t = float(row["timestamp_sec"])
        if selected and t - float(selected[-1]["timestamp_sec"]) < min_gap_sec:
            if float(row[feature]) < float(selected[-1][feature]):
                selected[-1] = row
        else:
            selected.append(row)
    return selected


def stage_ranges(sequence: list[dict[str, Any]], action: str, start_sec: float, end_sec: float | None) -> list[dict[str, Any]]:
    ranges = []
    last_key = None
    start_t = None
    prev_t = None
    for row in sequence:
        t = safe_float(row.get("timestamp_sec"))
        if t is None or t < start_sec:
            continue
        if end_sec is not None and t > end_sec:
            break
        result = result_for(row, action)
        key = (result.get("stage"), bool(result.get("summary_rules_passed")), result.get("count"))
        if key != last_key:
            if last_key is not None:
                ranges.append(
                    {
                        "start_sec": start_t,
                        "end_sec": prev_t,
                        "stage": last_key[0],
                        "gate_ok": last_key[1],
                        "count": last_key[2],
                    }
                )
            start_t = t
            last_key = key
        prev_t = t
    if last_key is not None:
        ranges.append(
            {
                "start_sec": start_t,
                "end_sec": prev_t,
                "stage": last_key[0],
                "gate_ok": last_key[1],
                "count": last_key[2],
            }
        )
    return ranges


def build_report(output_dir: Path, args: argparse.Namespace) -> str:
    features = read_features(output_dir / "features.csv")
    sequence = read_sequence(output_dir / "mixed_action_sequence.jsonl")
    sequence_by_frame = {int(row["frame_index"]): row for row in sequence}
    summary = json.loads((output_dir / "mixed_action_summary.json").read_text(encoding="utf-8"))

    lines = [
        "# Task10 Recorded Action Diagnostics",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Frames: {summary.get('frames')}",
        f"- Duration sec: {summary.get('duration_sec')}",
        "",
        "## Counts",
        "",
    ]
    for action, item in summary.get("actions", {}).items():
        events = ", ".join(f"#{event['count']}@{event['timestamp_sec']:.2f}s" for event in item.get("count_events", []))
        lines.append(f"- {action}: {item.get('count', 0)}")
        if events:
            lines.append(f"  events: {events}")

    lines.extend(["", "## Squat Trough Candidates", ""])
    troughs = local_minima(features, "mean_knee_angle", args.squat_start_sec, 150.0, args.min_gap_sec)
    for row in troughs:
        seq = sequence_by_frame.get(int(row["frame_index"]), {})
        result = result_for(seq, "squat")
        score = safe_float(result.get("score"))
        score_text = "n/a" if score is None else f"{score:.3f}"
        lines.append(
            f"- t={row['timestamp_sec']:.2f}s frame={row['frame_index']} "
            f"knee={row['mean_knee_angle']:.1f} hip={row['hip_height_above_ankles_torso']:.2f} "
            f"stage={result.get('stage')} score={score_text} gate={result.get('summary_rules_passed')} "
            f"count={result.get('count')} reason={result.get('reason')}"
        )

    lines.extend(["", "## Jumping-Jack Stage Ranges", ""])
    for item in stage_ranges(sequence, "jumping_jack", 0.0, args.jumping_jack_end_sec):
        lines.append(
            f"- {item['start_sec']:.2f}-{item['end_sec']:.2f}s "
            f"stage={item['stage']} gate={item['gate_ok']} count={item['count']}"
        )

    lines.extend(["", "## Squat Stage Ranges", ""])
    for item in stage_ranges(sequence, "squat", args.squat_start_sec, None):
        lines.append(
            f"- {item['start_sec']:.2f}-{item['end_sec']:.2f}s "
            f"stage={item['stage']} gate={item['gate_ok']} count={item['count']}"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.exists():
        candidate = DEFAULT_ROOT / output_dir
        if candidate.exists():
            output_dir = candidate
    for required in ["features.csv", "mixed_action_sequence.jsonl", "mixed_action_summary.json"]:
        if not (output_dir / required).exists():
            raise SystemExit(f"Missing {required} in {output_dir}")

    md_path = output_dir / f"{args.name}.md"
    md_path.write_text(build_report(output_dir, args), encoding="utf-8")
    print(f"Diagnostics: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
