#!/usr/bin/env python3
"""Build a batch audit table/report from pose trajectory summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "datasets" / "pose" / "videos"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

ACTION_PATTERNS = [
    ("squat", ["深蹲", "squat"]),
    ("jumping_jack", ["开合跳", "jumping_jack", "jumpingjack", "jump jack"]),
    ("wave", ["挥手", "wave"]),
    ("punch", ["出拳", "punch"]),
    ("high_knee", ["高抬腿", "high_knee", "high knee"]),
    ("rest", ["静息", "rest", "idle"]),
]

SAMPLE_TYPE_PATTERNS = [
    ("positive", ["正样本", "positive", "pos"]),
    ("negative", ["负样本", "negative", "neg"]),
    ("confusion", ["混淆", "confusion"]),
    ("calibration", ["校准", "calibration"]),
]

VIEW_PATTERNS = [
    ("front", ["正面", "front"]),
    ("side", ["侧面", "side"]),
    ("diagonal", ["斜侧", "斜面", "斜", "diagonal", "oblique"]),
]

CSV_COLUMNS = [
    "sample_id",
    "summary_path",
    "source_video",
    "source_video_duplicate_count",
    "source_video_duplicate_rank",
    "output_dir",
    "person",
    "action",
    "sample_type",
    "view",
    "label_from_summary",
    "view_from_summary",
    "extractor",
    "fps",
    "processed_frames",
    "duration_sec",
    "detected_ratio",
    "mean_visibility",
    "mean_core_visible_ratio",
    "squat_score",
    "jumping_jack_score",
    "hip_height_range",
    "mean_knee_angle_min",
    "mean_knee_angle_range",
    "foot_spread_range",
    "wrist_spread_range",
    "hands_above_shoulders_mean",
    "motion_energy_mean",
    "audit_status",
    "audit_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pose trajectory summary.json files into audit reports.")
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing per-video output folders with summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for asset_audit.csv and asset_audit.md.",
    )
    parser.add_argument("--name", default="asset_audit", help="Output filename stem.")
    parser.add_argument("--min-detected-ratio", type=float, default=0.95)
    parser.add_argument("--min-visibility", type=float, default=0.75)
    return parser.parse_args()


def find_summaries(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root does not exist: {root}")
    return sorted(root.glob("*/summary.json"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def find_pattern(text: str, patterns: list[tuple[str, list[str]]]) -> str:
    text_lower = text.lower()
    for value, keys in patterns:
        for key in keys:
            if key.lower() in text_lower:
                return value
    return "unknown"


def infer_person(text: str) -> str:
    match = re.search(r"(人物\d+|person[_-]?\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def get_stat(summary: dict[str, Any], feature: str, key: str) -> float | None:
    return safe_float(summary.get("feature_stats", {}).get(feature, {}).get(key))


def get_motion_hint(summary: dict[str, Any], group: str, key: str) -> float | None:
    return safe_float(summary.get("motion_hints", {}).get(group, {}).get(key))


def infer_row(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    source_video = str(summary.get("source_video") or "")
    text = f"{summary_path.parent.name} {source_video}"

    label_from_summary = summary.get("label") or ""
    view_from_summary = summary.get("view") or ""
    action = str(label_from_summary) if label_from_summary else find_pattern(text, ACTION_PATTERNS)
    view = str(view_from_summary) if view_from_summary else find_pattern(text, VIEW_PATTERNS)
    sample_type = find_pattern(text, SAMPLE_TYPE_PATTERNS)

    squat_hint = summary.get("motion_hints", {}).get("squat_like", {})
    jumping_hint = summary.get("motion_hints", {}).get("jumping_jack_like", {})

    row = {
        "sample_id": summary_path.parent.name,
        "summary_path": str(summary_path),
        "source_video": source_video,
        "source_video_duplicate_count": 1,
        "source_video_duplicate_rank": 1,
        "output_dir": str(summary_path.parent),
        "person": infer_person(text),
        "action": action or "unknown",
        "sample_type": sample_type,
        "view": view or "unknown",
        "label_from_summary": label_from_summary,
        "view_from_summary": view_from_summary,
        "extractor": summary.get("extractor"),
        "fps": safe_float(summary.get("input_video", {}).get("fps")),
        "processed_frames": safe_int(summary.get("processing", {}).get("processed_frames")),
        "duration_sec": safe_float(summary.get("processing", {}).get("processed_duration_sec")),
        "detected_ratio": safe_float(summary.get("pose_quality", {}).get("detected_ratio")),
        "mean_visibility": safe_float(summary.get("pose_quality", {}).get("mean_visibility")),
        "mean_core_visible_ratio": safe_float(summary.get("pose_quality", {}).get("mean_core_visible_ratio")),
        "squat_score": safe_int(squat_hint.get("score_0_to_3")),
        "jumping_jack_score": safe_int(jumping_hint.get("score_0_to_4")),
        "hip_height_range": get_motion_hint(summary, "squat_like", "hip_height_above_ankles_torso_range"),
        "mean_knee_angle_min": get_motion_hint(summary, "squat_like", "mean_knee_angle_min"),
        "mean_knee_angle_range": get_motion_hint(summary, "squat_like", "mean_knee_angle_range"),
        "foot_spread_range": get_motion_hint(summary, "jumping_jack_like", "foot_spread_shoulder_ratio_range"),
        "wrist_spread_range": get_motion_hint(summary, "jumping_jack_like", "wrist_spread_shoulder_ratio_range"),
        "hands_above_shoulders_mean": get_motion_hint(
            summary, "jumping_jack_like", "hands_above_shoulders_ratio_mean"
        ),
        "motion_energy_mean": get_motion_hint(summary, "jumping_jack_like", "motion_energy_mean"),
    }

    status, notes = audit_row(row)
    row["audit_status"] = status
    row["audit_notes"] = "; ".join(notes)
    return row


def mark_duplicate_sources(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_video") or "")
        groups[source].append(row)

    for source_rows in groups.values():
        if len(source_rows) <= 1:
            continue
        source_rows.sort(key=lambda item: str(item.get("summary_path") or ""))
        for index, row in enumerate(source_rows, start=1):
            row["source_video_duplicate_count"] = len(source_rows)
            row["source_video_duplicate_rank"] = index
            duplicate_note = f"duplicate source video output {index}/{len(source_rows)}"
            notes = str(row.get("audit_notes") or "")
            row["audit_notes"] = f"{notes}; {duplicate_note}" if notes else duplicate_note
            row["audit_status"] = "review"


def audit_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    status = "ok"

    detected_ratio = safe_float(row.get("detected_ratio"))
    mean_visibility = safe_float(row.get("mean_visibility"))
    duration_sec = safe_float(row.get("duration_sec"))

    if detected_ratio is not None and detected_ratio < 0.95:
        notes.append(f"low pose detection ratio {detected_ratio:.2f}")
        status = "review"
    if mean_visibility is not None and mean_visibility < 0.75:
        notes.append(f"low mean visibility {mean_visibility:.2f}")
        status = "review"
    if duration_sec is not None and duration_sec < 2.0:
        notes.append(f"short clip {duration_sec:.2f}s")
        status = "review"

    action = str(row.get("action") or "unknown")
    sample_type = str(row.get("sample_type") or "unknown")
    squat_score = safe_int(row.get("squat_score"))
    jumping_jack_score = safe_int(row.get("jumping_jack_score"))

    if action == "squat":
        if sample_type == "positive" and squat_score is not None and squat_score < 2:
            notes.append(f"squat positive but squat score is {squat_score}/3")
            status = "review"
        if sample_type in {"negative", "confusion"} and squat_score is not None and squat_score >= 2:
            notes.append(f"non-positive sample has high squat score {squat_score}/3")
            status = "review"
        notes.append("squat: ignore hand posture as strong condition")

    if action == "jumping_jack":
        if sample_type == "positive" and jumping_jack_score is not None and jumping_jack_score < 3:
            notes.append(f"jumping-jack positive but score is {jumping_jack_score}/4")
            status = "review"
        if sample_type in {"negative", "confusion"} and jumping_jack_score is not None and jumping_jack_score >= 3:
            notes.append(f"non-positive sample has high jumping-jack score {jumping_jack_score}/4")
            status = "review"

    if not notes:
        notes.append("no obvious issue")
    return status, notes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def number_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [safe_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def group_counts(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key) or "unknown") for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row.get("action") or "unknown")].append(row)

    review_rows = [row for row in rows if row.get("audit_status") != "ok"]
    lines: list[str] = []
    lines.append("# Task10 Asset Audit Report")
    lines.append("")
    lines.append(f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Total samples: {len(rows)}")
    lines.append(f"- Review samples: {len(review_rows)}")
    lines.append("")

    lines.append("## Dataset Mix")
    lines.append("")
    for title, key in [("Action", "action"), ("Sample type", "sample_type"), ("View", "view"), ("Status", "audit_status")]:
        lines.append(f"### {title}")
        for name, count in group_counts(rows, key).most_common():
            lines.append(f"- {name}: {count}")
        lines.append("")

    lines.append("## Quality Summary")
    lines.append("")
    lines.append(f"- Mean detected ratio: {fmt(mean_or_none(number_values(rows, 'detected_ratio')))}")
    lines.append(f"- Mean visibility: {fmt(mean_or_none(number_values(rows, 'mean_visibility')))}")
    lines.append(f"- Mean core visible ratio: {fmt(mean_or_none(number_values(rows, 'mean_core_visible_ratio')))}")
    lines.append("")

    lines.append("## Action Observations")
    lines.append("")
    for action, action_rows in sorted(by_action.items()):
        lines.append(f"### {action}")
        lines.append(f"- Samples: {len(action_rows)}")
        lines.append(f"- Views: {dict(group_counts(action_rows, 'view'))}")
        lines.append(f"- Sample types: {dict(group_counts(action_rows, 'sample_type'))}")

        if action == "squat":
            positives = [r for r in action_rows if r.get("sample_type") == "positive"]
            src = positives or action_rows
            lines.append(f"- Avg squat score: {fmt(mean_or_none(number_values(src, 'squat_score')), 2)} / 3")
            lines.append(f"- Avg hip-height range: {fmt(mean_or_none(number_values(src, 'hip_height_range')))}")
            lines.append(f"- Avg min mean-knee angle: {fmt(mean_or_none(number_values(src, 'mean_knee_angle_min')), 1)} deg")
            lines.append("- Spec hint: hand posture should be treated as variation, not as a required squat condition.")

        if action == "jumping_jack":
            positives = [r for r in action_rows if r.get("sample_type") == "positive"]
            src = positives or action_rows
            lines.append(f"- Avg jumping-jack score: {fmt(mean_or_none(number_values(src, 'jumping_jack_score')), 2)} / 4")
            lines.append(f"- Avg foot-spread range: {fmt(mean_or_none(number_values(src, 'foot_spread_range')))}")
            lines.append(f"- Avg wrist-spread range: {fmt(mean_or_none(number_values(src, 'wrist_spread_range')))}")
            lines.append("- Spec hint: require hand/foot opening together; arm-only or jump-only clips should be confusion samples.")

        lines.append("")

    lines.append("## Review List")
    lines.append("")
    if not review_rows:
        lines.append("- No review samples found by current coarse audit rules.")
    else:
        for row in review_rows:
            lines.append(
                f"- `{row['sample_id']}` | action={row['action']} type={row['sample_type']} "
                f"view={row['view']} | {row['audit_notes']}"
            )
    lines.append("")

    lines.append("## Next Recommendations")
    lines.append("")
    lines.append("- Use this report to select representative positive samples and high-risk confusion samples.")
    lines.append("- Do not manually write notes for every clip; update action specs from repeated patterns.")
    lines.append("- For squat, prioritize hip drop, knee angle, and stand-down-stand sequence.")
    lines.append("- For jumping jack, prioritize synchronized hand/foot open-close sequence.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summaries = find_summaries(args.input_root)
    if not summaries:
        raise SystemExit(f"No summary.json files found under: {args.input_root}")

    rows = [infer_row(path) for path in summaries]
    mark_duplicate_sources(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}.csv"
    md_path = args.output_dir / f"{args.name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Audited samples: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
