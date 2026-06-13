#!/usr/bin/env python3
"""Run draft frame-level action state machines over extracted feature files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "datasets" / "pose" / "videos"
DEFAULT_SPEC_DIR = PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

ACTION_PATTERNS = [
    ("squat", ["深蹲", "squat"]),
    ("jumping_jack", ["开合跳", "jumping_jack", "jumpingjack", "jump jack"]),
    ("rest", ["静息", "rest", "idle"]),
]

SAMPLE_TYPE_PATTERNS = [
    ("positive", ["正样本", "positive", "pos"]),
    ("negative", ["负样本", "negative", "neg"]),
    ("confusion", ["混淆", "confusion"]),
]

VIEW_PATTERNS = [
    ("front", ["正面", "front"]),
    ("side", ["侧面", "side"]),
    ("diagonal", ["斜侧", "斜面", "斜", "diagonal", "oblique"]),
]

REPORT_COLUMNS = [
    "sample_id",
    "source_video",
    "label_action",
    "sample_type",
    "view",
    "spec_action",
    "expected_positive",
    "predicted_positive",
    "status",
    "supported",
    "summary_rules_passed",
    "cycle_count",
    "max_score",
    "dominant_stage",
    "reason",
    "state_summary_path",
    "state_sequence_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline frame-level state machines for action specs.")
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="state_machine_report")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_specs(path: Path) -> list[dict[str, Any]]:
    specs = []
    for spec_path in sorted(path.glob("*.json")):
        spec = read_json(spec_path)
        action_name = str(spec.get("action_name") or "")
        if action_name in {"squat", "jumping_jack"}:
            spec["_path"] = str(spec_path)
            specs.append(spec)
    if not specs:
        raise SystemExit(f"No supported specs found in {path}")
    return specs


def find_output_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root does not exist: {root}")
    dirs = [p.parent for p in root.glob("*/features.csv")]
    return sorted(set(dirs))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def find_pattern(text: str, patterns: list[tuple[str, list[str]]]) -> str:
    text_lower = text.lower()
    for value, keys in patterns:
        for key in keys:
            if key.lower() in text_lower:
                return value
    return "unknown"


def infer_metadata(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    source_video = str(summary.get("source_video") or "")
    text = f"{output_dir.name} {source_video}"
    action = str(summary.get("label") or "") or find_pattern(text, ACTION_PATTERNS)
    sample_type = find_pattern(text, SAMPLE_TYPE_PATTERNS)
    view = str(summary.get("view") or "") or find_pattern(text, VIEW_PATTERNS)
    return {
        "sample_id": output_dir.name,
        "source_video": source_video,
        "label_action": action,
        "sample_type": sample_type,
        "view": view,
    }


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    nums = [safe_float(row.get(key)) for row in rows]
    return [num for num in nums if num is not None]


def feature_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    def stat(key: str, mode: str) -> float | None:
        nums = values(rows, key)
        if not nums:
            return None
        if mode == "min":
            return min(nums)
        if mode == "max":
            return max(nums)
        if mode == "mean":
            return sum(nums) / len(nums)
        if mode == "range":
            return max(nums) - min(nums)
        raise ValueError(mode)

    return {
        "mean_knee_angle_min": stat("mean_knee_angle", "min"),
        "mean_knee_angle_range": stat("mean_knee_angle", "range"),
        "hip_height_range": stat("hip_height_above_ankles_torso", "range"),
        "foot_spread_range": stat("foot_spread_shoulder_ratio", "range"),
        "wrist_spread_range": stat("wrist_spread_shoulder_ratio", "range"),
        "hands_above_shoulders_mean": stat("hands_above_shoulders_ratio", "mean"),
        "motion_energy_mean": stat("motion_energy", "mean"),
    }


def compare(value: float, op: str, expected: float) -> bool:
    if op == "<":
        return value < expected
    if op == "<=":
        return value <= expected
    if op == ">":
        return value > expected
    if op == ">=":
        return value >= expected
    if op == "==":
        return value == expected
    raise ValueError(f"Unsupported op: {op}")


def check_rule(row: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, str]:
    feature = str(rule["feature"])
    op = str(rule["op"])
    threshold = safe_float(rule["value"])
    value = safe_float(row.get(feature))
    if threshold is None:
        return False, f"{feature}: invalid threshold"
    if value is None:
        return False, f"{feature}: missing"
    ok = compare(value, op, threshold)
    return ok, f"{feature}={value:.3f} {op} {threshold:.3f}"


def summary_rules_pass(row: dict[str, Any], spec: dict[str, Any], view: str) -> tuple[bool, bool, str]:
    view_rule = spec.get("summary_rules", {}).get(view)
    if not view_rule:
        return False, False, f"unsupported view {view}"
    if not view_rule.get("enabled", False):
        return False, False, f"disabled view {view}: {view_rule.get('reason', 'no reason')}"

    quality = view_rule.get("quality_gates", {})
    for feature, threshold in [
        ("detected_ratio", quality.get("detected_ratio_min")),
        ("mean_visibility", quality.get("mean_visibility_min")),
        ("mean_core_visible_ratio", quality.get("mean_core_visible_ratio_min")),
    ]:
        if threshold is None:
            continue
        ok, reason = check_rule(row, {"feature": feature, "op": ">=", "value": threshold})
        if not ok:
            return True, False, f"quality gate failed: {reason}"

    for rule in view_rule.get("reject_any", []):
        ok, reason = check_rule(row, rule)
        if ok:
            return True, False, f"reject rule hit: {reason}"

    reasons = []
    for rule in view_rule.get("accept_all", []):
        ok, reason = check_rule(row, rule)
        if not ok:
            return True, False, f"accept rule failed: {reason}"
        reasons.append(reason)

    return True, True, "; ".join(reasons)


def squat_total_rules_pass(row: dict[str, Any], spec: dict[str, Any], view: str) -> tuple[bool, bool, str]:
    """Looser squat gate for counting all squat-like reps before quality grading."""
    if spec.get("action_name") != "squat":
        return summary_rules_pass(row, spec, view)

    view_rule = spec.get("summary_rules", {}).get(view)
    if not view_rule:
        return False, False, f"unsupported view {view}"
    if not view_rule.get("enabled", False):
        return False, False, f"disabled view {view}: {view_rule.get('reason', 'no reason')}"

    quality = view_rule.get("quality_gates", {})
    for feature, threshold in [
        ("detected_ratio", quality.get("detected_ratio_min")),
        ("mean_visibility", quality.get("mean_visibility_min")),
        ("mean_core_visible_ratio", quality.get("mean_core_visible_ratio_min")),
    ]:
        if threshold is None:
            continue
        ok, reason = check_rule(row, {"feature": feature, "op": ">=", "value": threshold})
        if not ok:
            return True, False, f"quality gate failed: {reason}"

    relaxed_rules_by_view: dict[str, list[dict[str, Any]]] = {
        "front": [
            {
                "feature": "mean_knee_angle_min",
                "op": "<=",
                "value": 150.0,
                "reason": "Count shallow squat-like reps, but reject near-straight-leg bending.",
            },
            {
                "feature": "mean_knee_angle_range",
                "op": ">=",
                "value": 35.0,
                "reason": "Require a visible knee-flexion cycle for total squat count.",
            },
            {
                "feature": "hip_height_range",
                "op": ">=",
                "value": 0.22,
                "reason": "Require visible hip drop for total squat count.",
            },
        ],
        "side": [
            {
                "feature": "mean_knee_angle_min",
                "op": "<=",
                "value": 80.0,
                "reason": "Count side-view shallow squat-like reps, but reject bend-over-only motion.",
            },
            {
                "feature": "mean_knee_angle_range",
                "op": ">=",
                "value": 60.0,
                "reason": "Require a visible side-view knee-flexion cycle.",
            },
            {
                "feature": "motion_energy_mean",
                "op": ">=",
                "value": 0.008,
                "reason": "Reject near-static side samples.",
            },
        ],
    }
    relaxed_rules = relaxed_rules_by_view.get(view)
    if not relaxed_rules:
        return True, False, f"unsupported relaxed squat total gate for view {view}"

    # Keep hard rejects that protect against obvious tracking instability.
    for rule in view_rule.get("reject_any", []):
        feature = str(rule.get("feature"))
        if view == "front" and feature == "mean_knee_angle_min":
            continue
        if view == "side" and feature == "mean_knee_angle_min":
            continue
        ok, reason = check_rule(row, rule)
        if ok:
            return True, False, f"reject rule hit: {reason}"

    reasons = []
    for rule in relaxed_rules:
        ok, reason = check_rule(row, rule)
        if not ok:
            return True, False, f"total-count rule failed: {reason}"
        reasons.append(reason)

    return True, True, "; ".join(reasons)


def squat_quality_from_summary(
    row: dict[str, Any],
    view: str,
    standard_passed: bool,
    standard_reason: str,
) -> dict[str, Any]:
    """Classify a counted squat-like rep into standard or shallow/insufficient depth."""
    if standard_passed:
        return {
            "quality": "standard",
            "quality_reason": "standard_rules_passed",
            "advice": "",
        }

    knee_min = safe_float(row.get("mean_knee_angle_min"))
    knee_range = safe_float(row.get("mean_knee_angle_range"))
    hip_range = safe_float(row.get("hip_height_range"))

    if view == "front":
        if knee_min is not None and knee_min > 140.0:
            reason = "knee_angle_not_low_enough"
            advice = "下蹲更深一点，保持膝盖明显弯曲"
        elif knee_range is not None and knee_range < 45.0:
            reason = "knee_flexion_range_not_enough"
            advice = "下蹲和起身幅度再明显一点"
        elif hip_range is not None and hip_range < 0.30:
            reason = "hip_drop_not_enough"
            advice = "髋部下降幅度再明显一点"
        else:
            reason = "standard_gate_failed"
            advice = "动作幅度再完整一点"
    elif view == "side":
        if knee_min is not None and knee_min > 65.0:
            reason = "knee_angle_not_low_enough"
            advice = "下蹲更深一点，保持膝盖明显弯曲"
        elif knee_range is not None and knee_range < 85.0:
            reason = "knee_flexion_range_not_enough"
            advice = "下蹲和起身幅度再明显一点"
        else:
            reason = "standard_gate_failed"
            advice = "动作幅度再完整一点"
    else:
        reason = "standard_gate_failed"
        advice = "动作幅度再完整一点"

    return {
        "quality": "shallow",
        "quality_reason": reason,
        "advice": advice,
        "standard_gate_reason": standard_reason,
    }


def normalize(value: float | None, min_value: float | None, max_value: float | None, inverse: bool = False) -> float | None:
    if value is None or min_value is None or max_value is None:
        return None
    span = max_value - min_value
    if abs(span) < 1e-9:
        return None
    raw = (value - min_value) / span
    if inverse:
        raw = 1.0 - raw
    return max(0.0, min(1.0, raw))


def smooth(values_list: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values_list
    half = window // 2
    out: list[float | None] = []
    for i in range(len(values_list)):
        nums = [v for v in values_list[max(0, i - half) : min(len(values_list), i + half + 1)] if v is not None]
        out.append(sum(nums) / len(nums) if nums else None)
    return out


def stage_from_scores(
    scores: list[float | None],
    low_threshold: float,
    high_threshold: float,
    low_stage: str,
    rising_stage: str,
    high_stage: str,
    falling_stage: str,
) -> list[str]:
    stages: list[str] = []
    previous = None
    for score in scores:
        if score is None:
            stages.append("unknown")
            continue
        if score <= low_threshold:
            stage = low_stage
        elif score >= high_threshold:
            stage = high_stage
        else:
            stage = rising_stage if previous is None or score >= previous else falling_stage
        stages.append(stage)
        previous = score
    return stages


def count_cycles(stages: list[str], low_stage: str, high_stage: str, min_high_hold: int, min_low_hold: int) -> int:
    seen_high = False
    high_run = 0
    low_run = 0
    count = 0
    for stage in stages:
        if stage == high_stage:
            high_run += 1
            low_run = 0
            if high_run >= min_high_hold:
                seen_high = True
        elif stage == low_stage:
            low_run += 1
            high_run = 0
            if seen_high and low_run >= min_low_hold:
                count += 1
                seen_high = False
        else:
            high_run = 0
            low_run = 0
    return count


def score_squat_with_thresholds(
    rows: list[dict[str, str]],
    spec: dict[str, Any],
    stage_thresholds: dict[str, Any] | None = None,
) -> tuple[list[float | None], list[str], int]:
    knee_values = values(rows, "mean_knee_angle")
    hip_values = values(rows, "hip_height_above_ankles_torso")
    knee_min = min(knee_values) if knee_values else None
    knee_max = max(knee_values) if knee_values else None
    hip_min = min(hip_values) if hip_values else None
    hip_max = max(hip_values) if hip_values else None

    scores = []
    for row in rows:
        knee = safe_float(row.get("mean_knee_angle"))
        hip = safe_float(row.get("hip_height_above_ankles_torso"))
        knee_depth = normalize(knee, knee_min, knee_max, inverse=True)
        hip_depth = normalize(hip, hip_min, hip_max, inverse=True)
        parts = []
        if knee_depth is not None:
            parts.append((0.65, knee_depth))
        if hip_depth is not None:
            parts.append((0.35, hip_depth))
        if not parts:
            scores.append(None)
        else:
            weight_sum = sum(weight for weight, _ in parts)
            scores.append(sum(weight * value for weight, value in parts) / weight_sum)

    params = spec.get("frame_state_machine", {})
    window = int(params.get("smoothing_window_frames", 5))
    scores = smooth(scores, window)
    thresholds = stage_thresholds or params.get("stage_thresholds", {})
    stages = stage_from_scores(
        scores,
        float(thresholds.get("stand_max", 0.20)),
        float(thresholds.get("down_min", 0.75)),
        "stand",
        "downing",
        "down",
        "rising",
    )
    count = count_cycles(
        stages,
        "stand",
        "down",
        int(params.get("min_down_hold_frames", 2)),
        int(params.get("min_return_to_stand_frames", 2)),
    )
    return scores, stages, count


def score_squat(rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[float | None], list[str], int]:
    return score_squat_with_thresholds(rows, spec)


def score_squat_total(rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[float | None], list[str], int]:
    params = spec.get("frame_state_machine", {})
    thresholds = dict(params.get("stage_thresholds", {}))
    total_thresholds = dict(params.get("total_count_stage_thresholds", {}))
    thresholds.update(total_thresholds)
    thresholds.setdefault("stand_max", 0.20)
    thresholds.setdefault("down_min", 0.58)
    return score_squat_with_thresholds(rows, spec, thresholds)


def score_jumping_jack(rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[float | None], list[str], int]:
    foot_values = values(rows, "foot_spread_shoulder_ratio")
    wrist_values = values(rows, "wrist_spread_shoulder_ratio")
    foot_min = min(foot_values) if foot_values else None
    foot_max = max(foot_values) if foot_values else None
    wrist_min = min(wrist_values) if wrist_values else None
    wrist_max = max(wrist_values) if wrist_values else None

    scores = []
    for row in rows:
        foot = normalize(safe_float(row.get("foot_spread_shoulder_ratio")), foot_min, foot_max)
        wrist = normalize(safe_float(row.get("wrist_spread_shoulder_ratio")), wrist_min, wrist_max)
        hands = safe_float(row.get("hands_above_shoulders_ratio"))
        parts = []
        if foot is not None:
            parts.append((0.45, foot))
        if wrist is not None:
            parts.append((0.35, wrist))
        if hands is not None:
            parts.append((0.20, max(0.0, min(1.0, hands))))
        if not parts:
            scores.append(None)
        else:
            weight_sum = sum(weight for weight, _ in parts)
            scores.append(sum(weight * value for weight, value in parts) / weight_sum)

    params = spec.get("frame_state_machine", {})
    window = int(params.get("smoothing_window_frames", 5))
    scores = smooth(scores, window)
    thresholds = params.get("stage_thresholds", {})
    stages = stage_from_scores(
        scores,
        float(thresholds.get("closed_max", 0.25)),
        float(thresholds.get("open_min", 0.70)),
        "closed",
        "opening",
        "open",
        "closing",
    )
    count = count_cycles(
        stages,
        "closed",
        "open",
        int(params.get("min_open_hold_frames", 1)),
        int(params.get("min_return_to_closed_frames", 1)),
    )
    return scores, stages, count


def run_state_machine(rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[float | None], list[str], int]:
    action = spec.get("action_name")
    if action == "squat":
        return score_squat(rows, spec)
    if action == "jumping_jack":
        return score_jumping_jack(rows, spec)
    return [None] * len(rows), ["unsupported"] * len(rows), 0


def run_total_count_state_machine(rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[float | None], list[str], int]:
    action = spec.get("action_name")
    if action == "squat":
        return score_squat_total(rows, spec)
    return run_state_machine(rows, spec)


def expected_positive(meta: dict[str, str], spec: dict[str, Any]) -> bool:
    return meta.get("label_action") == spec.get("action_name") and meta.get("sample_type") == "positive"


def write_state_sequence(
    path: Path,
    rows: list[dict[str, str]],
    scores: list[float | None],
    stages: list[str],
    spec_action: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row, score, stage in zip(rows, scores, stages):
            item = {
                "frame_index": int(row["frame_index"]),
                "timestamp_sec": safe_float(row.get("timestamp_sec")),
                "action": spec_action,
                "stage": stage,
                "score": score,
                "features": {
                    "mean_knee_angle": safe_float(row.get("mean_knee_angle")),
                    "hip_height_above_ankles_torso": safe_float(row.get("hip_height_above_ankles_torso")),
                    "foot_spread_shoulder_ratio": safe_float(row.get("foot_spread_shoulder_ratio")),
                    "wrist_spread_shoulder_ratio": safe_float(row.get("wrist_spread_shoulder_ratio")),
                    "hands_above_shoulders_ratio": safe_float(row.get("hands_above_shoulders_ratio")),
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_output_dir(output_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_path = output_dir / "summary.json"
    features_path = output_dir / "features.csv"
    summary = read_json(summary_path)
    rows = read_csv_rows(features_path)
    meta = infer_metadata(output_dir, summary)
    stats = feature_stats(rows)
    summary_row: dict[str, Any] = {
        **stats,
        "detected_ratio": summary.get("pose_quality", {}).get("detected_ratio"),
        "mean_visibility": summary.get("pose_quality", {}).get("mean_visibility"),
        "mean_core_visible_ratio": summary.get("pose_quality", {}).get("mean_core_visible_ratio"),
    }

    results = []
    for spec in specs:
        spec_action = str(spec.get("action_name"))
        supported, summary_passed, summary_reason = summary_rules_pass(summary_row, spec, meta["view"])
        scores, stages, cycle_count = run_state_machine(rows, spec)
        max_score = max([score for score in scores if score is not None], default=None)
        predicted = bool(supported and summary_passed and cycle_count >= 1)
        expected = expected_positive(meta, spec)
        if expected and not supported:
            status = "skipped_unsupported_view"
        elif predicted == expected:
            status = "pass"
        else:
            status = "fail"

        stage_counts = Counter(stages)
        dominant_stage = stage_counts.most_common(1)[0][0] if stage_counts else "unknown"
        sequence_path = output_dir / f"state_sequence_{spec_action}.jsonl"
        state_summary_path = output_dir / f"state_summary_{spec_action}.json"
        write_state_sequence(sequence_path, rows, scores, stages, spec_action)
        state_summary = {
            **meta,
            "spec_action": spec_action,
            "supported": supported,
            "summary_rules_passed": summary_passed,
            "summary_reason": summary_reason,
            "cycle_count": cycle_count,
            "max_score": max_score,
            "stage_counts": dict(stage_counts),
            "expected_positive": expected,
            "predicted_positive": predicted,
            "status": status,
            "state_sequence_path": str(sequence_path),
        }
        write_json(state_summary_path, state_summary)
        results.append(
            {
                **meta,
                "spec_action": spec_action,
                "expected_positive": expected,
                "predicted_positive": predicted,
                "status": status,
                "supported": supported,
                "summary_rules_passed": summary_passed,
                "cycle_count": cycle_count,
                "max_score": max_score,
                "dominant_stage": dominant_stage,
                "reason": summary_reason,
                "state_summary_path": str(state_summary_path),
                "state_sequence_path": str(sequence_path),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_spec[str(row.get("spec_action"))].append(row)

    lines = []
    lines.append("# Task10 Offline State Machine Report")
    lines.append("")
    lines.append(f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Total evaluations: {len(rows)}")
    lines.append(f"- Status: {dict(Counter(str(row.get('status')) for row in rows))}")
    lines.append("")

    for spec_action, spec_rows in sorted(by_spec.items()):
        counted = [row for row in spec_rows if row.get("status") != "skipped_unsupported_view"]
        passed = [row for row in counted if row.get("status") == "pass"]
        failed = [row for row in counted if row.get("status") == "fail"]
        supported_positive_rows = [
            row
            for row in spec_rows
            if row.get("expected_positive") and row.get("supported")
        ]
        skipped_positive_rows = [
            row
            for row in spec_rows
            if row.get("expected_positive") and not row.get("supported")
        ]
        gated_rows = [
            row
            for row in counted
            if int(row.get("cycle_count") or 0) >= 1
            and not row.get("summary_rules_passed")
        ]
        lines.append(f"## {spec_action}")
        lines.append("")
        lines.append(f"- Counted evaluations: {len(counted)}")
        lines.append(f"- Passed: {len(passed)}")
        lines.append(f"- Failed: {len(failed)}")
        lines.append(f"- Skipped unsupported view: {len(spec_rows) - len(counted)}")
        if counted:
            lines.append(f"- Offline state-machine coarse accuracy: {len(passed) / len(counted):.3f}")
        if supported_positive_rows:
            counts = [int(row.get("cycle_count") or 0) for row in supported_positive_rows]
            lines.append(f"- Supported positive sample cycle counts: {counts}")
        if skipped_positive_rows:
            ids = ", ".join(f"`{row['sample_id']}`" for row in skipped_positive_rows)
            lines.append(f"- Skipped positive samples: {ids}")
        if gated_rows:
            lines.append(f"- Samples with raw cycles blocked by summary gates: {len(gated_rows)}")
        lines.append("")
        if gated_rows:
            lines.append("### Gated Raw Cycles")
            for row in gated_rows:
                lines.append(
                    f"- `{row['sample_id']}` label={row['label_action']} type={row['sample_type']} "
                    f"view={row['view']} cycles={row['cycle_count']} | {row['reason']}"
                )
            lines.append("")
        if failed:
            lines.append("### Failures")
            for row in failed:
                lines.append(
                    f"- `{row['sample_id']}` label={row['label_action']} type={row['sample_type']} "
                    f"view={row['view']} expected={row['expected_positive']} "
                    f"predicted={row['predicted_positive']} cycles={row['cycle_count']} | {row['reason']}"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    specs = load_specs(args.spec_dir)
    output_dirs = find_output_dirs(args.input_root)
    all_rows: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        all_rows.extend(process_output_dir(output_dir, specs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}.csv"
    md_path = args.output_dir / f"{args.name}.md"
    write_csv(csv_path, all_rows)
    write_markdown(md_path, all_rows)
    print(f"Evaluated rows: {len(all_rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
