#!/usr/bin/env python3
"""Evaluate draft action specs against Task10 asset audit rows.

This evaluator is intentionally summary-level. It validates whether current
action-spec thresholds separate positive/confusion samples before we implement
frame-level state machines.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_AUDIT_CSV = SCRIPT_DIR / "datasets" / "pose" / "reports" / "asset_audit.csv"
DEFAULT_SPEC_DIR = PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"


OUTPUT_COLUMNS = [
    "sample_id",
    "source_video",
    "action_label",
    "sample_type",
    "view",
    "spec_action",
    "expected_positive",
    "predicted_positive",
    "status",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate draft action specs against asset_audit.csv.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="spec_eval")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_audit_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_specs(path: Path) -> list[dict[str, Any]]:
    specs = []
    for spec_path in sorted(path.glob("*.json")):
        spec = read_json(spec_path)
        spec["_path"] = str(spec_path)
        specs.append(spec)
    if not specs:
        raise SystemExit(f"No specs found in {path}")
    return specs


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def check_rule(row: dict[str, str], rule: dict[str, Any]) -> tuple[bool, str]:
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


def evaluate_spec(row: dict[str, str], spec: dict[str, Any]) -> tuple[bool, str]:
    action = str(spec.get("action_name"))
    view = row.get("view") or "unknown"
    view_rule = spec.get("summary_rules", {}).get(view)
    if not view_rule:
        return False, f"unsupported view {view}"
    if not view_rule.get("enabled", False):
        return False, f"disabled view {view}: {view_rule.get('reason', 'no reason')}"

    quality = view_rule.get("quality_gates", {})
    quality_checks = [
        ("detected_ratio", ">=", quality.get("detected_ratio_min")),
        ("mean_visibility", ">=", quality.get("mean_visibility_min")),
        ("mean_core_visible_ratio", ">=", quality.get("mean_core_visible_ratio_min")),
    ]
    for feature, op, threshold in quality_checks:
        if threshold is None:
            continue
        ok, reason = check_rule(row, {"feature": feature, "op": op, "value": threshold})
        if not ok:
            return False, f"quality gate failed for {action}: {reason}"

    reject_reasons = []
    for rule in view_rule.get("reject_any", []):
        ok, reason = check_rule(row, rule)
        if ok:
            reject_reasons.append(f"reject rule hit: {reason}")
    if reject_reasons:
        return False, "; ".join(reject_reasons)

    accept_reasons = []
    for rule in view_rule.get("accept_all", []):
        ok, reason = check_rule(row, rule)
        if not ok:
            return False, f"accept rule failed: {reason}"
        accept_reasons.append(reason)

    return True, "; ".join(accept_reasons)


def expected_positive(row: dict[str, str], spec: dict[str, Any]) -> bool:
    return row.get("action") == spec.get("action_name") and row.get("sample_type") == "positive"


def should_skip(row: dict[str, str], spec: dict[str, Any]) -> bool:
    view = row.get("view") or "unknown"
    view_rule = spec.get("summary_rules", {}).get(view)
    if not view_rule:
        return False
    if view_rule.get("enabled", False):
        return False
    # Disabled views are not counted as failures while the spec explicitly marks
    # them as unsupported/experimental.
    return row.get("action") == spec.get("action_name") and row.get("sample_type") == "positive"


def evaluate(rows: list[dict[str, str]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        for spec in specs:
            predicted, reason = evaluate_spec(row, spec)
            expected = expected_positive(row, spec)
            skip = should_skip(row, spec)
            if skip:
                status = "skipped_unsupported_view"
            elif predicted == expected:
                status = "pass"
            else:
                status = "fail"
            results.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_video": row.get("source_video"),
                    "action_label": row.get("action"),
                    "sample_type": row.get("sample_type"),
                    "view": row.get("view"),
                    "spec_action": spec.get("action_name"),
                    "expected_positive": expected,
                    "predicted_positive": predicted,
                    "status": status,
                    "reason": reason,
                }
            )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_spec[str(row.get("spec_action"))].append(row)

    lines = []
    lines.append("# Task10 Spec Evaluation")
    lines.append("")
    lines.append(f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Total evaluations: {len(rows)}")
    lines.append(f"- Status: {dict(Counter(str(row.get('status')) for row in rows))}")
    lines.append("")

    for spec_action, spec_rows in sorted(by_spec.items()):
        counted = [row for row in spec_rows if row.get("status") != "skipped_unsupported_view"]
        passed = [row for row in counted if row.get("status") == "pass"]
        failed = [row for row in counted if row.get("status") == "fail"]
        lines.append(f"## {spec_action}")
        lines.append("")
        lines.append(f"- Counted evaluations: {len(counted)}")
        lines.append(f"- Passed: {len(passed)}")
        lines.append(f"- Failed: {len(failed)}")
        lines.append(f"- Skipped unsupported view: {len(spec_rows) - len(counted)}")
        if counted:
            lines.append(f"- Coarse accuracy: {len(passed) / len(counted):.3f}")
        lines.append("")
        if failed:
            lines.append("### Failures")
            for row in failed:
                lines.append(
                    f"- `{row['sample_id']}` label={row['action_label']} type={row['sample_type']} "
                    f"view={row['view']} expected={row['expected_positive']} "
                    f"predicted={row['predicted_positive']} | {row['reason']}"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_audit_csv(args.audit_csv)
    specs = load_specs(args.spec_dir)
    results = evaluate(rows, specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}.csv"
    md_path = args.output_dir / f"{args.name}.md"
    write_csv(csv_path, results)
    write_markdown(md_path, results)
    print(f"Evaluated rows: {len(results)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
