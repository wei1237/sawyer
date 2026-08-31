#!/usr/bin/env python3
"""Summarise target-anchor and stage-replay ablation experiments."""

import argparse
import csv
import math
import os
from collections import defaultdict


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_ROOT = os.path.join(CODE_DIR, "demo_library", "experiment_logs")
DEFAULT_OUTPUT = os.path.join(CODE_DIR, "relation_ablation_summary.csv")


def _parse_bool(value):
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "success", "succeeded", "pass"):
        return True
    if text in ("0", "false", "no", "failed", "failure", "fail"):
        return False
    return None


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _success_value(row, source):
    manual = _parse_bool(row.get("manual_success_label"))
    execution = _parse_bool(row.get("execution_success"))
    if source == "manual":
        return manual
    if source == "execution":
        return execution
    return manual if manual is not None else execution


def _read_rows(log_root):
    rows = []
    if not os.path.isdir(log_root):
        return rows
    for root, _dirs, files in os.walk(log_root):
        for name in files:
            if name != "mt3_relation_trials.csv":
                continue
            path = os.path.join(root, name)
            with open(path, "r", newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    row["_source_csv"] = path
                    rows.append(row)
    return rows


def _deduplicate(rows):
    latest = {}
    for index, row in enumerate(rows):
        key = (
            row.get("trial_id", ""),
            row.get("task_type", ""),
            row.get("method_variant", ""),
        )
        latest[key] = (index, row)
    return [item[1] for item in sorted(latest.values())]


def _wilson_interval(successes, count, z=1.96):
    if count <= 0:
        return "", ""
    proportion = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values):
    values = [value for value in values if value is not None]
    return "" if not values else sum(values) / float(len(values))


def _normalise_variant(row):
    relation = row.get("relation_alignment_mode", "") or "target_anchor"
    transfer = row.get("trajectory_transfer_mode", "")
    if not transfer:
        replay = _parse_bool(row.get("replay_used"))
        transfer = "stage_replay" if replay else "scripted_execution"
    if relation in ("target_displacement", "target_only", "no_relation"):
        relation = "target_displacement"
    else:
        relation = "target_anchor"
    return relation, transfer


def _summarise(rows, success_source):
    groups = defaultdict(list)
    skipped_unlabelled = 0
    for row in rows:
        if row.get("outcome") == "dry_run":
            continue
        success = _success_value(row, success_source)
        if success is None:
            skipped_unlabelled += 1
            continue
        relation, transfer = _normalise_variant(row)
        key = (
            row.get("task_type", ""),
            row.get("condition_id", ""),
            row.get("method_variant", ""),
            relation,
            transfer,
        )
        groups[key].append((row, success))

    summary = []
    for key, items in sorted(groups.items()):
        successes = sum(1 for _row, success in items if success)
        count = len(items)
        low, high = _wilson_interval(successes, count)
        summary.append({
            "task_type": key[0],
            "condition_id": key[1],
            "method_variant": key[2],
            "relation_alignment_mode": key[3],
            "trajectory_transfer_mode": key[4],
            "trials": count,
            "successes": successes,
            "success_rate": float(successes) / float(count),
            "success_ci95_low": low,
            "success_ci95_high": high,
            "mean_total_time_s": _mean([
                _parse_float(row.get("total_time_s")) for row, _ in items
            ]),
            "mean_perception_time_s": _mean([
                _parse_float(row.get("perception_time_s")) for row, _ in items
            ]),
            "mean_execution_time_s": _mean([
                _parse_float(row.get("execution_time_s")) for row, _ in items
            ]),
        })
    return summary, skipped_unlabelled


def _write_summary(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = [
        "task_type",
        "condition_id",
        "method_variant",
        "relation_alignment_mode",
        "trajectory_transfer_mode",
        "trials",
        "successes",
        "success_rate",
        "success_ci95_low",
        "success_ci95_high",
        "mean_total_time_s",
        "mean_perception_time_s",
        "mean_execution_time_s",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    if not rows:
        print("No labelled relation-ablation trials found.")
        return
    header = (
        "%-24s %-22s %-30s %7s %9s %s"
        % ("task", "condition", "variant", "trials", "success", "95% CI")
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-24s %-22s %-30s %7d %8.1f%% [%.1f%%, %.1f%%]"
            % (
                row["task_type"][:24],
                row["condition_id"][:22],
                row["method_variant"][:30],
                row["trials"],
                100.0 * row["success_rate"],
                100.0 * row["success_ci95_low"],
                100.0 * row["success_ci95_high"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--success-source",
        choices=("manual", "execution", "auto"),
        default="manual",
        help=(
            "manual is required for paper results; execution only measures "
            "controller completion; auto prefers manual and falls back to execution"
        ),
    )
    args = parser.parse_args()

    source_rows = _deduplicate(_read_rows(args.log_root))
    summary, skipped = _summarise(source_rows, args.success_source)
    _write_summary(summary, os.path.abspath(args.output))
    _print_summary(summary)
    print("\nSummary:", os.path.abspath(args.output))
    if skipped:
        print(
            "Skipped %d unlabelled trials. Fill manual_success_label or use "
            "--success-source execution for controller-only diagnostics." % skipped
        )


if __name__ == "__main__":
    main()
