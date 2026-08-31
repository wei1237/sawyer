#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manually annotate wall/rim contact for the latest vertical_insert trial.

Usage:
    python3 mark_insert_trial.py no_contact
    python3 mark_insert_trial.py contact
    python3 mark_insert_trial.py uncertain
    python3 mark_insert_trial.py no_contact --note "clean insertion"
    python3 mark_insert_trial.py contact --trial-id insert_20260815_044431

The script updates BOTH:
  - mt3_relation_trials.csv
  - mt3_relation_trials.jsonl

By default it edits the latest trial.  --trial-id can be used to target a
specific trial safely.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime


DEFAULT_LOG_DIR = (
    "/mnt/hgfs2/code/learning_thousand_tasks/"
    "demo_library/experiment_logs/vertical_insert"
)
CSV_NAME = "mt3_relation_trials.csv"
JSONL_NAME = "mt3_relation_trials.jsonl"

ALLOWED_LABELS = ("no_contact", "contact", "uncertain")

NEW_FIELDS = [
    "manual_wall_contact",
    "manual_contact_note",
    "manual_contact_timestamp",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate wall/rim contact for a vertical_insert trial."
    )
    parser.add_argument(
        "label",
        choices=ALLOWED_LABELS,
        help="Manual wall-contact label.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional short note, e.g. 'clean insertion'.",
    )
    parser.add_argument(
        "--trial-id",
        default="",
        help="Optional exact trial_id. If omitted, annotate the latest trial.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Experiment log directory.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_before_manual_contact backup files.",
    )
    return parser.parse_args()


def _load_csv(path):
    if not os.path.exists(path):
        raise RuntimeError("CSV log not found: %s" % path)

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV log has no trial rows: %s" % path)

    return fieldnames, rows


def _load_jsonl(path):
    if not os.path.exists(path):
        raise RuntimeError("JSONL log not found: %s" % path)

    rows = []
    raw_blank_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                raw_blank_lines += 1
                continue
            try:
                obj = json.loads(text)
            except Exception as exc:
                raise RuntimeError(
                    "Invalid JSONL at line %d: %s" % (line_no, exc)
                )
            if not isinstance(obj, dict):
                raise RuntimeError(
                    "JSONL line %d is not an object." % line_no
                )
            rows.append(obj)

    if not rows:
        raise RuntimeError("JSONL log has no trial rows: %s" % path)

    return rows, raw_blank_lines


def _find_index(rows, trial_id):
    if trial_id:
        matches = [
            i for i, row in enumerate(rows)
            if str(row.get("trial_id", "")) == str(trial_id)
        ]
        if not matches:
            raise RuntimeError("trial_id not found: %s" % trial_id)
        if len(matches) > 1:
            raise RuntimeError(
                "trial_id appears multiple times; refusing ambiguous edit: %s"
                % trial_id
            )
        return matches[0]

    return len(rows) - 1


def _backup(path):
    backup = path + ".bak_before_manual_contact"
    shutil.copy2(path, backup)
    return backup


def _write_csv(path, fieldnames, rows):
    for field in NEW_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    tmp = path + ".tmp_manual_contact"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    os.replace(tmp, path)


def _write_jsonl(path, rows):
    tmp = path + ".tmp_manual_contact"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp, path)


def main():
    args = _parse_args()

    csv_path = os.path.join(args.log_dir, CSV_NAME)
    jsonl_path = os.path.join(args.log_dir, JSONL_NAME)

    csv_fields, csv_rows = _load_csv(csv_path)
    json_rows, _ = _load_jsonl(jsonl_path)

    csv_idx = _find_index(csv_rows, args.trial_id)
    json_idx = _find_index(json_rows, args.trial_id)

    csv_trial_id = str(csv_rows[csv_idx].get("trial_id", ""))
    json_trial_id = str(json_rows[json_idx].get("trial_id", ""))

    if csv_trial_id != json_trial_id:
        raise RuntimeError(
            "CSV/JSONL target mismatch: CSV trial_id=%r, JSONL trial_id=%r. "
            "No files were changed."
            % (csv_trial_id, json_trial_id)
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    annotation = {
        "manual_wall_contact": args.label,
        "manual_contact_note": args.note,
        "manual_contact_timestamp": timestamp,
    }

    csv_rows[csv_idx].update(annotation)
    json_rows[json_idx].update(annotation)

    backups = []
    if not args.no_backup:
        backups.append(_backup(csv_path))
        backups.append(_backup(jsonl_path))

    _write_csv(csv_path, csv_fields, csv_rows)
    _write_jsonl(jsonl_path, json_rows)

    row = csv_rows[csv_idx]

    print("=" * 64)
    print("Manual insertion contact annotation saved")
    print("trial_id              =", csv_trial_id)
    print("condition_id          =", row.get("condition_id", ""))
    print("repeat_id             =", row.get("repeat_id", ""))
    print("task_success          =", row.get("task_success", ""))
    print("pure_replay_success   =", row.get("pure_replay_success", ""))
    print("manual_wall_contact   =", args.label)
    print("manual_contact_note   =", args.note)
    print("timestamp             =", timestamp)
    print("CSV                    =", csv_path)
    print("JSONL                  =", jsonl_path)
    if backups:
        print("backup CSV             =", backups[0])
        print("backup JSONL           =", backups[1])
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[mark_insert_trial] ERROR:", exc, file=sys.stderr)
        sys.exit(1)
