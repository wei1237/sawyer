#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize vertical-insert perception bias by true cylinder XY location.

Works with both the old mt3_relation_trials.csv schema and the v4 diagnostic
schema.  It never edits the input CSV.
"""

import argparse
import csv
import json
import math
from collections import defaultdict


def _float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _xyz(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        data = value
    else:
        try:
            data = json.loads(value)
        except Exception:
            return None
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return None
    x, y = _float(data[0]), _float(data[1])
    if x is None or y is None:
        return None
    z = _float(data[2]) if len(data) > 2 else None
    return [x, y, z]


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _fmt_mm(value):
    return "" if value is None else f"{value * 1000.0:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--round", type=int, default=2,
                    help="decimal places for grouping GT x/y (default: 2)")
    ap.add_argument("--csv-out", default="",
                    help="optional path for grouped summary CSV")
    args = ap.parse_args()

    with open(args.csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        if row.get("task_type") != "cylinder_insert_socket":
            continue
        gt = _xyz(row.get("target_gt_world_xyz") or row.get("target_gt_xyz"))
        est = _xyz(row.get("target_est_xyz"))
        if gt is None or est is None:
            continue
        key = (round(gt[0], args.round), round(gt[1], args.round))
        dx = est[0] - gt[0]
        dy = est[1] - gt[1]
        err = math.hypot(dx, dy)
        item = {
            "trial_id": row.get("trial_id", ""),
            "dx": dx,
            "dy": dy,
            "err": err,
            "raw_dx": _float(row.get("target_raw_center_dx_gt_m")),
            "raw_dy": _float(row.get("target_raw_center_dy_gt_m")),
            "raw_err": _float(row.get("target_raw_center_error_xy_m")),
            "corr_dx": _float(row.get("target_geometry_correction_dx_m")),
            "corr_dy": _float(row.get("target_geometry_correction_dy_m")),
            "corr_error_change": _float(row.get("target_geometry_correction_error_change_m")),
        }
        groups[key].append(item)

    summary = []
    for (gx, gy), items in sorted(groups.items()):
        rec = {
            "gt_x": gx,
            "gt_y": gy,
            "n": len(items),
            "mean_error_mm": _fmt_mm(_mean([i["err"] for i in items])),
            "median_error_mm": _fmt_mm(_median([i["err"] for i in items])),
            "std_error_mm": _fmt_mm(_std([i["err"] for i in items])),
            "max_error_mm": _fmt_mm(max(i["err"] for i in items)),
            "mean_dx_mm": _fmt_mm(_mean([i["dx"] for i in items])),
            "mean_dy_mm": _fmt_mm(_mean([i["dy"] for i in items])),
            "mean_raw_error_mm": _fmt_mm(_mean([i["raw_err"] for i in items])),
            "mean_raw_dx_mm": _fmt_mm(_mean([i["raw_dx"] for i in items])),
            "mean_raw_dy_mm": _fmt_mm(_mean([i["raw_dy"] for i in items])),
            "mean_geometry_correction_dx_mm": _fmt_mm(_mean([i["corr_dx"] for i in items])),
            "mean_geometry_correction_dy_mm": _fmt_mm(_mean([i["corr_dy"] for i in items])),
            "mean_correction_error_change_mm": _fmt_mm(_mean([i["corr_error_change"] for i in items])),
        }
        summary.append(rec)

    fields = [
        "gt_x", "gt_y", "n", "mean_error_mm", "median_error_mm",
        "std_error_mm", "max_error_mm", "mean_dx_mm", "mean_dy_mm",
        "mean_raw_error_mm", "mean_raw_dx_mm", "mean_raw_dy_mm",
        "mean_geometry_correction_dx_mm", "mean_geometry_correction_dy_mm",
        "mean_correction_error_change_mm",
    ]
    print("\t".join(fields))
    for rec in summary:
        print("\t".join(str(rec.get(k, "")) for k in fields))

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary)
        print(f"\nsummary saved: {args.csv_out}")


if __name__ == "__main__":
    main()
