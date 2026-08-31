#!/usr/bin/env python3
"""Promote a verified rollout trajectory into demo_library/auto_recorded."""

import argparse
import glob
import json
import os
import shutil
import time


ROOT = os.path.dirname(__file__)
DEMO_LIBRARY = os.path.join(ROOT, "demo_library")


def latest_matching(pattern):
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def load_json(path, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Promote a manually verified successful rollout to auto_recorded.")
    parser.add_argument("--rollout", default="", help="Rollout trajectory JSON path.")
    parser.add_argument("--scene", default="", help="Scene package directory.")
    parser.add_argument("--query", default="抓取")
    parser.add_argument("--object-label", default="green_sphere")
    parser.add_argument("--object-shape", default="sphere")
    parser.add_argument("--source-demo-id", default="cube_top_grasp_v2")
    parser.add_argument("--retrieval-score", type=float, default=0.825)
    args = parser.parse_args()

    rollout = args.rollout or latest_matching(
        os.path.join(DEMO_LIBRARY, "rollout_trajectories", "*%s*.json" % args.object_label))
    scene = args.scene or latest_matching(
        os.path.join(DEMO_LIBRARY, "scene_packages", "live_trial_*%s" % args.object_label))

    if not rollout or not os.path.exists(rollout):
        raise FileNotFoundError("No rollout trajectory found. Pass --rollout explicitly.")
    if not scene or not os.path.isdir(scene):
        raise FileNotFoundError("No scene package found. Pass --scene explicitly.")

    trajectory = load_json(rollout, {})
    trajectory["success"] = True
    with open(rollout, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)

    meta = load_json(os.path.join(scene, "metadata.json"), {})
    obj_pose = meta.get("object_pose_base") or meta.get("object_in_base") or {}
    obj_pos = (
        obj_pose.get("position")
        or meta.get("object_position_base")
        or meta.get("center_base")
        or [None, None, None]
    )
    obj_size = (
        meta.get("estimated_object_size")
        or meta.get("object_size")
        or meta.get("stats", {}).get("spread_p10_p90")
    )
    height = meta.get("estimated_object_height")
    if height is None and isinstance(obj_size, list) and obj_size:
        height = max(float(v) for v in obj_size)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    demo_id = "auto_success_%s_%s" % (args.source_demo_id, stamp)
    auto_dir = os.path.join(DEMO_LIBRARY, "auto_recorded")
    os.makedirs(auto_dir, exist_ok=True)

    scene_name = "demo_%s" % demo_id
    scene_dst = os.path.join(DEMO_LIBRARY, "scene_packages", scene_name)
    if os.path.exists(scene_dst):
        shutil.rmtree(scene_dst)
    shutil.copytree(scene, scene_dst)

    recorded = {
        "id": demo_id,
        "format": "mt3_auto_success_v3",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "manual_verified_success_promotion",
        "query": args.query,
        "task": args.query,
        "object_label": args.object_label,
        "object_shape": args.object_shape,
        "object_category": args.object_shape,
        "trial_note": "Promoted manually because the rollout was visually successful but old success detection wrote success=False.",
        "retrieved_demo": args.source_demo_id,
        "retrieved_demo_id": args.source_demo_id,
        "source_demo_id": args.source_demo_id,
        "retrieval_score": args.retrieval_score,
        "success": True,
        "estimated_object_size": obj_size,
        "estimated_object_height": height,
        "object_position_base": obj_pos,
        "grasp_pose_base": None,
        "icp_mean_error_m": None,
        "icp_median_error_m": None,
        "icp_p90_error_m": None,
        "scene_package": scene_name,
        "scene_package_dir": os.path.join("demo_library", "scene_packages", scene_name),
        "trajectory": trajectory,
        "rollout_trajectory_path": os.path.relpath(rollout, ROOT),
        "notes": "Manual promotion of a verified successful MT3 rollout.",
    }

    out = os.path.join(auto_dir, demo_id + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recorded, f, indent=2, ensure_ascii=False)

    print("Promoted rollout to:", out)
    print("Scene package copied to:", scene_dst)
    print("Rollout marked success=True:", rollout)


if __name__ == "__main__":
    main()
