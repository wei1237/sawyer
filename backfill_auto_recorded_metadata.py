#!/usr/bin/env python3
"""Backfill flat metadata fields for auto-recorded MT3 demo artifacts."""

import glob
import json
import os


AUTO_RECORDED_DIR = os.path.join(
    os.path.dirname(__file__), "demo_library", "auto_recorded")


def infer_label_shape(data):
    label = data.get("object_label")
    shape = data.get("object_shape")
    rollout = str(data.get("rollout_trajectory_path", ""))
    text = " ".join([
        str(label or ""),
        str(shape or ""),
        rollout,
        str(data.get("id", "")),
    ]).lower()

    if not label or label in ("None", "unknown"):
        if "cylinder" in text:
            label = "green_cylinder"
        elif "cuboid" in text or "rect" in text:
            label = "green_cuboid"
        elif "sphere" in text or "ball" in text:
            label = "green_sphere"
        elif "cube" in text:
            label = "green_cube"
        else:
            label = "unknown"

    if not shape or shape in ("None", "unknown"):
        if "cylinder" in str(label).lower() or "cylinder" in text:
            shape = "cylinder"
        elif "cuboid" in str(label).lower() or "cuboid" in text or "rect" in text:
            shape = "cuboid"
        elif "sphere" in str(label).lower() or "sphere" in text or "ball" in text:
            shape = "sphere"
        elif "cube" in str(label).lower() or "cube" in text:
            shape = "cube"
        else:
            shape = "unknown"

    return label, shape


def backfill(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectory = data.get("trajectory", {}) or {}
    object_info = data.get("object_info", {}) or {}
    pose = data.get("bottleneck_pose_base_frame", {}) or {}
    icp = data.get("icp_metrics", {}) or {}

    label, shape = infer_label_shape(data)
    size = data.get("estimated_object_size") or object_info.get("size_m")
    height = data.get("estimated_object_height") or object_info.get("height_m")
    if height is None and isinstance(size, list) and size:
        height = max(float(v) for v in size)

    pos_dict = pose.get("position_m", {}) or {}
    ori_dict = pose.get("orientation_xyzw", {}) or {}
    if pos_dict and ori_dict:
        grasp_pose_base = {
            "position": [pos_dict.get("x"), pos_dict.get("y"), pos_dict.get("z")],
            "orientation_xyzw": [
                ori_dict.get("x"),
                ori_dict.get("y"),
                ori_dict.get("z"),
                ori_dict.get("w"),
            ],
        }
    else:
        grasp_pose_base = data.get("grasp_pose_base")

    data["query"] = data.get("query") or data.get("language_description")
    data["task"] = data.get("task") or data.get("language_description")
    data["object_label"] = label
    data["object_shape"] = shape
    data["object_category"] = data.get("object_category") or object_info.get("category") or shape
    data["trial_note"] = data.get("trial_note") or ""
    data["retrieved_demo"] = data.get("retrieved_demo") or data.get("source_demo_id")
    data["retrieved_demo_id"] = data.get("retrieved_demo_id") or data.get("source_demo_id")
    data["success"] = bool(trajectory.get("success", data.get("success", False)))
    data["estimated_object_size"] = size
    data["estimated_object_height"] = height
    data["object_position_base"] = (
        data.get("object_position_base") or object_info.get("position_base"))
    data["grasp_pose_base"] = grasp_pose_base
    data["icp_mean_error_m"] = data.get("icp_mean_error_m") or icp.get("mean_error_m")
    data["icp_median_error_m"] = data.get("icp_median_error_m") or icp.get("median_error_m")
    data["icp_p90_error_m"] = data.get("icp_p90_error_m") or icp.get("p90_error_m")

    if object_info:
        object_info["label"] = label
        object_info["category"] = data["object_category"]
        object_info["height_m"] = height
        data["object_info"] = object_info

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def main():
    paths = sorted(glob.glob(os.path.join(AUTO_RECORDED_DIR, "*.json")))
    print("auto_recorded count:", len(paths))
    for path in paths:
        data = backfill(path)
        print(
            os.path.basename(path),
            "success=%s" % data.get("success"),
            "label=%s" % data.get("object_label"),
            "shape=%s" % data.get("object_shape"),
            "waypoints=%s" % (data.get("trajectory") or {}).get("num_waypoints"),
        )


if __name__ == "__main__":
    main()
