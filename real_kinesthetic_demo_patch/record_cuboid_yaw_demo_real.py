#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Sawyer yawed-cuboid top-grasp demo via kinesthetic/Zero-G guidance."""

from __future__ import print_function

import json
import math
import os
import time

import rospy
from tf.transformations import euler_from_quaternion

from real_kinesthetic_recorder import (
    KinestheticRecorder, ensure_dir, event_index, pose_as_bottleneck,
)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CODE_DIR, "demo_library", "real", "recorded")
ROLLOUT_DIR = os.path.join(
    CODE_DIR, "demo_library", "real", "rollout_trajectories")


def required_xyz(prefix):
    keys = ["~%s_x" % prefix, "~%s_y" % prefix, "~%s_z" % prefix]
    missing = [k for k in keys if not rospy.has_param(k)]
    if missing:
        raise RuntimeError("Explicit real %s coordinates required: %s" %
                           (prefix, ", ".join(missing)))
    return [float(rospy.get_param(k)) for k in keys]


def normalize_deg(x):
    x = (float(x) + 180.0) % 360.0 - 180.0
    return x


def main():
    rospy.init_node("mt3_record_real_cuboid_yaw_demo", anonymous=True)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(ROLLOUT_DIR)

    demo_name = rospy.get_param("~demo_name", "cuboid_green_top_yaw_grasp_real")
    object_pos = required_xyz("object")
    object_size = [float(v) for v in rospy.get_param("~object_size", [0.04, 0.08, 0.035])]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(ROLLOUT_DIR, "%s_kinesthetic_%s" % (demo_name, stamp))
    recorder = KinestheticRecorder("rotated_top_grasp", session_dir)
    trajectory = recorder.record()
    if trajectory is None:
        return False
    rollout_path = os.path.join(session_dir, "rollout.json")
    recorder.save_rollout(trajectory, rollout_path)

    close_idx = event_index(trajectory["events"], ["gripper_close"])
    first = trajectory["poses"][0]
    close_pose = trajectory["poses"][close_idx]
    if rospy.has_param("~gripper_yaw_deg"):
        yaw_deg = float(rospy.get_param("~gripper_yaw_deg"))
        yaw_source = "ros_param"
    else:
        yaw_deg = normalize_deg(math.degrees(euler_from_quaternion(
            close_pose["orientation"])[2]))
        yaw_source = "close_pose_rpy_yaw"

    top_trajectory = dict(trajectory)
    top_trajectory["format"] = "end_effector_pose_twist_gripper_v2"

    demo = {
        "id": demo_name,
        "format": "mt3_recorded_v2",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recording_mode": "human_kinesthetic_zero_g",
        "object_info": {
            "position_base": object_pos,
            "size_m": object_size,
            "category": "cuboid",
            "label": rospy.get_param("~object_label", "green_cuboid"),
            "color": rospy.get_param("~object_color", "green"),
        },
        "bottleneck_pose_base_frame": pose_as_bottleneck(first),
        "grasp_pose_base_frame": {
            "position_m": {
                "x": float(close_pose["position"][0]),
                "y": float(close_pose["position"][1]),
                "z": float(close_pose["position"][2]),
            },
            "orientation_xyzw": {
                "x": float(close_pose["orientation"][0]),
                "y": float(close_pose["orientation"][1]),
                "z": float(close_pose["orientation"][2]),
                "w": float(close_pose["orientation"][3]),
            },
        },
        "trajectory": top_trajectory,
        "gripper_yaw_deg": float(yaw_deg),
        "gripper_yaw_source": yaw_source,
        "grasp_strategy": "top_down_yaw_short_side",
        "language_tags": [
            "grasp", "pick up", "top-down grasp", "cuboid",
            "rectangular prism", "yaw grasp", "rotated gripper", "抓取", "长方体"],
        "language_description": rospy.get_param(
            "~language_description",
            "Pick up the rectangular object from above with a yawed gripper"),
        "approach_direction": [0.0, 0.0, -1.0],
        "retract_direction": [0.0, 0.0, 1.0],
        "real_scene_snapshot": recorder.snapshot,
        "rollout_trajectory_path": rollout_path,
        "notes": "Human-guided real Sawyer demonstration; no arm trajectory was commanded by recorder.",
    }
    out = os.path.join(OUTPUT_DIR, "%s.json" % demo_name)
    with open(out, "w") as f:
        json.dump(demo, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Real cuboid-yaw demo saved: %s", out)
    rospy.loginfo("Recorded gripper_yaw_deg=%.2f (%s)", yaw_deg, yaw_source)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("record_cuboid_yaw_demo_real failed: %s", exc)
        raise
