#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Sawyer cylinder/socket demonstration via human kinesthetic Zero-G guidance.

No arm motion is commanded. The human performs pick + insertion + release, while
this script records base->right_hand and gripper events into the structured schema
expected by the insertion pipeline.
"""

from __future__ import print_function

import json
import os
import time

import rospy

from mt3_cylinder_insert_generalization import (
    DEFAULT_SOCKET_PROFILE, compute_insert_target)
from real_kinesthetic_recorder import (
    KinestheticRecorder, build_grasp_segment, build_terminal_segment,
    ensure_dir,
)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CODE_DIR, "demo_library", "real", "recorded")
ROLLOUT_DIR = os.path.join(
    CODE_DIR, "demo_library", "real", "rollout_trajectories")


def required_xyz(prefix):
    keys = ["~%s_x" % prefix, "~%s_y" % prefix, "~%s_z" % prefix]
    missing = [k for k in keys if not rospy.has_param(k)]
    if missing:
        raise RuntimeError(
            "Explicit real %s coordinates required for demo metadata: %s" %
            (prefix, ", ".join(missing)))
    return [float(rospy.get_param(k)) for k in keys]


def main():
    rospy.init_node("record_cylinder_insert_demo_real", anonymous=True)
    ensure_dir(DEMO_DIR)
    ensure_dir(ROLLOUT_DIR)

    demo_id = rospy.get_param("~demo_id", "green_cylinder_insert_blue_socket_real")
    cylinder_xyz = required_xyz("target")
    socket_xyz = required_xyz("socket")
    cylinder_size = [float(v) for v in rospy.get_param("~cylinder_size", [0.045, 0.045, 0.100])]
    socket_size = [float(v) for v in rospy.get_param("~socket_size", [0.105, 0.105, 0.100])]
    socket_opening = [float(v) for v in rospy.get_param("~socket_opening", [0.055, 0.055])]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(ROLLOUT_DIR, "%s_kinesthetic_%s" % (demo_id, stamp))
    recorder = KinestheticRecorder("cylinder_insert_socket", session_dir)
    trajectory = recorder.record()
    if trajectory is None:
        return False
    rollout_path = os.path.join(session_dir, "rollout.json")
    recorder.save_rollout(trajectory, rollout_path)

    terminal_mark = None
    for e in trajectory.get("events", []):
        if e.get("name") == "terminal_bottleneck":
            terminal_mark = int(e["sample_index"])
    grasp_traj, grasp_bn, grasp_close = build_grasp_segment(
        trajectory,
        terminal_start_idx=terminal_mark,
        pre_samples=int(rospy.get_param("~grasp_bottleneck_pre_samples", 75)),
        post_samples=int(rospy.get_param("~grasp_bottleneck_post_samples", 30)))
    insert_traj, insert_bn, insert_release = build_terminal_segment(
        trajectory,
        kind="insertion",
        pre_samples=int(rospy.get_param("~insertion_bottleneck_pre_samples", 75)),
        post_samples=int(rospy.get_param("~insertion_bottleneck_post_samples", 90)))
    if grasp_traj is None or insert_traj is None:
        raise RuntimeError("Could not extract grasp/insertion segments from explicit c/o events")

    socket_profile = {
        "name": rospy.get_param("~socket_name", "blue_insert_socket"),
        "category": rospy.get_param("~socket_category", "shallow_circular_socket"),
        "size_m": socket_size,
        "opening_m": socket_opening,
        "surface_z_offset": float(rospy.get_param("~socket_surface_z_offset", 0.0)),
    }
    # Preserve the insertion pipeline semantics: place_xyz is the demonstrated
    # cylinder insertion target computed from the socket relation, while
    # insertion_release_pose_base_frame stores the actual right_hand release pose.
    insert_result = compute_insert_target(
        socket_xyz,
        cylinder_position_base=cylinder_xyz,
        cylinder_size=cylinder_size,
        socket_profile=socket_profile,
        override_offset_xyz=rospy.get_param("~socket_insert_offset_xyz", None))
    release_xyz = [float(v) for v in insert_result["place_xyz"]]
    offset_xyz = [float(v) for v in insert_result["offset_xyz"]]

    demo = {
        "id": demo_id,
        "format": "mt3_cylinder_insert_recorded_v1",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recording_mode": "human_kinesthetic_zero_g",
        "task_type": "cylinder_insert_socket",
        "task": rospy.get_param(
            "~task", "insert the green cylinder into the blue socket"),
        "object_info": {
            "position_base": cylinder_xyz,
            "size_m": cylinder_size,
            "category": "cylinder",
            "color": rospy.get_param("~object_color", "green"),
        },
        "anchor_info": {
            "name": socket_profile["name"],
            "category": socket_profile["category"],
            "position_base": socket_xyz,
            "size_m": socket_size,
            "opening_m": socket_opening,
            "profile": socket_profile,
            "sdf": "blue_insert_socket.sdf",
        },
        "place_info": {
            "mode": "socket_insertion",
            "place_xyz": release_xyz,
            "offset_xyz": offset_xyz,
            "socket_size": socket_size,
            "socket_opening": socket_opening,
            "place_pose_base_frame": {"position": release_xyz},
            "source": "socket_relation_geometry",
        },
        "grasp_bottleneck_pose_base_frame": grasp_bn,
        "grasp_close_pose_base_frame": grasp_close,
        "grasp_trajectory": grasp_traj,
        "insertion_bottleneck_pose_base_frame": insert_bn,
        "insertion_release_pose_base_frame": insert_release,
        "insertion_trajectory": insert_traj,
        "scene": {
            "target": {"position_base": cylinder_xyz, "method": "real_manual_param"},
            "anchor": {"position_base": socket_xyz, "method": "real_manual_param"},
        },
        "scene_packages": {},
        "real_scene_snapshot": recorder.snapshot,
        "trajectory": trajectory,
        "rollout_trajectory_path": rollout_path,
    }
    out_path = os.path.join(DEMO_DIR, "%s.json" % demo_id)
    with open(out_path, "w") as f:
        json.dump(demo, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Real insertion demo saved: %s", out_path)
    rospy.loginfo("Demo release offset to socket = [%.3f, %.3f, %.3f] m", *offset_xyz)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("record_cylinder_insert_demo_real failed: %s", exc)
        raise
