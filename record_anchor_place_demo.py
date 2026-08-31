#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record an anchored pick-place demonstration.

Baseline task:
    pick the green cube and place it on/in the blue anchor tray.

This script uses the existing stable Sawyer pick-place executor.  It sets the
runtime grasp/place ROS params, lets mt3_sawyer_place.py execute and record the
rollout, then saves a demo-library JSON with target + anchor metadata.
"""

import json
import os
import subprocess
import sys
import time
import math

import numpy as np
import rospy

from mt3_anchor_perception import DualMaskAnchorPerception
from mt3_anchor_place_generalization import compute_anchor_place_target
from mt3_relation_scene_package import save_dual_object_scene_packages


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CODE_DIR, "demo_library", "recorded")
ROLLOUT_DIR = os.path.join(CODE_DIR, "demo_library", "rollout_trajectories")
DEFAULT_TARGET_XYZ = [0.65, 0.20, -0.458]
DEFAULT_ANCHOR_XYZ = [0.65, 0.00, -0.470]
DEFAULT_TOP_FLANGE_Z_OFFSET = 0.040


def _repo_root_from_code_dir():
    return os.path.dirname(os.path.dirname(CODE_DIR))


def _executor_path():
    explicit = rospy.get_param("~executor_path", "")
    if explicit:
        return os.path.expanduser(explicit)
    home_candidate = os.path.expanduser(
        "~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_place.py")
    if os.path.exists(home_candidate):
        return home_candidate
    return os.path.join(
        _repo_root_from_code_dir(),
        "ros_ws", "src", "sawyer_gazebo", "scripts", "mt3_sawyer_place.py")


def _param_xyz(prefix, default_xyz):
    return [
        float(rospy.get_param("~%s_x" % prefix, default_xyz[0])),
        float(rospy.get_param("~%s_y" % prefix, default_xyz[1])),
        float(rospy.get_param("~%s_z" % prefix, default_xyz[2])),
    ]


def _jsonable_scene(scene):
    def convert(value):
        if isinstance(value, dict):
            return {
                str(k): convert(v)
                for k, v in value.items()
                if k not in ("rgb", "depth", "intrinsics", "object_points")
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return convert(scene or {})


def _load_scene():
    use_perception = bool(rospy.get_param("~use_perception", False))
    object_size = rospy.get_param("~object_size", [0.045, 0.045, 0.045])
    if use_perception:
        detector = DualMaskAnchorPerception()
        scene = detector.detect_scene(timeout_s=float(rospy.get_param(
            "~perception_timeout_s", 8.0)))
        if scene is None:
            raise RuntimeError("anchored perception failed")
        target_xyz = scene["target"]["position_base"]
        anchor_xyz = scene["anchor"]["position_base"]
        detected_size = scene["target"].get("estimated_size")
        if detected_size and len(detected_size) >= 3:
            object_size = detected_size
        return scene, target_xyz, anchor_xyz, object_size

    target_xyz = _param_xyz("target", DEFAULT_TARGET_XYZ)
    anchor_xyz = _param_xyz("anchor", DEFAULT_ANCHOR_XYZ)
    scene = _manual_scene_with_optional_rgbd_packages(target_xyz, anchor_xyz)
    return scene, target_xyz, anchor_xyz, object_size


def _manual_scene_with_optional_rgbd_packages(target_xyz, anchor_xyz):
    """Keep demo poses manual while saving mask/RGB-D point clouds when present."""
    scene = {
        "target": {"position_base": target_xyz, "method": "manual_param"},
        "anchor": {"position_base": anchor_xyz, "method": "manual_param"},
    }
    if not bool(rospy.get_param("~save_perception_scene_package", True)):
        return scene
    try:
        detector = DualMaskAnchorPerception()
        detected = detector.detect_scene(timeout_s=float(rospy.get_param(
            "~perception_timeout_s", 8.0)))
        if detected is None:
            rospy.logwarn(
                "Manual anchor demo: perception scene package skipped; "
                "using manual positions only.")
            return scene
        scene = detected
        scene["target"] = dict(scene.get("target") or {})
        scene["anchor"] = dict(scene.get("anchor") or {})
        scene["target"]["detected_position_base"] = scene["target"].get(
            "position_base")
        scene["anchor"]["detected_position_base"] = scene["anchor"].get(
            "position_base")
        scene["target"]["position_base"] = [float(v) for v in target_xyz]
        scene["anchor"]["position_base"] = [float(v) for v in anchor_xyz]
        scene["target"]["method"] = (
            "manual_param_pose+perception_scene_package")
        scene["anchor"]["method"] = (
            "manual_param_pose+perception_scene_package")
        if isinstance(scene["target"].get("pose_base"), dict):
            scene["target"]["pose_base"] = dict(scene["target"]["pose_base"])
            scene["target"]["pose_base"]["position"] = [
                float(v) for v in target_xyz]
        if isinstance(scene["anchor"].get("pose_base"), dict):
            scene["anchor"]["pose_base"] = dict(scene["anchor"]["pose_base"])
            scene["anchor"]["pose_base"]["position"] = [
                float(v) for v in anchor_xyz]
        rospy.loginfo(
            "Manual anchor demo: using hardcoded target/anchor poses while "
            "saving perception scene packages.")
        return scene
    except Exception as exc:
        rospy.logwarn(
            "Manual anchor demo: perception scene package failed: %s; "
            "using manual positions only.", exc)
        return scene


def _top_grasp_contact_z(object_xyz, object_size):
    object_height = float(object_size[2]) if len(object_size) >= 3 else 0.045
    clearance = float(rospy.get_param("~grasp_top_clearance", 0.005))
    return float(object_xyz[2]) + object_height + clearance


def _top_grasp_flange_z(object_xyz, object_size):
    offset = float(rospy.get_param(
        "~flange_grasp_z_offset",
        rospy.get_param(
            "/sawyer_auto_grasp/flange_grasp_z_offset",
            DEFAULT_TOP_FLANGE_Z_OFFSET)))
    return _top_grasp_contact_z(object_xyz, object_size) + offset


def _write_execution_params(target_xyz, anchor_xyz, object_size, place_result,
                            rollout_path):
    q = rospy.get_param("~grasp_orientation_xyzw", [-1.0, 0.0, 0.0, 0.0])
    grasp_z = _top_grasp_contact_z(target_xyz, object_size)
    rospy.set_param("/sawyer_auto_grasp/grasp_x", float(target_xyz[0]))
    rospy.set_param("/sawyer_auto_grasp/grasp_y", float(target_xyz[1]))
    rospy.set_param("/sawyer_auto_grasp/grasp_z", float(grasp_z))
    rospy.set_param("/sawyer_auto_grasp/grasp_qx", float(q[0]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qy", float(q[1]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qz", float(q[2]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qw", float(q[3]))
    rospy.set_param("/sawyer_auto_grasp/flange_grasp_z_offset", float(
        rospy.get_param(
            "~flange_grasp_z_offset",
            rospy.get_param(
                "/sawyer_auto_grasp/flange_grasp_z_offset",
                DEFAULT_TOP_FLANGE_Z_OFFSET))))
    rospy.set_param("/sawyer_auto_grasp/object_size", object_size)

    place_xyz = place_result["place_xyz"]
    rospy.set_param("/sawyer_auto_grasp/place_x", float(place_xyz[0]))
    rospy.set_param("/sawyer_auto_grasp/place_y", float(place_xyz[1]))
    rospy.set_param("/sawyer_auto_grasp/place_z", float(place_xyz[2]))
    rospy.set_param("/sawyer_auto_grasp/place_direction",
                    "anchor_on_blue_platform")
    rospy.set_param("/sawyer_auto_grasp/place_clearance", float(rospy.get_param(
        "~place_clearance", 0.050)))
    rospy.set_param("/sawyer_auto_grasp/place_lift_height", float(rospy.get_param(
        "~place_lift_height", 0.150)))

    rospy.set_param("/sawyer_auto_grasp/use_demo_replay", False)
    rospy.set_param("/sawyer_auto_grasp/use_place_release_replay", False)
    rospy.set_param("/sawyer_auto_grasp/demo_replay_trajectory_path", "")
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_path", rollout_path)
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_rate_hz", float(
        rospy.get_param("~trajectory_record_rate_hz", 30.0)))


def _gripper_binary(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return 1 if float(value) >= 0.5 else 0
    except Exception:
        text = str(value).strip().lower()
        if text in ("closed", "close", "closing", "1", "true"):
            return 1
        if text in ("open", "opening", "0", "false"):
            return 0
    return None


def _find_release_open_index(poses):
    last_transition = None
    prev = None
    seen_closed = False
    fallback = None
    for idx, sample in enumerate(poses):
        state = _gripper_binary(sample.get("gripper_next"))
        if state is None:
            state = _gripper_binary(sample.get("gripper_state"))
        if prev == 1 and state == 0:
            last_transition = idx
        if seen_closed and state == 0 and fallback is None:
            fallback = idx
        if state == 1:
            seen_closed = True
        if state is not None:
            prev = state
    return last_transition if last_transition is not None else fallback


def _find_grasp_close_index(poses):
    prev = None
    for idx, sample in enumerate(poses):
        state = _gripper_binary(sample.get("gripper_state"))
        if prev == 0 and state == 1:
            return idx, "gripper_state_transition"
        if state is not None:
            prev = state
    for idx, sample in enumerate(poses):
        state = _gripper_binary(sample.get("gripper_next"))
        if state == 1:
            return min(idx + 1, len(poses) - 1), "gripper_next"
    return None, ""


def _find_grasp_event_index(trajectory, pose_count):
    events = (trajectory or {}).get("events") or []
    close_names = set(["gripper_close", "grasp_close", "top_grasp_close"])
    candidate = None
    for event in events:
        if str(event.get("name", "")).strip() not in close_names:
            continue
        try:
            idx = int(event.get("sample_index"))
        except Exception:
            continue
        candidate = max(0, min(pose_count - 1, idx))
    return candidate


def _find_release_event_index(trajectory, pose_count):
    events = (trajectory or {}).get("events") or []
    release_names = set([
        "place_release_open",
        "release_open",
        "gripper_release_open",
    ])
    candidate = None
    for event in events:
        if str(event.get("name", "")).strip() not in release_names:
            continue
        try:
            idx = int(event.get("sample_index"))
        except Exception:
            continue
        candidate = max(0, min(pose_count - 1, idx))
    return candidate


def _trajectory_pose_xyz(sample):
    pos = (sample or {}).get("position", [0.0, 0.0, 0.0])
    if isinstance(pos, dict):
        return [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        ]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _pose_z_at(poses, idx):
    try:
        return float(_trajectory_pose_xyz(poses[int(idx)])[2])
    except Exception:
        return None


def _close_index_valid_for_z(poses, idx, expected_z, tolerance):
    if idx is None or expected_z is None:
        return idx is not None
    z = _pose_z_at(poses, idx)
    if z is None:
        return False
    return abs(z - float(expected_z)) <= float(tolerance)


def _find_grasp_close_index_near_z(trajectory, poses, expected_z):
    if expected_z is None:
        return None
    release_idx = _find_release_event_index(trajectory, len(poses))
    if release_idx is None:
        release_idx = _find_release_open_index(poses)
    search_end = int(release_idx) if release_idx is not None else len(poses)
    search_end = max(1, min(len(poses), search_end))
    best_idx = None
    best_err = None
    for idx in range(search_end):
        z = _pose_z_at(poses, idx)
        if z is None:
            continue
        err = abs(z - float(expected_z))
        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def _xy_aligned_grasp_start_index(poses, close_idx, max_pre):
    close_idx = int(close_idx)
    fixed_start = max(0, close_idx - max(1, int(max_pre)))
    try:
        close_xyz = _trajectory_pose_xyz(poses[close_idx])
    except Exception:
        return fixed_start
    xy_tol = float(rospy.get_param(
        "~grasp_replay_top_xy_aligned_tolerance", 0.018))
    min_pre = int(rospy.get_param("~grasp_replay_min_pre_samples", 4))
    start_idx = fixed_start
    for idx in range(close_idx - 1, fixed_start - 1, -1):
        try:
            p = _trajectory_pose_xyz(poses[idx])
            xy_dist = math.sqrt(
                (p[0] - close_xyz[0]) ** 2 +
                (p[1] - close_xyz[1]) ** 2)
        except Exception:
            continue
        if xy_dist > xy_tol:
            start_idx = idx + 1
            break
    if close_idx - start_idx < min_pre:
        rospy.logwarn(
            "Grasp trajectory XY-aligned window has only %d pre-close "
            "samples (min_pre=%d); keeping it to avoid lateral approach.",
            close_idx - start_idx, min_pre)
    return max(0, min(start_idx, close_idx - 1))


def _stamp_grasp_close_labels(segment_poses, segment_velocities,
                              local_close_idx):
    local_close_idx = int(local_close_idx)
    if not (0 <= local_close_idx < len(segment_poses)):
        return
    for i, sample in enumerate(segment_poses):
        sample["gripper_state"] = 1 if i >= local_close_idx else 0
        sample["gripper_next"] = 1 if i == local_close_idx else 0
    velocity_close_idx = local_close_idx - 1
    if 0 <= velocity_close_idx < len(segment_velocities):
        for i, sample in enumerate(segment_velocities):
            sample["gripper_next"] = 1 if i == velocity_close_idx else 0


def _extract_grasp_trajectory(trajectory, expected_close_z=None):
    poses = (trajectory or {}).get("poses", [])
    if len(poses) < 5:
        return None, None, None

    z_tolerance = float(rospy.get_param("~grasp_close_z_tolerance", 0.080))
    close_idx = _find_grasp_event_index(trajectory, len(poses))
    event_source = "explicit_event"
    if not _close_index_valid_for_z(
            poses, close_idx, expected_close_z, z_tolerance):
        if close_idx is not None and expected_close_z is not None:
            z = _pose_z_at(poses, close_idx)
            rospy.logwarn(
                "Grasp explicit close event rejected: idx=%d z=%.3f "
                "expected=%.3f err=%.1fcm",
                close_idx, z if z is not None else float("nan"),
                float(expected_close_z),
                abs((z if z is not None else 0.0) -
                    float(expected_close_z)) * 100.0)
        close_idx = None

    if close_idx is None:
        state_idx, state_source = _find_grasp_close_index(poses)
        if _close_index_valid_for_z(
                poses, state_idx, expected_close_z, z_tolerance):
            close_idx = state_idx
            event_source = state_source
        elif state_idx is not None and expected_close_z is not None:
            z = _pose_z_at(poses, state_idx)
            rospy.logwarn(
                "Grasp state close event rejected: idx=%d z=%.3f "
                "expected=%.3f err=%.1fcm",
                state_idx, z if z is not None else float("nan"),
                float(expected_close_z),
                abs((z if z is not None else 0.0) -
                    float(expected_close_z)) * 100.0)

    if close_idx is None and expected_close_z is not None:
        close_idx = _find_grasp_close_index_near_z(
            trajectory, poses, expected_close_z)
        event_source = "nearest_expected_grasp_z"

    if close_idx is None:
        rospy.logwarn("Grasp trajectory not extracted: no close event")
        return None, None, None

    close_z = _pose_z_at(poses, close_idx)
    if expected_close_z is not None and close_z is not None:
        rospy.loginfo(
            "Grasp close selected: idx=%d source=%s z=%.3f expected=%.3f "
            "err=%.1fcm",
            close_idx, event_source, close_z, float(expected_close_z),
            abs(close_z - float(expected_close_z)) * 100.0)

    pre_samples = int(rospy.get_param("~grasp_bottleneck_pre_samples", 75))
    post_samples = int(rospy.get_param("~grasp_bottleneck_post_samples", 30))
    min_pre_samples = int(rospy.get_param(
        "~grasp_replay_min_pre_samples", 4))
    start_idx = _xy_aligned_grasp_start_index(
        poses, close_idx, max(1, pre_samples))
    end_idx = min(len(poses), int(close_idx) + max(1, post_samples) + 1)
    local_close_idx = int(close_idx) - start_idx
    if local_close_idx < min_pre_samples:
        rospy.logwarn(
            "Grasp trajectory close_index=%d is short before close "
            "(min_pre=%d); replay may not include enough descent.",
            local_close_idx, min_pre_samples)

    segment_poses = [dict(sample) for sample in poses[start_idx:end_idx]]
    velocities = (trajectory or {}).get("velocities", [])
    segment_velocities = (
        [dict(sample) for sample in velocities[start_idx:end_idx]]
        if velocities else [])
    _stamp_grasp_close_labels(segment_poses, segment_velocities,
                              local_close_idx)
    base_position = (
        list(segment_poses[0].get("position", []))
        if segment_poses else [])
    grasp_trajectory = {
        "format": "mt3_anchor_grasp_trajectory_v1",
        "frame": (trajectory or {}).get("frame", "base"),
        "sample_rate_hz": (trajectory or {}).get("sample_rate_hz"),
        "source_rollout_format": (trajectory or {}).get("format"),
        "source_start_index": int(start_idx),
        "source_close_index": int(close_idx),
        "close_event_source": event_source,
        "source_end_index": int(end_idx),
        "close_index": int(local_close_idx),
        "base_index": 0,
        "base_position": [float(v) for v in base_position[:3]],
        "poses": segment_poses,
        "velocities": segment_velocities,
        "success": bool((trajectory or {}).get("success", False)),
    }
    bottleneck_pose = segment_poses[0] if segment_poses else None
    close_pose = poses[close_idx]
    rospy.loginfo(
        "Grasp trajectory extracted: source=[%d:%d] close=%d local=%d "
        "poses=%d velocities=%d",
        start_idx, end_idx, close_idx, local_close_idx,
        len(segment_poses), len(segment_velocities))
    return grasp_trajectory, bottleneck_pose, close_pose


def _extract_place_trajectory(trajectory):
    poses = (trajectory or {}).get("poses", [])
    if len(poses) < 5:
        return None, None, None

    open_idx = _find_release_event_index(trajectory, len(poses))
    event_source = "explicit_event"
    if open_idx is None:
        open_idx = _find_release_open_index(poses)
        event_source = "gripper_state_transition"
    if open_idx is None:
        rospy.logwarn("Place trajectory not extracted: no release event")
        return None, None, None

    pre_samples = int(rospy.get_param("~place_bottleneck_pre_samples", 75))
    post_samples = int(rospy.get_param("~place_bottleneck_post_samples", 165))
    start_idx = max(0, int(open_idx) - max(1, pre_samples))
    end_idx = min(len(poses), int(open_idx) + max(1, post_samples) + 1)

    segment_poses = poses[start_idx:end_idx]
    velocities = (trajectory or {}).get("velocities", [])
    segment_velocities = velocities[start_idx:end_idx] if velocities else []
    place_trajectory = {
        "format": "mt3_anchor_place_terminal_trajectory_v1",
        "frame": (trajectory or {}).get("frame", "base"),
        "sample_rate_hz": (trajectory or {}).get("sample_rate_hz"),
        "source_rollout_format": (trajectory or {}).get("format"),
        "source_start_index": start_idx,
        "source_release_index": int(open_idx),
        "release_event_source": event_source,
        "source_end_index": end_idx,
        "release_index": int(open_idx) - start_idx,
        "poses": segment_poses,
        "velocities": segment_velocities,
        "success": bool((trajectory or {}).get("success", False)),
    }
    bottleneck_pose = segment_poses[0] if segment_poses else None
    release_pose = poses[open_idx]
    return place_trajectory, bottleneck_pose, release_pose


def _run_executor():
    script = _executor_path()
    if not os.path.exists(script):
        raise RuntimeError("executor script not found: %s" % script)
    proc = subprocess.Popen(
        ["python", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            rospy.loginfo("[anchor-record executor] %s", line)
    proc.wait()
    return proc.returncode == 0


def main():
    rospy.init_node("record_anchor_place_demo", anonymous=True)
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(ROLLOUT_DIR, exist_ok=True)

    demo_id = rospy.get_param("~demo_id", "cube_place_on_blue_platform_10cm")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rollout_path = os.path.join(ROLLOUT_DIR, "%s_rollout_%s.json" % (
        demo_id, stamp))

    scene, target_xyz, anchor_xyz, object_size = _load_scene()
    grasp_z = _top_grasp_contact_z(target_xyz, object_size)
    grasp_flange_z = _top_grasp_flange_z(target_xyz, object_size)
    anchor_profile = {
        "name": rospy.get_param("~anchor_name", "blue_placement_platform"),
        "category": rospy.get_param("~anchor_category", "small_platform"),
        "surface_z_offset": float(rospy.get_param("~anchor_surface_z_offset", 0.0)),
    }
    place_result = compute_anchor_place_target(
        anchor_xyz,
        object_position_base=target_xyz,
        object_size=object_size,
        anchor_profile=anchor_profile,
        override_offset_xyz=rospy.get_param("~anchor_place_offset_xyz", None))
    scene_packages = save_dual_object_scene_packages(
        scene,
        demo_id,
        role="recorded_anchor_place_demo",
        target_label=rospy.get_param("~target_label", "green_cube"),
        anchor_label=anchor_profile["name"],
        relation_kind="anchor_pick_place",
        extra_metadata={
            "demo_id": demo_id,
            "task_type": "anchor_pick_place",
            "object_size": [float(v) for v in object_size],
            "anchor_profile": anchor_profile,
            "place_target_xyz": [float(v) for v in place_result["place_xyz"]],
            "place_offset_xyz": [float(v) for v in place_result["offset_xyz"]],
        })

    _write_execution_params(
        target_xyz, anchor_xyz, object_size, place_result, rollout_path)

    dry_run = bool(rospy.get_param("~dry_run", False))
    success = True if dry_run else _run_executor()
    if not success:
        raise RuntimeError("anchor demonstration execution failed")

    trajectory = {
        "format": "not_executed_dry_run",
        "frame": "base",
        "poses": [],
        "velocities": [],
        "success": bool(success),
    }
    if os.path.exists(rollout_path):
        with open(rollout_path, "r") as f:
            trajectory = json.load(f)
    grasp_trajectory, grasp_bottleneck, grasp_close = (
        _extract_grasp_trajectory(
            trajectory, expected_close_z=grasp_flange_z))
    place_trajectory, place_bottleneck, place_release = (
        _extract_place_trajectory(trajectory))

    demo = {
        "id": demo_id,
        "format": "mt3_anchor_recorded_v1",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "anchor_pick_place",
        "task": rospy.get_param(
            "~task", "pick the green cube and place it on the blue platform"),
        "object_info": {
            "position_base": [float(v) for v in target_xyz],
            "size_m": [float(v) for v in object_size],
            "category": rospy.get_param("~object_category", "cube"),
            "color": rospy.get_param("~object_color", "green"),
        },
        "anchor_info": {
            "name": anchor_profile["name"],
            "category": anchor_profile["category"],
            "position_base": [float(v) for v in anchor_xyz],
            "sdf": "blue_placement_platform.sdf",
        },
        "place_info": {
            "mode": "anchor",
            "place_xyz": [float(v) for v in place_result["place_xyz"]],
            "offset_xyz": [float(v) for v in place_result["offset_xyz"]],
            "anchor_profile": anchor_profile,
            "place_pose_base_frame": {
                "position": [float(v) for v in place_result["place_xyz"]],
            },
        },
        "grasp_pose_base_frame": {
            "position_m": {
                "x": float(target_xyz[0]),
                "y": float(target_xyz[1]),
                "z": float(grasp_z),
            },
            "expected_flange_z": float(grasp_flange_z),
            "orientation_xyzw": {
                "x": -1.0,
                "y": 0.0,
                "z": 0.0,
                "w": 0.0,
            },
        },
        "grasp_bottleneck_pose_base_frame": grasp_bottleneck or {
            "position": [
                float(target_xyz[0]),
                float(target_xyz[1]),
                float(grasp_z) + 0.150,
            ],
            "orientation": [-1.0, 0.0, 0.0, 0.0],
        },
        "grasp_close_pose_base_frame": grasp_close,
        "grasp_trajectory": grasp_trajectory,
        "place_bottleneck_pose_base_frame": place_bottleneck,
        "place_release_pose_base_frame": place_release,
        "place_trajectory": place_trajectory,
        "scene": _jsonable_scene(scene),
        "scene_packages": scene_packages,
        "trajectory": trajectory,
        "rollout_trajectory_path": rollout_path,
    }

    out_path = os.path.join(DEMO_DIR, "%s.json" % demo_id)
    with open(out_path, "w") as f:
        json.dump(demo, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Anchor demo saved: %s", out_path)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("record_anchor_place_demo failed: %s", exc)
        sys.exit(1)
