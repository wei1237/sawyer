#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchored MT3-style pick-place generalization.

Minimal baseline:
  1. Detect target object and anchor object from two masks.
  2. Load an anchored demonstration from demo_library/recorded.
  3. Transfer the demo grasp relative to the target object.
  4. Transfer the demo place pose relative to the anchor object.
  5. Execute with mt3_sawyer_place.py and replay the demo release segment.

This is intentionally separate from the existing no-anchor mt3_pipeline.py so
the current working experiments remain stable.
"""

import json
import os
import subprocess
import sys
import time
import csv
import math
import shutil

import numpy as np
import rospy

from mt3_demo_library import DemoLibrary
from mt3_alignment import TrajectoryAligner, pose_compose, quat_multiply
from mt3_anchor_perception import DualMaskAnchorPerception
from mt3_anchor_place_generalization import (
    compute_anchor_place_target,
    compute_target_displacement_place_target,
)
from mt3_icp_registration import save_icp_outputs
from mt3_relation_scene_package import (
    SCENE_PACKAGE_DIR,
    save_dual_object_scene_packages,
)


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CODE_DIR, "demo_library", "recorded")
ROLLOUT_DIR = os.path.join(CODE_DIR, "demo_library", "rollout_trajectories")
SHARED_EXPERIMENT_LOG_DIR = (
    "/mnt/hgfs2/code/learning_thousand_tasks/demo_library/experiment_logs"
)
EXPERIMENT_LOG_DIR = (
    os.path.join(SHARED_EXPERIMENT_LOG_DIR, "anchor_place")
    if os.path.isdir(os.path.dirname(SHARED_EXPERIMENT_LOG_DIR))
    else os.path.join(
        CODE_DIR, "demo_library", "experiment_logs", "anchor_place")
)


def _param_bool(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _execution_environment():
    text = str(rospy.get_param(
        "~execution_environment",
        os.environ.get("MT3_EXECUTION_ENVIRONMENT", "simulation"))
    ).strip().lower()
    if text in ("real", "robot", "sawyer_real", "physical"):
        return "real"
    if text in ("sim", "simulation", "gazebo", ""):
        return "simulation"
    return text


def _demo_recorded_dir():
    env = _execution_environment()
    env_dir = os.path.join(CODE_DIR, "demo_library", env, "recorded")
    if os.path.isdir(env_dir):
        return env_dir
    if env == "simulation":
        return DEMO_DIR
    return env_dir


def _reset_executor_timing_params():
    for name, value in [
            ("/sawyer_auto_grasp/planning_time_s", 0.0),
            ("/sawyer_auto_grasp/robot_execution_time_s", 0.0),
            ("/sawyer_auto_grasp/planning_call_count", 0),
            ("/sawyer_auto_grasp/robot_execution_call_count", 0),
            ("/sawyer_auto_grasp/timing_source", "parent_reset")]:
        try:
            rospy.set_param(name, value)
        except Exception:
            pass


def _read_executor_timing_params():
    def _float_param(name):
        try:
            return float(rospy.get_param(name, ""))
        except Exception:
            return ""

    def _int_param(name):
        try:
            return int(rospy.get_param(name, ""))
        except Exception:
            return ""

    return {
        "planning_time_s": _float_param("/sawyer_auto_grasp/planning_time_s"),
        "robot_execution_time_s": _float_param(
            "/sawyer_auto_grasp/robot_execution_time_s"),
        "planning_call_count": _int_param(
            "/sawyer_auto_grasp/planning_call_count"),
        "robot_execution_call_count": _int_param(
            "/sawyer_auto_grasp/robot_execution_call_count"),
        "timing_source": str(rospy.get_param(
            "/sawyer_auto_grasp/timing_source", "")),
    }


def _failure_category(stage, reason_text):
    text = ("%s %s" % (stage or "", reason_text or "")).lower()
    if not text.strip():
        return ""
    if any(k in text for k in ["perception", "mask", "detect", "pointcloud"]):
        return "perception_or_pose_failure"
    if any(k in text for k in ["planning", "no motion plan", "plan failed"]):
        return "motion_planning_failure"
    if any(k in text for k in ["controller", "control_failed", "path_tolerance", "aborted"]):
        return "controller_execution_failure"
    if "replay" in text or "bottleneck" in text:
        return "replay_failure"
    if any(k in text for k in ["place", "release", "placement", "verification", "postcheck"]):
        return "placement_failure"
    return "other_execution_failure"


def _place_failure_detail(failure_category, reason_text, target_error_xy, anchor_error_xy):
    text = str(reason_text or "").lower()
    if not failure_category:
        return ""
    try:
        if target_error_xy != "" and float(target_error_xy) > 0.025:
            return "target_misaligned"
    except Exception:
        pass
    try:
        if anchor_error_xy != "" and float(anchor_error_xy) > 0.025:
            return "anchor_misaligned"
    except Exception:
        pass
    if "planning" in text or "no motion plan" in text or "plan failed" in text:
        return "motion_planning_failure"
    if "replay" in text:
        return "place_replay_failure"
    if any(k in text for k in ["release", "place", "placement"]):
        return "release_or_place_execution_failed"
    return failure_category


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


def _json_vec(values, length=3):
    out = []
    values = values or []
    for i in range(length):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append("")
    return json.dumps(out, ensure_ascii=False)


def _xy_only_xyz(values):
    values = values or []
    out = []
    for i in range(2):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append("")
    out.append("")
    return json.dumps(out, ensure_ascii=False)


def _xy_error(est, gt):
    try:
        if est is None or gt is None:
            return ""
        return math.sqrt((float(est[0]) - float(gt[0])) ** 2
                         + (float(est[1]) - float(gt[1])) ** 2)
    except Exception:
        return ""


def _quat_yaw_deg(q):
    try:
        if q is None or len(q) < 4:
            return ""
        x, y, z, w = [float(v) for v in q[:4]]
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z))
        return math.degrees(yaw)
    except Exception:
        return ""


def _wrap_degrees(angle):
    try:
        value = float(angle)
    except Exception:
        return ""
    while value > 180.0:
        value -= 360.0
    while value < -180.0:
        value += 360.0
    return value


def _yaw_error_deg(final_yaw, target_yaw, symmetry_deg=None):
    try:
        raw = abs(float(_wrap_degrees(float(final_yaw) - float(target_yaw))))
        if symmetry_deg is None or float(symmetry_deg) <= 0.0:
            return raw
        period = float(symmetry_deg)
        candidates = []
        for k in range(-4, 5):
            candidates.append(abs(_wrap_degrees(
                float(final_yaw) - float(target_yaw) + k * period)))
        return min(candidates)
    except Exception:
        return ""


def _default_place_yaw_symmetry_deg():
    shape = str(rospy.get_param("~object_shape", "cube")).lower()
    if shape in ("cube", "square", "box"):
        return 90.0
    if shape in ("rectangular_prism", "cuboid", "rectangle"):
        return 180.0
    return 360.0


def _gazebo_pose(model_param, fallback_keywords):
    try:
        from gazebo_msgs.msg import ModelStates
        explicit = str(rospy.get_param(model_param, "")).strip()
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=0.5)
        names = list(msg.name)
        chosen = None
        if explicit and explicit in names:
            chosen = explicit
        elif explicit:
            rospy.logwarn(
                "Gazebo model '%s' not found for %s; trying keyword fallback.",
                explicit, model_param)
        else:
            keywords = [str(k).lower() for k in fallback_keywords if k]

            def _score(name):
                low = name.lower()
                if any(skip in low for skip in ["sawyer", "table", "workbench", "ground"]):
                    return -100
                return sum(1 for key in keywords if key in low)

            ranked = sorted(names, key=_score, reverse=True)
            if ranked and _score(ranked[0]) > 0:
                chosen = ranked[0]
        if not chosen:
            rospy.logwarn(
                "Gazebo pose unavailable for %s; models=%s",
                model_param, names)
            return None
        pose = msg.pose[names.index(chosen)]
        return {
            "name": chosen,
            "xyz": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }
    except Exception:
        return None


def _ros_float(param_name):
    try:
        if rospy.has_param(param_name):
            val = rospy.get_param(param_name)
            return float(val) if val != "" and val is not None else ""
    except Exception:
        pass
    return ""


def _ros_vec(param_name, expected_len):
    try:
        if rospy.has_param(param_name):
            val = rospy.get_param(param_name)
            if isinstance(val, (list, tuple)) and len(val) >= expected_len:
                return [float(v) for v in val[:expected_len]]
    except Exception:
        pass
    return ["", ""]


def _simple_xy_distance(target_xyz, anchor_xyz):
    """Euclidean xy distance between two points (no offset subtraction)."""
    try:
        if target_xyz is None or anchor_xyz is None:
            return ""
        if len(target_xyz) < 2 or len(anchor_xyz) < 2:
            return ""
        dx = float(target_xyz[0]) - float(anchor_xyz[0])
        dy = float(target_xyz[1]) - float(anchor_xyz[1])
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return ""


def _list_min_float(values):
    try:
        nums = [float(v) for v in (values or [])]
        return min(nums) if nums else ""
    except Exception:
        return ""


def _precise_place_success(center_error_m, object_size, anchor_size):
    try:
        if center_error_m == "":
            return ""
        platform_min = _list_min_float((anchor_size or [])[:2])
        object_max = max(float(object_size[0]), float(object_size[1]))
        if platform_min == "":
            return ""
        default_limit = max(0.0, 0.5 * (float(platform_min) - object_max))
        limit = float(rospy.get_param(
            "~precise_place_max_center_error_m", default_limit))
        return bool(float(center_error_m) <= limit)
    except Exception:
        return ""


def _relation_xy_error(target_xyz, anchor_xyz, desired_offset_xyz):
    try:
        if target_xyz is None or anchor_xyz is None or desired_offset_xyz is None:
            return ""
        dx = (float(target_xyz[0]) - float(anchor_xyz[0])) - float(desired_offset_xyz[0])
        dy = (float(target_xyz[1]) - float(anchor_xyz[1])) - float(desired_offset_xyz[1])
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return ""


def _validate_post_anchor_place_success(place_result, initial_anchor_xyz):
    """Verify final target placement using Gazebo ground truth."""
    target_final = _gazebo_pose(
        "~target_gt_model", ["grasp_object", "green", "cube", "object"])
    anchor_final = _gazebo_pose(
        "~anchor_gt_model",
        ["blue_placement_platform", "blue", "platform", "anchor"])
    target_xyz = (target_final or {}).get("xyz")
    anchor_xyz = (anchor_final or {}).get("xyz")
    target_q = (target_final or {}).get("orientation_xyzw")
    target_yaw_deg = _quat_yaw_deg(target_q)
    place_xyz = place_result.get("place_xyz", [])
    desired_offset = place_result.get("offset_xyz", [])
    if target_xyz is None:
        rospy.logwarn(
            "POST-ANCHOR-PLACE CHECK: target Gazebo pose unavailable; "
            "keeping executor result.")
        return {
            "ok": True,
            "failure_stage": "",
            "failure_reason": "",
            "postcheck_success": "",
            "postcheck_reason": "gazebo_pose_unavailable",
            "final_object_model_name": "",
            "final_anchor_model_name": (anchor_final or {}).get("name", ""),
            "final_object_xyz": None,
            "final_object_orientation_xyzw": None,
            "final_object_yaw_deg": "",
            "final_anchor_xyz": anchor_xyz,
            "final_target_error_xy_m": "",
            "final_relation_error_xy_m": "",
            "insert_depth_m": "",
        }

    final_place_error = _xy_error(target_xyz, place_xyz)
    final_relation_error = _relation_xy_error(
        target_xyz, anchor_xyz or initial_anchor_xyz, desired_offset)
    max_place_error = float(rospy.get_param("~post_place_max_xy_error_m", 0.060))
    max_relation_error = float(rospy.get_param(
        "~post_anchor_relation_max_xy_error_m", max_place_error))

    place_ok = final_place_error != "" and float(final_place_error) <= max_place_error
    relation_ok = (
        final_relation_error == ""
        or float(final_relation_error) <= max_relation_error)
    ok = bool(place_ok and relation_ok)
    reason = "" if ok else "目标物体最终放置位置或相对托盘关系偏差过大"
    rospy.loginfo(
        "POST-ANCHOR-PLACE CHECK: place_err=%s max=%.3f relation_err=%s "
        "max_rel=%.3f ok=%s",
        ("%.3f" % final_place_error if final_place_error != "" else "n/a"),
        max_place_error,
        ("%.3f" % final_relation_error if final_relation_error != "" else "n/a"),
        max_relation_error,
        ok)
    return {
        "ok": ok,
        "failure_stage": "" if ok else "placement_verification",
        "failure_reason": reason,
        "postcheck_success": ok,
        "postcheck_reason": reason,
        "final_object_model_name": (target_final or {}).get("name", ""),
        "final_anchor_model_name": (anchor_final or {}).get("name", ""),
        "final_object_xyz": target_xyz,
        "final_object_orientation_xyzw": target_q,
        "final_object_yaw_deg": target_yaw_deg,
        "final_anchor_xyz": anchor_xyz,
        "final_target_error_xy_m": final_place_error,
        "final_relation_error_xy_m": final_relation_error,
        "insert_depth_m": "",
    }


def _mask_pixels(path):
    try:
        import numpy as np
        return int(np.count_nonzero(np.load(path)))
    except Exception:
        return ""


def _point_count(obj):
    try:
        return len(((obj.get("pose_source") or {}).get("object_points")) or [])
    except Exception:
        return ""


def _replay_info(replay_path):
    replay_type = ""
    release_index = ""
    if replay_path and os.path.exists(replay_path):
        try:
            with open(replay_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            replay_type = payload.get("trajectory_source", payload.get("format", ""))
            traj = payload.get("trajectory", {}) or {}
            release_index = payload.get("release_index", traj.get("release_index", ""))
        except Exception:
            pass
    return replay_type, release_index


def _load_scene(demo=None):
    object_size = rospy.get_param("~object_size", [0.045, 0.045, 0.045])
    if demo is not None and not rospy.has_param("~anchor_plane_z"):
        demo_anchor = _demo_position(demo, "anchor_info", fallback=None)
        if demo_anchor is not None and len(demo_anchor) >= 3:
            rospy.set_param("~anchor_plane_z", float(demo_anchor[2]))
    if _param_bool("~use_perception", True):
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

    target_xyz = _param_xyz("target", [0.60, 0.00, -0.58])
    anchor_xyz = _param_xyz("anchor", [0.60, -0.18, -0.58])
    scene = {
        "target": {"position_base": target_xyz, "method": "manual_param"},
        "anchor": {"position_base": anchor_xyz, "method": "manual_param"},
    }
    return scene, target_xyz, anchor_xyz, object_size


def _demo_position(demo, key, fallback=None):
    block = demo.get(key, {}) or {}
    if "position_base" in block:
        return [float(v) for v in block["position_base"][:3]]
    if "position" in block:
        pos = block.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    pos_m = block.get("position_m", {})
    if pos_m:
        return [float(pos_m["x"]), float(pos_m["y"]), float(pos_m["z"])]
    if fallback is not None:
        return [float(v) for v in fallback[:3]]
    return None


def _trajectory_pose_xyz(sample):
    pos = (sample or {}).get("position", [0.0, 0.0, 0.0])
    if isinstance(pos, dict):
        return [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        ]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _xy_aligned_grasp_start_index(poses, close_idx, max_pre):
    """Find the top-grasp descent start after XY is already aligned."""
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
            "Grasp replay: XY-aligned top-grasp window has only %d "
            "pre-close samples (min_pre=%d); keeping it to avoid lateral "
            "approach replay.",
            close_idx - start_idx, min_pre)
    return max(0, min(start_idx, close_idx - 1))


def _demo_grasp_pose(demo):
    gp = demo.get("grasp_pose_base_frame", {}) or {}
    pos_m = gp.get("position_m", {})
    ori = gp.get("orientation_xyzw", {})
    position = [
        float(pos_m.get("x", 0.60)),
        float(pos_m.get("y", 0.00)),
        float(pos_m.get("z", -0.58)),
    ]
    orientation = [
        float(ori.get("x", -1.0)),
        float(ori.get("y", 0.0)),
        float(ori.get("z", 0.0)),
        float(ori.get("w", 0.0)),
    ]
    return position, orientation


def _position_m_block(xyz):
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
    }


def _demo_entry_for_alignment(demo, live_target_xyz):
    """Build the same demo fields expected by TrajectoryAligner."""
    entry = dict(demo)
    if "object_pose_base_frame" not in entry:
        demo_obj = _demo_position(demo, "object_info", fallback=live_target_xyz)
        entry["object_pose_base_frame"] = {
            "position_m": _position_m_block(demo_obj),
            "orientation_xyzw": {
                "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
            },
        }
    return entry


def _is_top_down_grasp(aligned):
    approach = (aligned or {}).get("approach_direction")
    if approach is None:
        return True
    try:
        return (
            abs(float(approach[0])) < 1e-4
            and abs(float(approach[1])) < 1e-4
            and float(approach[2]) < -0.5
        )
    except Exception:
        return True


def _apply_anchor_height_aware_top_grasp(aligned):
    """Mirror mt3_pipeline height-aware top grasp z mapping."""
    if not _param_bool("~use_height_aware_top_grasp", True):
        return
    if not _is_top_down_grasp(aligned):
        rospy.loginfo("  height-aware top grasp skipped: non-top grasp demo")
        return
    obj = aligned.get("object_pose_base", {}) or {}
    grasp = aligned.get("grasp_pose", {}) or {}
    obj_pos = obj.get("position")
    grasp_pos = grasp.get("position")
    if not obj_pos or not grasp_pos or len(obj_pos) < 3 or len(grasp_pos) < 3:
        return
    height = obj.get("estimated_object_height", None)
    if height is None:
        size = obj.get("estimated_object_size", None)
        if size and len(size) == 3:
            height = max(float(size[2]), float(size[0]), float(size[1]))
    try:
        height = float(height)
    except Exception:
        return
    if height <= 0:
        return
    current_offset = float(grasp_pos[2]) - float(obj_pos[2])
    clearance = float(rospy.get_param("~top_grasp_clearance", 0.030))
    desired_offset = max(current_offset, height * 0.5 + clearance)
    if desired_offset > current_offset + 1e-4:
        old_z = grasp_pos[2]
        grasp_pos[2] = float(obj_pos[2]) + desired_offset
        rospy.loginfo(
            "  height-aware top grasp z: %.4f -> %.4f "
            "(height=%.4f, clearance=%.4f, offset=%.4f)",
            old_z, grasp_pos[2], height, clearance, desired_offset)


def _package_has_pointcloud(package):
    if not package:
        return False
    package_dir = package.get("package_dir")
    if not package_dir:
        return False
    path = os.path.join(package_dir, "pointcloud.npy")
    if not os.path.exists(path):
        return False
    try:
        points = np.load(path)
        return points.ndim == 2 and points.shape[1] == 3 and len(points) >= 20
    except Exception:
        return False


def _target_package_from_recorded_demo(demo):
    packages = (demo or {}).get("scene_packages") or {}
    target_pkg = packages.get("target_package")
    if _package_has_pointcloud(target_pkg):
        rospy.loginfo(
            "  ICP using recorded anchor-place target package: %s",
            target_pkg["package_dir"])
        return target_pkg

    demo_id = (demo or {}).get("id", "")
    candidates = []
    if demo_id:
        candidates.extend([
            "recorded_anchor_place_demo_%s_target" % demo_id,
            "demo_%s_target" % demo_id,
            "demo_%s" % demo_id,
        ])
    explicit = rospy.get_param("~icp_demo_package_name", "")
    if explicit:
        candidates.insert(0, explicit)
    for name in candidates:
        package = {
            "package_dir": os.path.join(SCENE_PACKAGE_DIR, name),
            "name": name,
        }
        if _package_has_pointcloud(package):
            rospy.loginfo("  ICP using demo target package: %s",
                          package["package_dir"])
            return package
    if demo_id:
        rospy.logwarn("  ICP demo target package not found for demo_id=%s", demo_id)
    return None


def _run_anchor_icp_registration(demo_package, live_package, trial_id):
    if not _param_bool("~run_icp", True):
        return None
    if not demo_package or not live_package:
        return None
    try:
        icp_dir = os.path.join(SCENE_PACKAGE_DIR, "icp_%s_target" % trial_id)
        result = save_icp_outputs(
            demo_package["package_dir"],
            live_package["package_dir"],
            icp_dir)
        result["output_dir"] = icp_dir
        metrics = result.get("metrics", {})
        rospy.loginfo(
            "  Anchor target ICP saved: %s (median=%.4fm, p90=%.4fm, iters=%d)",
            icp_dir,
            metrics.get("median_error_m", -1.0),
            metrics.get("p90_error_m", -1.0),
            metrics.get("iterations", 0))
        return result
    except Exception as exc:
        rospy.logwarn("  Anchor target ICP failed: %s", exc)
        return None


def _package_source_frame(package_dir):
    meta_path = os.path.join(package_dir, "metadata.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pose = meta.get("pose") or {}
        return pose.get("source_frame") or pose.get("frame") or "head_camera"
    except Exception:
        return "head_camera"


def _transform_points_to_base(points, source_frame, aligner=None, min_points=10):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < int(min_points):
        return None
    if not source_frame:
        source_frame = "head_camera"
    aligner = aligner or TrajectoryAligner()
    if getattr(aligner, "_tf_buffer", None) is None:
        return None
    try:
        tf = aligner._tf_buffer.lookup_transform(
            "base", source_frame, rospy.Time(0), rospy.Duration(2.0))
        trans = tf.transform.translation
        rot = tf.transform.rotation
        tf_pos = [trans.x, trans.y, trans.z]
        tf_ori = [rot.x, rot.y, rot.z, rot.w]
        if len(points) > 1800:
            idx = np.linspace(0, len(points) - 1, 1800).astype(np.int64)
            points = points[idx]
        base_points = []
        identity = [0.0, 0.0, 0.0, 1.0]
        for point in points:
            pos_base, _ = pose_compose(
                tf_pos, tf_ori,
                [float(point[0]), float(point[1]), float(point[2])],
                identity)
            base_points.append(pos_base)
        return np.asarray(base_points, dtype=np.float64)
    except Exception as exc:
        rospy.logwarn("  Anchor pointcloud TF transform failed (%s -> base): %s",
                      source_frame, exc)
        return None


def _estimate_package_long_axis_yaw_base(package_dir, aligner=None):
    try:
        points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
    except Exception as exc:
        rospy.logwarn("  Anchor pointcloud yaw: cannot read %s: %s",
                      package_dir, exc)
        return None
    base_points = _transform_points_to_base(
        points, _package_source_frame(package_dir), aligner=aligner)
    if base_points is None or len(base_points) < 10:
        return None
    xy = base_points[:, :2]
    center = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - center, axis=1)
    if len(dist) >= 20:
        xy = xy[dist <= np.percentile(dist, 92)]
    if len(xy) < 10:
        return None
    centered = xy - np.mean(xy, axis=0)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _normalize_parallel_gripper_yaw_delta(yaw):
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    if yaw > math.pi / 2.0:
        yaw -= math.pi
    elif yaw < -math.pi / 2.0:
        yaw += math.pi
    return yaw


def _yaw_quaternion(yaw):
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _apply_anchor_rectangular_prism_yaw_alignment(aligned, icp_result, aligner):
    if not _param_bool("~use_pointcloud_yaw", True):
        return
    shape = str(rospy.get_param("~object_shape", "cube")).lower()
    enabled_shapes = [
        str(s).lower()
        for s in rospy.get_param(
            "~pointcloud_yaw_shapes", ["rectangular_prism", "cuboid"])
    ]
    if shape not in enabled_shapes:
        return
    grasp = aligned.get("grasp_pose", {})
    if not grasp.get("orientation"):
        return
    if not icp_result:
        rospy.logwarn("  anchor pointcloud yaw alignment skipped: no ICP result")
        return
    demo_package = icp_result.get("demo_package")
    live_package = icp_result.get("live_package")
    if not demo_package or not live_package:
        return
    demo_yaw = _estimate_package_long_axis_yaw_base(demo_package, aligner=aligner)
    live_yaw = _estimate_package_long_axis_yaw_base(live_package, aligner=aligner)
    if demo_yaw is None or live_yaw is None:
        rospy.logwarn(
            "  anchor pointcloud yaw alignment skipped: yaw estimate unavailable")
        return
    delta = _normalize_parallel_gripper_yaw_delta(live_yaw - demo_yaw)
    if abs(delta) < math.radians(2.0):
        aligned["pointcloud_yaw_alignment"] = {
            "demo_yaw_rad": float(demo_yaw),
            "live_yaw_rad": float(live_yaw),
            "delta_yaw_rad": float(delta),
            "demo_yaw_deg": float(math.degrees(demo_yaw)),
            "live_yaw_deg": float(math.degrees(live_yaw)),
            "delta_yaw_deg": float(math.degrees(delta)),
        }
        rospy.loginfo(
            "  anchor pointcloud yaw alignment: demo=%.1fdeg live=%.1fdeg "
            "delta=%.1fdeg (no change)",
            math.degrees(demo_yaw), math.degrees(live_yaw), math.degrees(delta))
        return
    old_ori = [float(v) for v in grasp["orientation"]]
    new_ori = quat_multiply(_yaw_quaternion(delta), old_ori)
    norm = math.sqrt(sum(v * v for v in new_ori))
    if norm > 1e-9:
        new_ori = [float(v / norm) for v in new_ori]
    grasp["orientation"] = new_ori
    if aligned.get("bottleneck_pose"):
        aligned["bottleneck_pose"]["orientation"] = list(new_ori)
    aligned["pointcloud_yaw_alignment"] = {
        "demo_yaw_rad": float(demo_yaw),
        "live_yaw_rad": float(live_yaw),
        "delta_yaw_rad": float(delta),
        "demo_yaw_deg": float(math.degrees(demo_yaw)),
        "live_yaw_deg": float(math.degrees(live_yaw)),
        "delta_yaw_deg": float(math.degrees(delta)),
    }
    rospy.loginfo(
        "  anchor pointcloud yaw alignment: demo=%.1fdeg live=%.1fdeg "
        "delta=%.1fdeg",
        math.degrees(demo_yaw), math.degrees(live_yaw), math.degrees(delta))


def _estimate_package_obb_center_base(package_dir, aligner=None):
    try:
        points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
    except Exception as exc:
        rospy.logwarn("  Anchor OBB center: cannot read %s: %s", package_dir, exc)
        return None
    base_points = _transform_points_to_base(
        points, _package_source_frame(package_dir), aligner=aligner)
    if base_points is None or len(base_points) < 20:
        return None
    xy = base_points[:, :2]
    center = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - center, axis=1)
    if len(dist) >= 30:
        xy = xy[dist <= np.percentile(dist, 92)]
    if len(xy) < 20:
        return None
    centered = xy - np.mean(xy, axis=0)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = centered @ axes
    lo = np.percentile(projected, 2, axis=0)
    hi = np.percentile(projected, 98, axis=0)
    mid = 0.5 * (lo + hi)
    obb_center = np.mean(xy, axis=0) + mid @ axes.T
    extents = hi - lo
    return {
        "x": float(obb_center[0]),
        "y": float(obb_center[1]),
        "extent_long": float(max(extents)),
        "extent_short": float(min(extents)),
    }


def _apply_anchor_rectangular_prism_obb_center_alignment(aligned, icp_result,
                                                         aligner):
    shape = str(rospy.get_param("~object_shape", "cube")).lower()
    enabled_shapes = [
        str(s).lower()
        for s in rospy.get_param(
            "~pointcloud_yaw_shapes", ["rectangular_prism", "cuboid"])
    ]
    if shape not in enabled_shapes or not icp_result:
        return False
    live_package = icp_result.get("live_package")
    if not live_package:
        return False
    center = _estimate_package_obb_center_base(live_package, aligner=aligner)
    if center is None:
        rospy.logwarn("  anchor pointcloud OBB center alignment skipped")
        return False
    obj = aligned.get("object_pose_base", {})
    grasp = aligned.get("grasp_pose", {})
    obj_pos = obj.get("position")
    grasp_pos = grasp.get("position")
    if not obj_pos or not grasp_pos or len(obj_pos) < 2 or len(grasp_pos) < 2:
        return False
    old_x, old_y = float(obj_pos[0]), float(obj_pos[1])
    dx = float(center["x"]) - old_x
    dy = float(center["y"]) - old_y
    if math.sqrt(dx * dx + dy * dy) > 0.06:
        rospy.logwarn(
            "  anchor pointcloud OBB center alignment skipped: correction too "
            "large (dx=%.3f dy=%.3f)", dx, dy)
        return False
    obj_pos[0] += dx
    obj_pos[1] += dy
    grasp_pos[0] += dx
    grasp_pos[1] += dy
    if aligned.get("bottleneck_pose"):
        bn_pos = aligned["bottleneck_pose"].get("position")
        if bn_pos and len(bn_pos) >= 2:
            bn_pos[0] += dx
            bn_pos[1] += dy
    aligned["pointcloud_obb_center_alignment"] = {
        "old_x": old_x,
        "old_y": old_y,
        "obb_center_x": float(center["x"]),
        "obb_center_y": float(center["y"]),
        "dx": float(dx),
        "dy": float(dy),
        "extent_long": center.get("extent_long", ""),
        "extent_short": center.get("extent_short", ""),
    }
    rospy.loginfo(
        "  anchor pointcloud OBB center alignment: xy [%.4f, %.4f] -> "
        "[%.4f, %.4f] (dx=%.3f dy=%.3f extents=%.3fx%.3f)",
        old_x, old_y, center["x"], center["y"], dx, dy,
        center.get("extent_long", 0.0), center.get("extent_short", 0.0))
    return True


def _robust_base_z_top(package_dir, aligner=None, percentile=90.0):
    try:
        meta_path = os.path.join(package_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj_pos = meta.get("position_base") or meta.get("object_position_base")
        obj_size = meta.get("estimated_size") or meta.get("object_size")
        if obj_pos and obj_size and len(obj_pos) >= 3 and len(obj_size) >= 3:
            return float(obj_pos[2]) + float(obj_size[2])
    except Exception:
        pass
    try:
        points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
    except Exception:
        return None
    base_points = _transform_points_to_base(
        points, _package_source_frame(package_dir), aligner=aligner)
    if base_points is None or len(base_points) < 10:
        return None
    z = base_points[:, 2]
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return None
    return float(np.percentile(z, percentile))


def _apply_anchor_icp_top_surface_grasp_z(aligned, demo, icp_result, aligner):
    if not _is_top_down_grasp(aligned):
        rospy.loginfo(
            "  anchor pointcloud top-surface z mapping skipped: non-top grasp demo")
        return False
    grasp = aligned.get("grasp_pose", {})
    grasp_pos = grasp.get("position")
    if not grasp_pos or len(grasp_pos) < 3:
        return False
    if not icp_result:
        return False
    demo_package = icp_result.get("demo_package")
    live_package = icp_result.get("live_package")
    if not demo_package or not live_package:
        return False
    demo_top_z = _robust_base_z_top(demo_package, aligner=aligner)
    live_top_z = _robust_base_z_top(live_package, aligner=aligner)
    if demo_top_z is None or live_top_z is None:
        rospy.logwarn(
            "  anchor pointcloud top-surface z mapping skipped: top z unavailable")
        return False
    demo_grasp_z = None
    try:
        gp = (demo or {}).get("grasp_pose_base_frame", {}).get("position_m", {})
        demo_grasp_z = float(gp["z"])
    except Exception:
        demo_grasp_z = None
    if demo_grasp_z is None:
        rospy.logwarn(
            "  anchor pointcloud top-surface z mapping skipped: demo grasp z unavailable")
        return False
    demo_clearance = max(0.005, demo_grasp_z - demo_top_z)
    old_z = float(grasp_pos[2])
    mapped_z = live_top_z + demo_clearance
    grasp_pos[2] = mapped_z
    if aligned.get("bottleneck_pose"):
        bn_pos = aligned["bottleneck_pose"].get("position")
        if bn_pos and len(bn_pos) >= 3:
            bn_pos[2] += mapped_z - old_z
    aligned["pointcloud_top_z_mapping"] = {
        "demo_top_z": demo_top_z,
        "live_top_z": live_top_z,
        "demo_grasp_z": demo_grasp_z,
        "demo_clearance_above_top": demo_clearance,
        "old_grasp_z": old_z,
        "mapped_grasp_z": mapped_z,
    }
    rospy.loginfo(
        "  anchor pointcloud top-surface z mapping: demo_top=%.4f live_top=%.4f "
        "demo_clearance=%.4f grasp_z %.4f -> %.4f",
        demo_top_z, live_top_z, demo_clearance, old_z, mapped_z)
    return True


def _estimate_object_size_from_pointcloud(points):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 10:
        return None
    lo = np.percentile(points, 10, axis=0)
    hi = np.percentile(points, 90, axis=0)
    size = np.maximum(hi - lo, 0.0)
    if not np.all(np.isfinite(size)) or np.max(size) <= 0:
        return None
    return [float(v) for v in size]


def _anchor_icp_object_pose_base(icp_result, fallback_pose, aligner):
    if not _param_bool("~use_icp_object_pose", False):
        return None
    try:
        if not icp_result:
            return None
        demo_points = np.load(os.path.join(
            icp_result["demo_package"], "pointcloud.npy")).astype(np.float64)
        demo_points = demo_points[np.all(np.isfinite(demo_points), axis=1)]
        if len(demo_points) < 5:
            return None
        demo_center = np.median(demo_points, axis=0)
        transform = np.asarray(icp_result["transform_demo_to_live"],
                               dtype=np.float64)
        live_center_camera = (
            transform[:3, :3] @ demo_center) + transform[:3, 3]
        source_frame = _package_source_frame(icp_result["live_package"])
        base_points = _transform_points_to_base(
            np.asarray([live_center_camera], dtype=np.float64),
            source_frame,
            aligner=aligner,
            min_points=1)
        if base_points is None or len(base_points) < 1:
            return None
        pose = dict(fallback_pose or {})
        pose["position"] = [float(v) for v in base_points[0].tolist()]
        pose["method"] = "anchor_target_ICP+depth_pointcloud"
        old = np.asarray((fallback_pose or {}).get("position", pose["position"]),
                         dtype=np.float64)
        pose["icp_vs_mask_center_m"] = float(
            np.linalg.norm(np.asarray(pose["position"], dtype=np.float64) - old))
        try:
            live_points = np.load(os.path.join(
                icp_result["live_package"], "pointcloud.npy")).astype(np.float64)
            live_size = _estimate_object_size_from_pointcloud(live_points)
            if live_size is not None:
                pose["estimated_object_size"] = live_size
        except Exception:
            pass
        rospy.loginfo(
            "  Anchor ICP object pose base: [%.4f, %.4f, %.4f] "
            "(diff_from_mask_center=%.4fm)",
            pose["position"][0], pose["position"][1], pose["position"][2],
            pose["icp_vs_mask_center_m"])
        return pose
    except Exception as exc:
        rospy.logwarn("  Failed to build anchor ICP object pose: %s", exc)
        return None


def _aligned_grasp_from_demo(demo, target_scene_entry, live_target_xyz,
                             object_size, icp_result=None):
    """Use the same TrajectoryAligner-style grasp mapping as mt3_pipeline."""
    aligner = TrajectoryAligner()
    target_pose = {
        "position": [float(v) for v in live_target_xyz[:3]],
        "orientation": [
            float(v) for v in
            (target_scene_entry or {}).get(
                "orientation_base", [0.0, 0.0, 0.0, 1.0])
        ],
        "estimated_object_size": [float(v) for v in object_size[:3]],
        "method": "anchor_place_base_pose",
    }
    icp_pose = _anchor_icp_object_pose_base(icp_result, target_pose, aligner)
    if icp_pose is not None:
        target_pose = icp_pose
        if target_pose.get("estimated_object_size"):
            object_size = target_pose["estimated_object_size"]
        rospy.loginfo("  Using anchor target ICP object pose for grasp alignment")
    demo_entry = _demo_entry_for_alignment(demo, live_target_xyz)
    demo_grasp, _ = _demo_grasp_pose(demo_entry)
    gp = demo_entry["grasp_pose_base_frame"]["orientation_xyzw"]
    demo_grasp_pose = {
        "position": demo_grasp,
        "orientation": [
            float(gp.get("x", -1.0)),
            float(gp.get("y", 0.0)),
            float(gp.get("z", 0.0)),
            float(gp.get("w", 0.0)),
        ],
    }
    obj_frame = demo_entry["object_pose_base_frame"]["position_m"]
    demo_obj_pos = [float(obj_frame["x"]), float(obj_frame["y"]), float(obj_frame["z"])]
    approach = demo_entry.get("approach_direction", [0.0, 0.0, -1.0])
    retract = demo_entry.get("retract_direction", [0.0, 0.0, 1.0])
    rotate_relative = not (
        abs(float(approach[0])) < 1e-6
        and abs(float(approach[1])) < 1e-6
        and float(approach[2]) < -0.5
    )
    grasp_aligned = aligner.compute_aligned_grasp(
        demo_grasp_pose, demo_obj_pos, target_pose,
        rotate_relative=rotate_relative)
    aligned = {
        "grasp_pose": grasp_aligned,
        "object_pose_base": target_pose,
        "approach_direction": approach,
        "retract_direction": retract,
        "gripper_opening": demo_entry.get("gripper_opening_m", 0.07),
        "tf_source": "anchor_place_base_pose+TrajectoryAligner",
        "retrieved_demo_entry": demo_entry,
    }
    try:
        bn_frame = demo_entry.get("bottleneck_pose_base_frame", {})
        if not bn_frame:
            bn_frame = demo_entry.get("grasp_bottleneck_pose_base_frame", {})
        bn_pos = _demo_position({"bn": bn_frame}, "bn", fallback=None)
        if bn_pos is not None:
            bn_delta = [
                bn_pos[0] - demo_obj_pos[0],
                bn_pos[1] - demo_obj_pos[1],
                bn_pos[2] - demo_obj_pos[2],
            ]
            bn_ori = (
                bn_frame.get("orientation_xyzw", {})
                if isinstance(bn_frame, dict) else {})
            aligned["bottleneck_pose"] = {
                "position": [
                    target_pose["position"][0] + bn_delta[0],
                    target_pose["position"][1] + bn_delta[1],
                    target_pose["position"][2] + bn_delta[2],
                ],
                "orientation": [
                    float(bn_ori.get("x", grasp_aligned["orientation"][0])),
                    float(bn_ori.get("y", grasp_aligned["orientation"][1])),
                    float(bn_ori.get("z", grasp_aligned["orientation"][2])),
                    float(bn_ori.get("w", grasp_aligned["orientation"][3])),
                ],
                "relative_to_object": bn_delta,
            }
    except Exception as exc:
        rospy.logwarn("  Failed to build aligned bottleneck pose: %s", exc)
    _apply_anchor_rectangular_prism_yaw_alignment(aligned, icp_result, aligner)
    _apply_anchor_rectangular_prism_obb_center_alignment(
        aligned, icp_result, aligner)
    if not _apply_anchor_icp_top_surface_grasp_z(
            aligned, demo, icp_result, aligner):
        _apply_anchor_height_aware_top_grasp(aligned)
    pos = aligned["grasp_pose"]["position"]
    rospy.loginfo(
        "  Anchor grasp mapped via TrajectoryAligner: [%.3f, %.3f, %.3f] "
        "rotate_relative=%s",
        pos[0], pos[1], pos[2], rotate_relative)
    return {
        "position": aligned["grasp_pose"]["position"],
        "orientation": aligned["grasp_pose"]["orientation"],
        "relative_to_target_xyz": aligned["grasp_pose"].get(
            "relative_grasp", {}).get("position", []),
        "aligned": aligned,
    }


def _latest_demo_path(task_type):
    candidates = []
    demo_dir = _demo_recorded_dir()
    if os.path.isdir(demo_dir):
        for name in os.listdir(demo_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(demo_dir, name)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if data.get("task_type") == task_type:
                    candidates.append((os.path.getmtime(path), path))
            except Exception:
                continue
    if not candidates:
        raise RuntimeError("no %s demo found in %s" % (task_type, demo_dir))
    candidates.sort(reverse=True)
    return candidates[0][1]


def _demo_path_by_id(demo_id):
    candidate = os.path.join(_demo_recorded_dir(), "%s.json" % demo_id)
    return candidate if os.path.exists(candidate) else ""


def _detected_features(object_size):
    dims = [float(v) for v in object_size[:3]]
    max_dim = max(max(dims), 0.001)
    return {
        "shape": rospy.get_param("~object_shape", "cube"),
        "dimensions_m": dims,
        "aspect_ratio": [v / max_dim for v in dims],
        "color_rgb": [0.0, 1.0, 0.0],
        "object_label": rospy.get_param("~target_label", "green_cube"),
    }


def _find_demo_path(detected_features=None):
    explicit = rospy.get_param("~demo_path", "")
    if explicit:
        return explicit, {"retrieval_mode": "explicit_demo_path"}
    demo_id = rospy.get_param("~demo_id", "")
    if demo_id:
        candidate = _demo_path_by_id(demo_id)
        if candidate:
            return candidate, {
                "retrieval_mode": "explicit_demo_id",
                "selected_demo_id": demo_id,
                "selected_demo_path": candidate,
            }

    query = rospy.get_param(
        "~query", "pick the green cube and place it on the blue platform")
    library = DemoLibrary(execution_environment=_execution_environment())
    demo, score, metadata = library.full_query(
        query,
        detected_features or _detected_features(rospy.get_param(
            "~object_size", [0.045, 0.045, 0.045])),
        task_type="anchor_pick_place",
        retrieval_mode=rospy.get_param("~retrieval_mode", "hierarchical"),
        return_metadata=True)
    path = metadata.get("selected_demo_path") or _demo_path_by_id(demo.get("id", ""))
    if not path:
        raise RuntimeError("retrieved anchor demo has no recorded JSON: %s" % demo.get("id", ""))
    metadata["selected_demo_path"] = path
    metadata["selected_score"] = float(score)
    metadata["query"] = query
    return path, metadata


def _append_csv_row(csv_path, row):
    """
    Safely append one experiment row to CSV.

    JSONL is the authoritative experiment record. CSV is a convenience table:
    never overwrite an unreadable CSV, append directly when the schema is
    unchanged, and rewrite through a temporary file only when the schema grows.
    """
    existing_rows = []
    existing_fields = []

    csv_exists = (
        os.path.exists(csv_path)
        and os.path.getsize(csv_path) > 0
    )

    if csv_exists:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                for old in reader:
                    existing_rows.append({
                        k: v for k, v in old.items() if k is not None
                    })
        except Exception as exc:
            backup_path = (
                csv_path
                + ".read_failure_backup_"
                + time.strftime("%Y%m%d_%H%M%S")
            )
            try:
                shutil.copy2(csv_path, backup_path)
                rospy.logwarn(
                    "  Existing CSV could not be read as UTF-8. "
                    "Backup created: %s",
                    backup_path)
            except Exception as backup_exc:
                rospy.logwarn(
                    "  Existing CSV is unreadable and backup creation "
                    "also failed: %s",
                    backup_exc)
            raise RuntimeError(
                "Existing CSV is unreadable; refusing to overwrite it: %s"
                % exc) from exc

    fieldnames = []
    for name in existing_fields + list(row.keys()):
        if name and name not in fieldnames:
            fieldnames.append(name)

    if not csv_exists:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        return

    schema_changed = fieldnames != existing_fields
    if not schema_changed:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow({k: row.get(k, "") for k in existing_fields})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        return

    temp_path = csv_path + ".tmp_" + str(os.getpid())
    try:
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old in existing_rows:
                writer.writerow({k: old.get(k, "") for k in fieldnames})
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(temp_path, csv_path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise


def _log_experiment_trial(trial_id, outcome, demo, retrieval_meta, scene,
                          grasp, place_result, rollout_path, replay_path,
                          object_size=None, scene_packages=None,
                          timing=None, failure_stage="", failure_reason="",
                          initial_target_gt=None, initial_anchor_gt=None,
                          postcheck=None, execution_success=None,
                          icp_result=None):
    if not _param_bool("~auto_log_experiment", True):
        return
    os.makedirs(EXPERIMENT_LOG_DIR, exist_ok=True)
    csv_path = os.path.join(EXPERIMENT_LOG_DIR, "mt3_relation_trials.csv")
    target = scene.get("target", {})
    anchor = scene.get("anchor", {})
    target_est = target.get("position_base")
    anchor_est = anchor.get("position_base")
    target_gt = initial_target_gt or _gazebo_pose(
        "~target_gt_model", ["grasp_object", "green", "cube", "object"])
    anchor_gt = initial_anchor_gt or _gazebo_pose(
        "~anchor_gt_model",
        ["blue_placement_platform", "blue", "platform", "anchor"])
    target_gt_xyz = (target_gt or {}).get("xyz")
    anchor_gt_xyz = (anchor_gt or {}).get("xyz")
    replay_type, release_index = _replay_info(replay_path)
    replay_used = _param_bool("~use_demo_replay", True) and bool(replay_path)
    relation_alignment_mode = rospy.get_param(
        "~relation_alignment_mode", "target_anchor")
    timing = timing or {}
    postcheck = postcheck or {}
    object_size_logged = object_size or rospy.get_param(
        "~object_size", [0.045, 0.045, 0.045])
    anchor_size = rospy.get_param("~anchor_size", [0.10, 0.10, 0.02])
    execution_ok = (
        bool(outcome == "success") if execution_success is None
        else bool(execution_success))
    relation_path = ((scene_packages or {}).get("relation") or {}).get("relation_path", "")
    target_error_xy = _xy_error(target_est, target_gt_xyz)
    anchor_error_xy = _xy_error(anchor_est, anchor_gt_xyz)
    aligned = (grasp or {}).get("aligned") or {}
    yaw_info = aligned.get("pointcloud_yaw_alignment", {}) or {}
    obb_info = aligned.get("pointcloud_obb_center_alignment", {}) or {}
    top_z_info = aligned.get("pointcloud_top_z_mapping", {}) or {}
    icp_metrics = (icp_result or {}).get("metrics", {}) if icp_result else {}
    initial_object_yaw_deg = _quat_yaw_deg(
        (target_gt or {}).get("orientation_xyzw"))
    final_object_yaw_deg = postcheck.get("final_object_yaw_deg", "")
    target_place_yaw_deg = rospy.get_param(
        "~place_target_yaw_deg", initial_object_yaw_deg)
    place_yaw_symmetry_deg = float(rospy.get_param(
        "~place_yaw_symmetry_deg", _default_place_yaw_symmetry_deg()))
    place_yaw_error_raw_deg = _yaw_error_deg(
        final_object_yaw_deg, target_place_yaw_deg, symmetry_deg=None)
    place_yaw_error_deg = _yaw_error_deg(
        final_object_yaw_deg, target_place_yaw_deg,
        symmetry_deg=place_yaw_symmetry_deg)
    place_center_error_xy_m = _simple_xy_distance(
        postcheck.get("final_object_xyz"),
        postcheck.get("final_anchor_xyz"))
    stable_success = postcheck.get(
        "postcheck_success",
        bool(outcome == "success") if outcome != "dry_run" else "")
    precise_success = _precise_place_success(
        place_center_error_xy_m, object_size_logged, anchor_size)

    # ── 读回执行器记录的抓取诊断参数 ──
    before_close_mouth_center_xy = _ros_vec("/sawyer_auto_grasp/before_close_mouth_center_xy", 2)
    before_close_mouth_error_xy_m = _ros_float(
        "/sawyer_auto_grasp/before_close_mouth_error_xy_m")
    before_close_mouth_x = _ros_float("/sawyer_auto_grasp/before_close_mouth_x")
    before_close_mouth_y = _ros_float("/sawyer_auto_grasp/before_close_mouth_y")
    before_close_mouth_error_x_m = _ros_float(
        "/sawyer_auto_grasp/before_close_mouth_error_x_m")
    before_close_mouth_error_y_m = _ros_float(
        "/sawyer_auto_grasp/before_close_mouth_error_y_m")
    execution_variant = str(rospy.get_param(
        "/sawyer_auto_grasp/execution_variant", ""))
    used_recovery = bool(rospy.get_param(
        "/sawyer_auto_grasp/used_recovery_logic", False))

    failure_category = (
        _failure_category(failure_stage, failure_reason)
        if outcome == "failed" else "")
    place_failure_detail = (
        _place_failure_detail(
            failure_category, failure_reason, target_error_xy, anchor_error_xy)
        if outcome == "failed" else "")
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_id": trial_id,
        "task_type": "anchor_pick_place",
        "query": rospy.get_param(
            "~query", "pick the green cube and place it on the blue platform"),
        "condition_id": rospy.get_param("~condition_id", ""),
        "repeat_id": rospy.get_param("~repeat_id", ""),
        "method_variant": rospy.get_param("~method_variant", "full"),
        "relation_alignment_mode": relation_alignment_mode,
        "trajectory_transfer_mode": (
            "stage_replay" if replay_used else "scripted_execution"),
        "outcome": outcome,
        "success": bool(outcome == "success"),
        "task_success": bool(outcome == "success"),
        "postcheck_success": postcheck.get(
            "postcheck_success",
            bool(outcome == "success") if outcome != "dry_run" else ""),
        "failure_stage": failure_stage if outcome == "failed" else "",
        "failure_reason": failure_reason if outcome == "failed" else "",
        "failure_category": failure_category,
        "place_failure_detail": place_failure_detail,
        "retrieval_mode": retrieval_meta.get("retrieval_mode", ""),
        "retrieved_demo_id": demo.get("id", ""),
        "language_score": retrieval_meta.get("language_score", ""),
        "geometry_score": retrieval_meta.get("geometry_score", retrieval_meta.get("selected_score", "")),
        "target_shape": rospy.get_param("~object_shape", "cube"),
        "target_size_xyz": _json_vec(object_size_logged),
        "target_position": _json_vec(target_gt_xyz),
        "target_position_xy": _json_vec((target_gt_xyz or ["", ""])[:2]),
        "target_gt_model": (target_gt or {}).get("name", ""),
        "target_gt_xyz": _xy_only_xyz(target_gt_xyz),
        "target_gt_world_xyz": _json_vec(target_gt_xyz),
        "initial_object_xyz": _json_vec(target_gt_xyz),
        "target_gt_frame": "gazebo_world_xy_only",
        "target_est_frame": "base",
        "target_error_z_m": "",
        "target_error_xyz_m": "",
        "target_est_xyz": _json_vec(target_est),
        "target_error_xy_m": target_error_xy,
        "target_x": (target_est or ["", "", ""])[0],
        "target_y": (target_est or ["", "", ""])[1],
        "target_z": (target_est or ["", "", ""])[2],
        "anchor_gt_model": (anchor_gt or {}).get("name", ""),
        "anchor_gt_xyz": _xy_only_xyz(anchor_gt_xyz),
        "initial_anchor_xyz": _json_vec(anchor_gt_xyz),
        "anchor_est_xyz": _json_vec(anchor_est),
        "anchor_error_xy_m": anchor_error_xy,
        "anchor_x": (anchor_est or ["", "", ""])[0],
        "anchor_y": (anchor_est or ["", "", ""])[1],
        "anchor_z": (anchor_est or ["", "", ""])[2],
        "anchor_gt_world_xyz": _json_vec(anchor_gt_xyz),
        "anchor_gt_frame": "gazebo_world_xy_only",
        "anchor_est_frame": "base",
        "anchor_error_z_m": "",
        "anchor_error_xyz_m": "",
        "platform_size": _list_min_float((anchor_size or [])[:2]),
        "platform_size_m": _list_min_float((anchor_size or [])[:2]),
        "platform_size_cm": (
            _list_min_float((anchor_size or [])[:2]) * 100.0
            if _list_min_float((anchor_size or [])[:2]) != "" else ""),
        "platform_size_xyz": _json_vec(anchor_size),
        "target_yaw_gt_deg": "",
        "target_yaw_est_deg": yaw_info.get("live_yaw_deg", ""),
        "demo_yaw_deg": yaw_info.get("demo_yaw_deg", ""),
        "live_yaw_deg": yaw_info.get("live_yaw_deg", ""),
        "delta_yaw_deg": yaw_info.get("delta_yaw_deg", ""),
        "yaw_error_deg": "",
        "obb_center_dx_m": obb_info.get("dx", ""),
        "obb_center_dy_m": obb_info.get("dy", ""),
        "obb_extent_long_m": obb_info.get("extent_long", ""),
        "obb_extent_short_m": obb_info.get("extent_short", ""),
        "demo_top_z": top_z_info.get("demo_top_z", ""),
        "live_top_z": top_z_info.get("live_top_z", ""),
        "demo_clearance_above_top": top_z_info.get(
            "demo_clearance_above_top", ""),
        "mapped_grasp_z": top_z_info.get("mapped_grasp_z", ""),
        "socket_gt_xyz": "",
        "socket_est_xyz": "",
        "socket_error_xy_m": "",
        "socket_opening": "",
        "grasp_x": grasp["position"][0],
        "grasp_y": grasp["position"][1],
        "grasp_z": grasp["position"][2],
        "grasp_xyz": _json_vec(grasp.get("position")),
        "bottleneck_xyz": _json_vec((demo.get("place_bottleneck_pose_base_frame") or {}).get("position", [])),
        "place_x": place_result["place_xyz"][0],
        "place_y": place_result["place_xyz"][1],
        "place_z": place_result["place_xyz"][2],
        "place_or_insert_xyz": _json_vec(place_result.get("place_xyz")),
        "place_offset_xyz": json.dumps(place_result.get("offset_xyz", [])),
        "final_object_model_name": postcheck.get("final_object_model_name", ""),
        "final_anchor_model_name": postcheck.get("final_anchor_model_name", ""),
        "final_object_xyz": _json_vec(postcheck.get("final_object_xyz")),
        "final_anchor_xyz": _json_vec(postcheck.get("final_anchor_xyz")),
        "final_target_error_xy_m": postcheck.get("final_target_error_xy_m", ""),
        "final_relation_error_xy_m": postcheck.get("final_relation_error_xy_m", ""),
        "place_center_error_xy_m": place_center_error_xy_m,
        "center_error_xy_mm": (
            float(place_center_error_xy_m) * 1000.0
            if place_center_error_xy_m != "" else ""),
        "initial_object_yaw_deg": initial_object_yaw_deg,
        "target_place_yaw_deg": target_place_yaw_deg,
        "final_object_yaw_deg": final_object_yaw_deg,
        "place_yaw_error_deg": place_yaw_error_deg,
        "place_yaw_error_raw_deg": place_yaw_error_raw_deg,
        "place_yaw_symmetry_deg": place_yaw_symmetry_deg,
        "stable_success": stable_success,
        "precise_success": precise_success,
        "insert_depth_m": postcheck.get("insert_depth_m", ""),
        "postcheck_reason": postcheck.get("postcheck_reason", ""),
        "replay_used": replay_used,
        "replay_type": replay_type,
        "release_index": release_index,
        "mask_pixels": _mask_pixels(target.get("mask_path")),
        "anchor_mask_pixels": _mask_pixels(anchor.get("mask_path")),
        "pointcloud_points": _point_count(target),
        "anchor_pointcloud_points": _point_count(anchor),
        "icp_mean_error_m": icp_metrics.get("mean_error_m", ""),
        "icp_median_error_m": icp_metrics.get("median_error_m", ""),
        "icp_p90_error_m": icp_metrics.get("p90_error_m", ""),
        "icp_iterations": icp_metrics.get("iterations", ""),
        "planning_success": "",
        "execution_success": execution_ok,
        "execution_variant": execution_variant,
        "used_recovery_logic": used_recovery,
        "before_close_mouth_center_xy": json.dumps(before_close_mouth_center_xy),
        "before_close_mouth_error_xy_m": before_close_mouth_error_xy_m,
        "before_close_mouth_x": before_close_mouth_x,
        "before_close_mouth_y": before_close_mouth_y,
        "before_close_mouth_error_x_m": before_close_mouth_error_x_m,
        "before_close_mouth_error_y_m": before_close_mouth_error_y_m,
        "manual_success_label": "",
        "total_time_s": timing.get("total_time_s", ""),
        "perception_time_s": timing.get("perception_time_s", ""),
        "retrieval_time_s": timing.get("retrieval_time_s", ""),
        "alignment_time_s": timing.get("alignment_time_s", ""),
        "planning_time_s": timing.get("planning_time_s", ""),
        "robot_execution_time_s": timing.get("robot_execution_time_s", ""),
        "execution_wall_time_s": timing.get("execution_time_s", ""),
        "planning_call_count": timing.get("planning_call_count", ""),
        "robot_execution_call_count": timing.get("robot_execution_call_count", ""),
        "timing_source": timing.get("timing_source", ""),
        "execution_time_s": timing.get("execution_time_s", ""),
        "rollout_path": rollout_path,
        "replay_path": replay_path,
        "scene_package_path": relation_path,
    }
    # ============================================================
    # SAFE EXPERIMENT LOGGING
    #
    # JSONL is authoritative and MUST be written first. CSV is a
    # convenience table and is allowed to fail without invalidating an
    # already completed robot trial.
    # ============================================================
    jsonl_path = os.path.join(EXPERIMENT_LOG_DIR, "mt3_relation_trials.jsonl")
    jsonl_ok = False
    csv_ok = False

    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        jsonl_ok = True
        rospy.loginfo("  Relation experiment JSONL logged: %s", jsonl_path)
    except Exception as exc:
        rospy.logerr(
            "  CRITICAL: failed to write authoritative relation "
            "experiment JSONL: %s",
            exc)

    try:
        _append_csv_row(csv_path, row)
        csv_ok = True
        rospy.loginfo("  Relation experiment CSV logged: %s", csv_path)
    except Exception as exc:
        rospy.logwarn("  Relation experiment CSV logging failed: %s", exc)
        if jsonl_ok:
            rospy.logwarn(
                "  Trial %s is SAFE in JSONL. CSV failure does NOT "
                "invalidate the robot trial.",
                trial_id)
        else:
            rospy.logerr(
                "  WARNING: both JSONL and CSV logging failed for trial_id=%s",
                trial_id)

    if jsonl_ok and csv_ok:
        rospy.loginfo("  Relation experiment trial fully logged: %s", trial_id)
    elif jsonl_ok:
        rospy.logwarn(
            "  Relation experiment trial preserved in JSONL only: %s",
            trial_id)
    elif csv_ok:
        rospy.logwarn(
            "  Relation experiment trial exists in CSV only (JSONL FAILED): %s",
            trial_id)
    else:
        rospy.logerr(
            "  Relation experiment trial could not be persisted: %s",
            trial_id)


def _place_xyz_from_demo(demo):
    place_info = (demo or {}).get("place_info", {}) or {}
    xyz = (
        place_info.get("place_xyz")
        or (place_info.get("place_pose_base_frame") or {}).get("position")
        or (place_info.get("alignment_pose_base_frame") or {}).get("position")
    )
    try:
        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except Exception:
        return None
    return None


def _orientation_from_pose_payload(pose, fallback=None):
    pose = pose or {}
    ori = pose.get("orientation")
    if isinstance(ori, (list, tuple)) and len(ori) >= 4:
        return [float(v) for v in ori[:4]]
    ori = pose.get("orientation_xyzw", {})
    if isinstance(ori, dict) and ori:
        return [
            float(ori.get("x", -1.0)),
            float(ori.get("y", 0.0)),
            float(ori.get("z", 0.0)),
            float(ori.get("w", 0.0)),
        ]
    return list(fallback or [-1.0, 0.0, 0.0, 0.0])


def _aligned_place_bottleneck_pose(demo, place_result):
    demo_place = _place_xyz_from_demo(demo)
    demo_bn = _demo_position(demo, "place_bottleneck_pose_base_frame")
    live_place = (place_result or {}).get("place_xyz")
    if demo_place is None or demo_bn is None or not live_place:
        return {}
    mapped = [
        float(live_place[0]) + (float(demo_bn[0]) - float(demo_place[0])),
        float(live_place[1]) + (float(demo_bn[1]) - float(demo_place[1])),
        float(live_place[2]) + (float(demo_bn[2]) - float(demo_place[2])),
    ]
    return {
        "position": mapped,
        "orientation": _orientation_from_pose_payload(
            (demo or {}).get("place_bottleneck_pose_base_frame", {})),
        "relative_to_demo_place": [
            float(demo_bn[0]) - float(demo_place[0]),
            float(demo_bn[1]) - float(demo_place[1]),
            float(demo_bn[2]) - float(demo_place[2]),
        ],
        "source": "demo_place_bottleneck_relative_to_place_xyz",
    }


def _write_replay_input(demo, trial_id, place_result=None):
    trajectory = demo.get("place_trajectory") or demo.get("trajectory")
    if not trajectory:
        return ""
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    path = os.path.join(ROLLOUT_DIR, "anchor_replay_input_%s.json" % trial_id)
    payload = {
        "format": "mt3_anchor_demo_replay_input_v1",
        "source_demo": demo.get("id", ""),
        "trajectory": trajectory,
        "trajectory_source": (
            "place_trajectory" if demo.get("place_trajectory")
            else "full_trajectory"),
        "place_bottleneck_pose_base_frame": demo.get(
            "place_bottleneck_pose_base_frame"),
        "aligned_place_bottleneck_pose": _aligned_place_bottleneck_pose(
            demo, place_result or {}),
        "aligned_place_pose": {
            "position": [
                float(v) for v in (place_result or {}).get("place_xyz", [])
            ],
            "orientation": _orientation_from_pose_payload(
                demo.get("place_release_pose_base_frame", {})),
        } if (place_result or {}).get("place_xyz") else {},
        "place_info": demo.get("place_info", {}),
        "anchor_info": demo.get("anchor_info", {}),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _pose_payload_from_aligned(pose):
    pose = pose or {}
    if not pose.get("position"):
        return {}
    return {
        "position": [float(v) for v in (pose.get("position") or [0.0, 0.0, 0.0])[:3]],
        "orientation": [
            float(v)
            for v in (pose.get("orientation") or [-1.0, 0.0, 0.0, 0.0])[:4]
        ],
    }


def _load_demo_rollout_trajectory(demo):
    path = (
        (demo or {}).get("rollout_trajectory_path")
        or (demo or {}).get("trajectory_path")
        or "")
    if path:
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            path = os.path.join(CODE_DIR, path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("poses"):
                rospy.loginfo(
                    "Grasp replay: loaded full rollout trajectory: %s "
                    "(poses=%d)",
                    path, len(data.get("poses", [])))
                return data
            rospy.logwarn(
                "Grasp replay: rollout trajectory has no poses: %s", path)
        except Exception as exc:
            rospy.logwarn(
                "Grasp replay: failed to load rollout trajectory %s: %s",
                path, exc)
    embedded = (demo or {}).get("trajectory")
    if isinstance(embedded, dict) and embedded.get("poses"):
        rospy.logwarn(
            "Grasp replay: using embedded demo trajectory fallback "
            "(poses=%d); rollout_trajectory_path unavailable",
            len(embedded.get("poses", [])))
        return embedded
    return None


def _write_grasp_replay_input(demo, trial_id, grasp=None):
    """Write a grasp replay input from this anchor-place demo.

    When a recorded demo bottleneck is available, follow the top/rotated grasp
    convention: replay the full rollout segment starting at that bottleneck,
    then transfer the mapped bottleneck pose at execution time.
    """
    demo_bottleneck_xyz = (
        _demo_position(demo, "bottleneck_pose_base_frame")
        or _demo_position(demo, "grasp_bottleneck_pose_base_frame"))
    saved_grasp_trajectory = demo.get("grasp_trajectory")
    use_saved_grasp_segment = bool(
        isinstance(saved_grasp_trajectory, dict) and
        len(saved_grasp_trajectory.get("poses", [])) >= 10)
    full_trajectory = None if use_saved_grasp_segment else _load_demo_rollout_trajectory(demo)
    trajectory = saved_grasp_trajectory if use_saved_grasp_segment else full_trajectory
    if not trajectory or not isinstance(trajectory, dict):
        return ""
    poses = trajectory.get("poses", [])
    if len(poses) < 10:
        return ""

    # Find gripper close: first frame where gripper_state flips 0→1
    close_idx = None
    if use_saved_grasp_segment:
        try:
            close_idx = int(trajectory.get("close_index"))
        except Exception:
            close_idx = None
        if close_idx is not None and not (0 <= close_idx < len(poses)):
            rospy.logwarn(
                "Grasp replay: saved close_index=%d out of range; "
                "using midpoint", close_idx)
            close_idx = None
    if close_idx is None:
        prev_state = None
        for i, sample in enumerate(poses):
            gs = sample.get("gripper_state")
            if gs is None:
                continue
            try:
                gs = int(gs)
            except Exception:
                continue
            if prev_state == 0 and gs == 1:
                close_idx = i
                break
            prev_state = gs
    if close_idx is None:
        # Fallback: look for gripper_next == 1 in velocities
        velocities = trajectory.get("velocities", [])
        for i, v in enumerate(velocities):
            if v.get("gripper_next") == 1:
                close_idx = i + 1  # velocity[i] applies after pose[i]
                break
    if close_idx is None or (close_idx < 10 and not use_saved_grasp_segment):
        rospy.logwarn("Grasp replay: cannot find gripper close event; "
                      "writing full trajectory")
        close_idx = len(poses) // 2

    pre = 60   # frames before close: approach + descent (~2s)
    post = 30  # frames after close: lift only, stop before place phase
    demo_bottleneck_xyz = (
        _demo_position(demo, "bottleneck_pose_base_frame")
        or _demo_position(demo, "grasp_bottleneck_pose_base_frame"))
    bottleneck_xyz = demo_bottleneck_xyz
    start_idx = (
        0 if use_saved_grasp_segment else
        max(0, close_idx - pre))
    end_idx = len(poses) if use_saved_grasp_segment else min(
        len(poses), close_idx + post + 1)
    min_pre_samples = int(rospy.get_param(
        "~grasp_replay_min_pre_samples", 8))
    if (not saved_grasp_trajectory and
            int(close_idx) - int(start_idx) < min_pre_samples):
        rospy.logwarn(
            "Grasp replay: bottleneck-selected start leaves only %d "
            "pre-close samples; using fixed close window instead.",
            int(close_idx) - int(start_idx))
        start_idx = max(0, close_idx - pre)
    if use_saved_grasp_segment:
        bottleneck_xyz = (
            demo_bottleneck_xyz
            or trajectory.get("base_position")
            or _trajectory_pose_xyz(poses[0]))
        rospy.loginfo(
            "Grasp replay: using fixed saved grasp_trajectory segment "
            "close_index=%d window=[%d:%d] poses=%d",
            close_idx, start_idx, end_idx, len(poses))
    else:
        bottleneck_xyz = (
            demo_bottleneck_xyz
            or trajectory.get("base_position")
            or _trajectory_pose_xyz(poses[start_idx]))
        rospy.logwarn(
            "Grasp replay: using legacy fixed full-rollout window "
            "close_index=%d window=[%d:%d] poses=%d",
            close_idx, start_idx, end_idx, len(poses))
    segment_poses = [dict(sample) for sample in poses[start_idx:end_idx]]
    velocities = trajectory.get("velocities", [])
    segment_velocities = (
        [dict(sample) for sample in velocities[start_idx:end_idx]]
        if velocities else [])
    local_close_idx = int(close_idx - start_idx)
    source_close_idx = int(
        trajectory.get("source_close_index", close_idx)
        if use_saved_grasp_segment
        else close_idx)
    if 0 <= local_close_idx < len(segment_poses):
        for i, sample in enumerate(segment_poses):
            sample["gripper_state"] = 1 if i >= local_close_idx else 0
            if i == local_close_idx:
                sample["gripper_next"] = 1
            elif "gripper_next" in sample:
                sample["gripper_next"] = 0
        # velocity[i] is applied after moving from pose[i] to pose[i + 1].
        velocity_close_idx = local_close_idx - 1
        if 0 <= velocity_close_idx < len(segment_velocities):
            for i, sample in enumerate(segment_velocities):
                if "gripper_next" in sample or i == velocity_close_idx:
                    sample["gripper_next"] = 1 if i == velocity_close_idx else 0

    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    path = os.path.join(ROLLOUT_DIR, "grasp_replay_input_%s.json" % trial_id)
    grasp_pose = demo.get("grasp_pose_base_frame") or {}
    grasp_traj = {
        "poses": segment_poses,
        "velocities": segment_velocities,
        "close_index": local_close_idx,
        "source_close_index": int(source_close_idx),
        "base_index": 0,
        "source_base_index": int(start_idx),
    }
    if bottleneck_xyz is not None:
        grasp_traj["base_position"] = [float(v) for v in bottleneck_xyz]
    aligned = (grasp or {}).get("aligned") or {}
    yaw_info = aligned.get("pointcloud_yaw_alignment", {}) or {}
    aligned_grasp_pose = _pose_payload_from_aligned(
        aligned.get("grasp_pose") or {
            "position": [
                float(grasp_pose.get("position_m", {}).get("x", 0)),
                float(grasp_pose.get("position_m", {}).get("y", 0)),
                float(grasp_pose.get("position_m", {}).get("z", 0)),
            ],
            "orientation": [
                float(grasp_pose.get("orientation_xyzw", {}).get("x", -1)),
                float(grasp_pose.get("orientation_xyzw", {}).get("y", 0)),
                float(grasp_pose.get("orientation_xyzw", {}).get("z", 0)),
                float(grasp_pose.get("orientation_xyzw", {}).get("w", 0)),
            ],
        })
    aligned_bottleneck_pose = _pose_payload_from_aligned(
        aligned.get("bottleneck_pose") or {})
    payload = {
        "format": "mt3_demo_replay_input_v1",
        "source_demo": demo.get("id", ""),
        "trajectory": grasp_traj,
        "close_index": local_close_idx,
        "source_close_index": int(source_close_idx),
        "replay_base_position": (
            [float(v) for v in bottleneck_xyz]
            if bottleneck_xyz is not None else None),
        "replay_yaw_delta_rad": float(yaw_info.get("delta_yaw_rad", 0.0)),
        "replay_yaw_delta_deg": float(yaw_info.get("delta_yaw_deg", 0.0)),
        "use_aligned_bottleneck_pose": bool(aligned_bottleneck_pose),
        "aligned_grasp_pose": aligned_grasp_pose,
        "aligned_bottleneck_pose": aligned_bottleneck_pose,
        "pointcloud_yaw_alignment": yaw_info,
        "pointcloud_obb_center_alignment": aligned.get(
            "pointcloud_obb_center_alignment", {}),
        "pointcloud_top_z_mapping": aligned.get("pointcloud_top_z_mapping", {}),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    rospy.loginfo("  grasp replay input: %s (frames %d:%d, close=%d, "
                  "%d poses %d velocities)",
                  path, start_idx, end_idx, close_idx,
                  len(segment_poses), len(segment_velocities))
    return path


def _write_execution_params(grasp, place_result, object_size, rollout_path,
                            replay_path, grasp_replay_path=""):
    pos = grasp["position"]
    q = grasp["orientation"]
    rospy.set_param("/sawyer_auto_grasp/grasp_x", float(pos[0]))
    rospy.set_param("/sawyer_auto_grasp/grasp_y", float(pos[1]))
    rospy.set_param("/sawyer_auto_grasp/grasp_z", float(pos[2]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qx", float(q[0]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qy", float(q[1]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qz", float(q[2]))
    rospy.set_param("/sawyer_auto_grasp/grasp_qw", float(q[3]))
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

    use_replay = _param_bool("~use_demo_replay", True) and bool(replay_path)
    use_grasp_replay = (
        use_replay and
        _param_bool("~use_grasp_replay", False) and
        bool(grasp_replay_path))
    rospy.set_param("/sawyer_auto_grasp/use_demo_replay", use_replay)
    rospy.set_param("/sawyer_auto_grasp/use_place_release_replay", use_replay)
    rospy.set_param("/sawyer_auto_grasp/use_grasp_replay", use_grasp_replay)
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_prefer_pose_replay",
                    _param_bool("~grasp_replay_prefer_pose_replay", True))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_use_segmented_replay",
                    _param_bool("~grasp_replay_use_segmented_replay", True))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_close_on_blocked",
                    _param_bool("~grasp_replay_close_on_blocked", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_use_top_mouth_center_final_correction",
        _param_bool(
            "~grasp_replay_use_top_mouth_center_final_correction", False))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_close_on_blocked_min_progress",
        float(rospy.get_param(
            "~grasp_replay_close_on_blocked_min_progress", 0.35)))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_anchor_close_waypoint",
                    _param_bool("~grasp_replay_anchor_close_waypoint", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_use_aligned_bottleneck_pose",
        _param_bool("~grasp_replay_use_aligned_bottleneck_pose", True))
    rospy.set_param("/sawyer_auto_grasp/place_replay_lock_xy",
                    _param_bool("~place_replay_lock_xy", True))
    rospy.set_param("/sawyer_auto_grasp/place_replay_use_recorded_orientation",
                    _param_bool("~place_replay_use_recorded_orientation", True))
    rospy.set_param("/sawyer_auto_grasp/demo_replay_trajectory_path", replay_path)
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_trajectory_path",
                    grasp_replay_path if use_grasp_replay else "")
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_path", rollout_path)
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_rate_hz", float(
        rospy.get_param("~trajectory_record_rate_hz", 10.0)))


def _run_executor():
    script = _executor_path()
    if not os.path.exists(script):
        raise RuntimeError("executor script not found: %s" % script)
    detected_failure = None
    error_patterns = [
        # grasp failures
        ("gripper did not close", "grasp_execution", "gripper_no_closure"),
        ("grasp failed", "grasp_execution", "grasp_execution_failed"),
        ("grasp replay failed", "grasp_replay", "grasp_replay_failed"),
        ("descent still too high", "grasp_execution", "descent_height_exceeded"),
        ("mouth-center correction skipped", "grasp_execution", "mouth_tf_unavailable"),
        # motion planning
        ("no motion plan", "motion_planning", "moveit_no_motion_plan"),
        ("planning failed", "motion_planning", "moveit_planning_failed"),
        ("plan failed", "motion_planning", "moveit_planning_failed"),
        # trajectory execution
        ("controller", "trajectory_execution", "controller_execution_failed"),
        ("control_failed", "trajectory_execution", "controller_control_failed"),
        ("path_tolerance", "trajectory_execution", "path_tolerance_violated"),
        ("aborted", "trajectory_execution", "trajectory_execution_aborted"),
        # place failures
        ("place replay execute failed", "place_replay", "place_replay_execute_failed"),
        ("failed/skipped", "place_replay", "place_replay_failed_or_skipped"),
        ("pick-place execution failed", "placement_execution", "place_execution_failed"),
        # post-check failure
        ("目标物体最终放置位置", "postcheck", "postcheck_relation_error"),
    ]
    proc = subprocess.Popen(
        ["python", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            rospy.loginfo("[anchor-pipeline executor] %s", line)
            low = line.lower()
            for pattern, stage, reason in error_patterns:
                if pattern in low and detected_failure is None:
                    detected_failure = (stage, reason)
    proc.wait()
    if proc.returncode != 0:
        return False, (
            detected_failure or
            ("execution_process", "executor_exit_code_%d" % proc.returncode))
    return True, ("", "")


def main():
    rospy.init_node("mt3_anchor_place_pipeline", anonymous=True)
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    run_start = time.time()

    warm_demo = None
    try:
        with open(_latest_demo_path("anchor_pick_place"), "r") as f:
            warm_demo = json.load(f)
    except Exception:
        warm_demo = None

    t_perception = time.time()
    scene, target_xyz, anchor_xyz, object_size = _load_scene(warm_demo)
    perception_time_s = time.time() - t_perception
    t_retrieval = time.time()
    demo_path, retrieval_meta = _find_demo_path(_detected_features(object_size))
    retrieval_time_s = time.time() - t_retrieval
    t_alignment = time.time()
    trial_id = rospy.get_param(
        "~trial_id", "anchor_place_%s" % time.strftime("%Y%m%d_%H%M%S"))
    with open(demo_path, "r") as f:
        demo = json.load(f)
    anchor_profile = {
        "name": rospy.get_param("~anchor_name", "blue_placement_platform"),
        "category": rospy.get_param("~anchor_category", "small_platform"),
        "surface_z_offset": float(rospy.get_param("~anchor_surface_z_offset", 0.0)),
    }
    relation_alignment_mode = rospy.get_param(
        "~relation_alignment_mode", "target_anchor")
    if relation_alignment_mode in (
            "target_displacement", "target_only", "no_relation"):
        place_result = compute_target_displacement_place_target(
            target_xyz,
            object_size=object_size,
            demo_entry=demo,
            default_offset_xyz=anchor_profile.get(
                "default_place_offset_xyz", [0.0, 0.0, 0.0]))
    else:
        place_result = compute_anchor_place_target(
            anchor_xyz,
            object_position_base=target_xyz,
            object_size=object_size,
            demo_entry=demo,
            anchor_profile=anchor_profile,
            override_offset_xyz=rospy.get_param(
                "~anchor_place_offset_xyz", None))
    scene_packages = save_dual_object_scene_packages(
        scene,
        trial_id,
        role="live_anchor_place_trial",
        target_label=rospy.get_param("~target_label", "green_cube"),
        anchor_label=anchor_profile["name"],
        relation_kind="anchor_pick_place",
        extra_metadata={
            "trial_id": trial_id,
            "source_demo": demo.get("id", ""),
            "retrieval": retrieval_meta,
            "task_type": "anchor_pick_place",
            "relation_alignment_mode": relation_alignment_mode,
            "object_size": [float(v) for v in object_size],
            "anchor_profile": anchor_profile,
            "place_target_xyz": [float(v) for v in place_result["place_xyz"]],
            "place_offset_xyz": [float(v) for v in place_result["offset_xyz"]],
        })
    live_target_package = (scene_packages or {}).get("target_package")
    demo_target_package = _target_package_from_recorded_demo(demo)
    icp_result = _run_anchor_icp_registration(
        demo_target_package, live_target_package, trial_id)
    grasp = _aligned_grasp_from_demo(
        demo, scene.get("target", {}), target_xyz, object_size,
        icp_result=icp_result)
    aligned_obj_size = (
        ((grasp.get("aligned") or {}).get("object_pose_base") or {})
        .get("estimated_object_size"))
    if aligned_obj_size and len(aligned_obj_size) >= 3:
        object_size = [float(v) for v in aligned_obj_size[:3]]
    alignment_time_s = time.time() - t_alignment

    rollout_path = os.path.join(ROLLOUT_DIR, "rollout_%s.json" % trial_id)
    replay_path = _write_replay_input(demo, trial_id, place_result)
    grasp_replay_path = (
        _write_grasp_replay_input(demo, trial_id, grasp=grasp)
        if _param_bool("~use_grasp_replay", False) else "")
    _write_execution_params(
        grasp, place_result, object_size, rollout_path, replay_path,
        grasp_replay_path)

    rospy.loginfo("=" * 60)
    rospy.loginfo("Anchored MT3 place pipeline")
    rospy.loginfo("  query: %s", retrieval_meta.get("query", ""))
    rospy.loginfo("  retrieval: %s selected=%s",
                  retrieval_meta.get("retrieval_mode", ""),
                  retrieval_meta.get("selected_demo_id", demo.get("id", "")))
    rospy.loginfo("  relation alignment: %s", relation_alignment_mode)
    rospy.loginfo("  demo: %s", demo_path)
    rospy.loginfo("  target: [%.3f %.3f %.3f]",
                  target_xyz[0], target_xyz[1], target_xyz[2])
    rospy.loginfo("  anchor: [%.3f %.3f %.3f]",
                  anchor_xyz[0], anchor_xyz[1], anchor_xyz[2])
    rospy.loginfo("  grasp: [%.3f %.3f %.3f]",
                  grasp["position"][0], grasp["position"][1],
                  grasp["position"][2])
    rospy.loginfo("  place: [%.3f %.3f %.3f]",
                  place_result["place_xyz"][0], place_result["place_xyz"][1],
                  place_result["place_xyz"][2])
    rospy.loginfo("  scene packages: target=%s anchor=%s relation=%s",
                  (scene_packages.get("target_package") or {}).get(
                      "package_dir", "(none)"),
                  (scene_packages.get("anchor_package") or {}).get(
                      "package_dir", "(none)"),
                  scene_packages.get("relation", {}).get(
                "relation_path", "(none)"))
    rospy.loginfo("  replay: %s", replay_path or "(disabled/no trajectory)")
    rospy.loginfo("  grasp replay: %s",
                  grasp_replay_path or "(disabled; using scripted top grasp)")
    rospy.loginfo("=" * 60)

    initial_target_gt = _gazebo_pose(
        "~target_gt_model", ["grasp_object", "green", "cube", "object"])
    initial_anchor_gt = _gazebo_pose(
        "~anchor_gt_model",
        ["blue_placement_platform", "blue", "platform", "anchor"])

    if _param_bool("~dry_run", False):
        rospy.loginfo("DRY RUN: params written; skipping execution")
        timing = {
            "total_time_s": time.time() - run_start,
            "perception_time_s": perception_time_s,
            "retrieval_time_s": retrieval_time_s,
            "alignment_time_s": alignment_time_s,
            "planning_time_s": 0.0,
            "robot_execution_time_s": 0.0,
            "planning_call_count": 0,
            "robot_execution_call_count": 0,
            "timing_source": "dry_run",
            "execution_time_s": 0.0,
        }
        _log_experiment_trial(
            trial_id, "dry_run", demo, retrieval_meta, scene,
            grasp, place_result, rollout_path, replay_path,
            object_size=object_size, scene_packages=scene_packages,
            timing=timing, initial_target_gt=initial_target_gt,
            initial_anchor_gt=initial_anchor_gt,
            icp_result=icp_result)
        return True

    _reset_executor_timing_params()
    t_execution = time.time()
    ok, failure_info = _run_executor()
    execution_time_s = time.time() - t_execution
    timing = {
        "total_time_s": time.time() - run_start,
        "perception_time_s": perception_time_s,
        "retrieval_time_s": retrieval_time_s,
        "alignment_time_s": alignment_time_s,
        "execution_time_s": execution_time_s,
    }
    timing.update(_read_executor_timing_params())
    if not ok:
        failure_stage, failure_reason = failure_info
        _log_experiment_trial(
            trial_id, "failed", demo, retrieval_meta, scene,
            grasp, place_result, rollout_path, replay_path,
            object_size=object_size, scene_packages=scene_packages,
            timing=timing, failure_stage=failure_stage or "execution",
            failure_reason=failure_reason or "anchor pipeline execution failed",
            initial_target_gt=initial_target_gt,
            initial_anchor_gt=initial_anchor_gt,
            execution_success=False,
            icp_result=icp_result)
        raise RuntimeError(failure_reason or "anchor pipeline execution failed")
    postcheck = _validate_post_anchor_place_success(
        place_result, (initial_anchor_gt or {}).get("xyz"))
    if not postcheck.get("ok", False):
        failure_stage = postcheck.get("failure_stage", "placement_verification")
        failure_reason = postcheck.get(
            "failure_reason",
            "目标物体最终放置位置或相对托盘关系偏差过大")
        _log_experiment_trial(
            trial_id, "failed", demo, retrieval_meta, scene,
            grasp, place_result, rollout_path, replay_path,
            object_size=object_size, scene_packages=scene_packages,
            timing=timing, failure_stage=failure_stage,
            failure_reason=failure_reason,
            initial_target_gt=initial_target_gt,
            initial_anchor_gt=initial_anchor_gt,
            postcheck=postcheck,
            execution_success=True,
            icp_result=icp_result)
        raise RuntimeError(failure_reason)
    _log_experiment_trial(
        trial_id, "success", demo, retrieval_meta, scene,
        grasp, place_result, rollout_path, replay_path,
        object_size=object_size, scene_packages=scene_packages,
        timing=timing, initial_target_gt=initial_target_gt,
        initial_anchor_gt=initial_anchor_gt,
        postcheck=postcheck, execution_success=True,
        icp_result=icp_result)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("mt3_anchor_place_pipeline failed: %s", exc)
        sys.exit(1)
