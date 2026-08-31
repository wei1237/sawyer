#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal real Sawyer anchor pick-place demonstration recorder.

The human performs the complete task in Zero-G/manual guidance.  This recorder
writes one *real* MT3 anchor-place demo that is simultaneously compatible with:

1. the verified ``mt3_sawyer_real_grasp.py`` top-grasp replay contract; and
2. the structured anchor-place pipeline (grasp_trajectory + place_trajectory).

Important semantics
-------------------
* ``object_info.position_base`` uses XY object center and Z bottom/table contact.
* ``place_info.place_xyz`` is the demonstrated OBJECT placement target, not the
  right_hand release pose.
* ``place_bottleneck_pose_base_frame`` and ``place_release_pose_base_frame`` are
  end-effector poses in Sawyer base.
* ``trajectory`` is intentionally an alias of ``grasp_trajectory`` so the
  already-verified ``RealTopGraspReplay`` can load this same demo directly.
* The complete raw kinesthetic rollout is stored under ``full_trajectory``.

The recorder sends no Sawyer arm motion commands.  Gripper commands/events are
handled by ``real_kinesthetic_recorder.KinestheticRecorder``.
"""

from __future__ import print_function

import copy
import json
import math
import os
import time

import rospy

from mt3_anchor_place_generalization import compute_anchor_place_target
from real_kinesthetic_recorder import (
    KinestheticRecorder,
    build_grasp_segment,
    build_terminal_segment,
    ensure_dir,
)


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CODE_DIR, "demo_library", "real", "recorded")
ROLLOUT_DIR = os.path.join(
    CODE_DIR, "demo_library", "real", "rollout_trajectories")

DEFAULT_TOP_GRASP_CALIBRATION_DEMOS = [
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/real/recorded/"
        "cube_green_top_grasp_real.json"),
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/recorded/"
        "cube_green_top_grasp_real.json"),
]


def required_xyz(prefix):
    keys = ["~%s_x" % prefix, "~%s_y" % prefix, "~%s_z" % prefix]
    missing = [k for k in keys if not rospy.has_param(k)]
    if missing:
        raise RuntimeError(
            "Explicit real %s coordinates required for demo metadata: %s" %
            (prefix, ", ".join(missing)))
    return [float(rospy.get_param(k)) for k in keys]


def _position_m(xyz):
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}


def _identity_orientation():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _atomic_json_dump(payload, path):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _validate_target_metadata(target_xyz, object_size):
    if len(object_size) != 3 or any(float(v) <= 0.0 for v in object_size):
        raise RuntimeError("~object_size must contain three positive values")

    semantics = str(rospy.get_param(
        "~target_z_semantics", "bottom_surface_base")).strip().lower()
    if semantics not in ("bottom_surface_base", "bottom", "object_bottom"):
        raise RuntimeError(
            "For the real top-grasp compatibility schema, ~target_z must be "
            "the target object's bottom/table-contact Z. Set "
            "_target_z_semantics:=bottom_surface_base")

    top_z = float(target_xyz[2]) + float(object_size[2])
    corrected_param = rospy.get_param("~target_top_z_corrected", None)
    if corrected_param is not None:
        corrected_param = float(corrected_param)
        err = abs(corrected_param - top_z)
        if err > 0.006:
            raise RuntimeError(
                "target_z + object height disagrees with "
                "~target_top_z_corrected by %.1f mm (computed %.4f vs %.4f)" %
                (err * 1000.0, top_z, corrected_param))
    return "bottom_surface_base", top_z


def _pose_position(block):
    block = block or {}
    pos = block.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return [float(v) for v in pos[:3]]
    pos_m = block.get("position_m")
    if isinstance(pos_m, dict):
        return [float(pos_m["x"]), float(pos_m["y"]), float(pos_m["z"])]
    raise RuntimeError("Pose block has no position: %s" % block)


def _pose_orientation(block):
    block = block or {}
    ori = block.get("orientation")
    if isinstance(ori, (list, tuple)) and len(ori) >= 4:
        return [float(v) for v in ori[:4]]
    ori_m = block.get("orientation_xyzw")
    if isinstance(ori_m, dict):
        return [
            float(ori_m.get("x", 0.0)),
            float(ori_m.get("y", 0.0)),
            float(ori_m.get("z", 0.0)),
            float(ori_m.get("w", 1.0)),
        ]
    return [0.0, 0.0, 0.0, 1.0]


def _load_json(path):
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        rospy.logwarn("Could not load calibration demo %s: %s", path, exc)
        return None


def _fallback_mouth_offset_from_formal_top_demo():
    """Return a verified TCP->mouth offset from an existing formal top demo.

    This is only a compatibility fallback for recorder versions that do not
    expose fingertip mouth state.  The preferred source is the current recorder
    mouth state captured during this anchor-place demonstration.
    """
    explicit = str(rospy.get_param(
        "~mouth_calibration_demo_path", "")).strip()
    candidates = ([os.path.expanduser(explicit)] if explicit else []) + \
        DEFAULT_TOP_GRASP_CALIBRATION_DEMOS

    for path in candidates:
        payload = _load_json(path)
        if not payload:
            continue
        calib = payload.get("top_grasp_mouth_center_calibration") or {}
        offset = calib.get("mouth_offset_xyz")
        if isinstance(offset, (list, tuple)) and len(offset) >= 3:
            return [float(v) for v in offset[:3]], path

        # Older formal demos may have only absolute mouth center.  Recover the
        # constant TCP->mouth offset from the demo close pose.
        mouth = calib.get("mouth_center_xyz")
        grasp = payload.get("grasp_pose_base_frame") or {}
        if isinstance(mouth, (list, tuple)) and len(mouth) >= 3:
            try:
                tcp = _pose_position(grasp)
                return [
                    float(mouth[0]) - tcp[0],
                    float(mouth[1]) - tcp[1],
                    float(mouth[2]) - tcp[2],
                ], path
            except Exception:
                pass
    return None, ""


def _mouth_calibration(recorder, grasp_close, object_pos):
    """Build the compatibility field consumed by RealTopGraspReplay.

    Prefer mouth state from the current kinesthetic recorder.  If the deployed
    recorder predates that telemetry, reuse only the verified constant
    TCP-to-mouth offset from the existing formal real top-grasp demo and apply
    it to this demo's recorded close TCP.
    """
    threshold = float(rospy.get_param(
        "~top_grasp_centering_warn_threshold_m", 0.003))

    state = getattr(recorder, "bottleneck_mouth_state", None) or {}
    out = {
        "available": bool(state.get("available", False)),
        "label": "real_anchor_place_grasp",
        "desired_xy": [float(object_pos[0]), float(object_pos[1])],
        "centering_warn_threshold_m": float(threshold),
        "source": "current_recorder_bottleneck_mouth_state",
    }
    for key in (
            "reason", "hand_xyz", "left_xyz", "right_xyz",
            "mouth_center_xyz", "mouth_offset_xyz", "used_mouth_offset_xy",
            "mouth_opening_m", "left_frame", "right_frame"):
        if key in state:
            out[key] = state[key]

    if not (out.get("available") and out.get("mouth_center_xyz")):
        mouth_offset, source_path = _fallback_mouth_offset_from_formal_top_demo()
        if mouth_offset is not None:
            close_tcp = _pose_position(grasp_close)
            mouth_center = [
                close_tcp[0] + mouth_offset[0],
                close_tcp[1] + mouth_offset[1],
                close_tcp[2] + mouth_offset[2],
            ]
            out.update({
                "available": True,
                "mouth_center_xyz": mouth_center,
                "mouth_offset_xyz": [float(v) for v in mouth_offset],
                "source": "verified_formal_top_grasp_tcp_to_mouth_offset",
                "calibration_demo_path": source_path,
                "reason": "current recorder mouth telemetry unavailable; "
                          "reused constant gripper TCP-to-mouth calibration",
            })

    if out.get("available") and out.get("mouth_center_xyz"):
        center = [float(v) for v in out["mouth_center_xyz"][:3]]
        err_x = center[0] - float(object_pos[0])
        err_y = center[1] - float(object_pos[1])
        err_norm = math.sqrt(err_x * err_x + err_y * err_y)
        out["mouth_center_xyz"] = center
        out["mouth_center_xy_error_m"] = [float(err_x), float(err_y)]
        out["mouth_center_xy_error_norm_m"] = float(err_norm)
        out["centering_passed"] = bool(err_norm <= threshold)
    return out


def _top_grasp_centering_diagnostics(recorder, grasp_close, object_pos,
                                     mouth_cal):
    threshold = float(rospy.get_param(
        "~top_grasp_centering_warn_threshold_m", 0.003))
    close_tcp = _pose_position(grasp_close)
    state = getattr(recorder, "close_mouth_state", None) or {}
    tcp_xyz = [float(v) for v in state.get("hand_xyz", close_tcp)[:3]]
    object_xy = [float(object_pos[0]), float(object_pos[1])]
    tcp_dx = tcp_xyz[0] - object_xy[0]
    tcp_dy = tcp_xyz[1] - object_xy[1]

    diag = {
        "format": "mt3_top_grasp_centering_diagnostic_v1",
        "label": "real_anchor_place_before_close",
        "available": bool(state.get("available", False)),
        "object_xy": object_xy,
        "tcp_xyz": tcp_xyz,
        "close_event_tcp_xyz": close_tcp,
        "centering_warn_threshold_m": float(threshold),
        "sample": "close_time_before_gripper_command",
        "tcp_object_offset_xy_m": [float(tcp_dx), float(tcp_dy)],
        "tcp_object_offset_xy_norm_m": float(
            math.sqrt(tcp_dx * tcp_dx + tcp_dy * tcp_dy)),
    }

    mouth_xyz = None
    if state.get("available") and state.get("mouth_center_xyz"):
        mouth_xyz = [float(v) for v in state["mouth_center_xyz"][:3]]
        diag["source"] = "current_recorder_close_mouth_state"
    elif mouth_cal.get("available") and mouth_cal.get("mouth_center_xyz"):
        mouth_xyz = [float(v) for v in mouth_cal["mouth_center_xyz"][:3]]
        diag["source"] = mouth_cal.get("source", "top_grasp_mouth_calibration")
        diag["reason"] = "close mouth telemetry unavailable; using compatibility calibration"
    else:
        diag["reason"] = state.get(
            "reason", "close_mouth_state_and_mouth_calibration_unavailable")

    if mouth_xyz is not None:
        mouth_dx = mouth_xyz[0] - object_xy[0]
        mouth_dy = mouth_xyz[1] - object_xy[1]
        mouth_norm = math.sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy)
        diag.update({
            "mouth_center_xyz": mouth_xyz,
            "mouth_object_offset_xy_m": [float(mouth_dx), float(mouth_dy)],
            "mouth_object_offset_xy_norm_m": float(mouth_norm),
            "centering_passed": bool(mouth_norm <= threshold),
        })
    return diag


def _copy_pose_block(block):
    """Return a JSON-safe pose block accepted by RealTopGraspReplay."""
    return {
        "position": _pose_position(block),
        "orientation": _pose_orientation(block),
        "timestamp": float((block or {}).get("timestamp", 0.0)),
        "frame": str((block or {}).get("frame", "base")),
    }


def main():
    rospy.init_node("record_anchor_place_demo_real", anonymous=True)
    ensure_dir(DEMO_DIR)
    ensure_dir(ROLLOUT_DIR)

    demo_id = str(rospy.get_param(
        "~demo_id", "cube_place_on_blue_platform_real")).strip()
    if not demo_id:
        raise RuntimeError("~demo_id must be non-empty")

    overwrite_existing = bool(rospy.get_param(
        "~overwrite_existing_demo", False))
    out_path = os.path.join(DEMO_DIR, "%s.json" % demo_id)
    if os.path.isfile(out_path) and not overwrite_existing:
        raise RuntimeError(
            "Real anchor-place demo already exists: %s. Use a new ~demo_id "
            "or set _overwrite_existing_demo:=true." % out_path)

    target_xyz = required_xyz("target")
    anchor_xyz = required_xyz("anchor")
    object_size = [float(v) for v in rospy.get_param(
        "~object_size", [0.045, 0.045, 0.045])]
    object_z_semantics, object_top_z = _validate_target_metadata(
        target_xyz, object_size)

    anchor_size = rospy.get_param("~anchor_size", None)
    if anchor_size is not None:
        anchor_size = [float(v) for v in anchor_size]

    # Match the formal top-grasp recorder's real-only requirements where the
    # deployed KinestheticRecorder supports them.
    rospy.set_param("~formal_recording", True)
    rospy.set_param("~require_gripper", True)
    rospy.set_param("~require_limb_telemetry", True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(
        ROLLOUT_DIR, "%s_kinesthetic_%s" % (demo_id, stamp))
    recorder = KinestheticRecorder("anchor_pick_place", session_dir)
    trajectory = recorder.record()
    if trajectory is None:
        return False

    rollout_path = os.path.join(session_dir, "rollout.json")
    recorder.save_rollout(trajectory, rollout_path)

    terminal_mark = None
    for event in trajectory.get("events", []):
        if event.get("name") == "terminal_bottleneck":
            terminal_mark = int(event["sample_index"])

    grasp_traj, grasp_bn, grasp_close = build_grasp_segment(
        trajectory,
        terminal_start_idx=terminal_mark,
        pre_samples=int(rospy.get_param("~grasp_bottleneck_pre_samples", 75)),
        post_samples=int(rospy.get_param("~grasp_bottleneck_post_samples", 30)))
    place_traj, place_bn, place_release = build_terminal_segment(
        trajectory,
        kind="place",
        pre_samples=int(rospy.get_param("~place_bottleneck_pre_samples", 75)),
        post_samples=int(rospy.get_param("~place_bottleneck_post_samples", 90)))
    if grasp_traj is None or place_traj is None:
        raise RuntimeError(
            "Could not extract grasp/place segments from explicit c/o events. "
            "Record c=close, t=terminal bottleneck, o=release before saving.")

    # Explicit compatibility aliases for the verified real top-grasp executor.
    grasp_bn_compat = _copy_pose_block(grasp_bn)
    grasp_close_compat = _copy_pose_block(grasp_close)
    top_trajectory = copy.deepcopy(grasp_traj)
    top_trajectory["format"] = "end_effector_pose_twist_gripper_v2"
    top_trajectory["close_index"] = int(grasp_traj.get("close_index", 0))
    top_trajectory["close_event_source"] = str(
        grasp_traj.get("close_event_source", "explicit_event"))
    top_trajectory["base_index"] = 0
    if grasp_traj.get("base_position"):
        top_trajectory["base_position"] = [
            float(v) for v in grasp_traj["base_position"][:3]]
    elif top_trajectory.get("poses"):
        top_trajectory["base_position"] = [
            float(v) for v in top_trajectory["poses"][0]["position"][:3]]

    mouth_cal = _mouth_calibration(recorder, grasp_close_compat, target_xyz)
    if not mouth_cal.get("available", False):
        raise RuntimeError(
            "Anchor-place demo cannot be used by the verified real grasp "
            "executor because gripper mouth calibration is unavailable. "
            "Use the current real_kinesthetic_recorder mouth telemetry or pass "
            "~mouth_calibration_demo_path pointing to the verified formal "
            "cube_green_top_grasp_real.json demo.")
    top_centering = _top_grasp_centering_diagnostics(
        recorder, grasp_close_compat, target_xyz, mouth_cal)

    anchor_name = str(rospy.get_param(
        "~anchor_name", "blue_placement_platform"))
    anchor_category = str(rospy.get_param(
        "~anchor_category", "small_platform"))
    anchor_surface_z_offset = float(rospy.get_param(
        "~anchor_surface_z_offset", 0.0))
    anchor_profile = {
        "name": anchor_name,
        "category": anchor_category,
        "surface_z_offset": anchor_surface_z_offset,
    }
    if anchor_size is not None:
        anchor_profile["size_m"] = anchor_size

    # place_xyz is the demonstrated OBJECT placement target. The right_hand
    # release pose remains a separate recorded end-effector field.
    place_result = compute_anchor_place_target(
        anchor_xyz,
        object_position_base=target_xyz,
        object_size=object_size,
        anchor_profile=anchor_profile,
        override_offset_xyz=rospy.get_param(
            "~anchor_place_offset_xyz", None))
    place_xyz = [float(v) for v in place_result["place_xyz"]]
    offset_xyz = [float(v) for v in place_result["offset_xyz"]]

    demo = {
        "id": demo_id,
        "format": "mt3_anchor_recorded_v2",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recording_mode": "human_kinesthetic_zero_g",
        "execution_environment": "real",
        "arm_motion_commanded_by_recorder": False,
        "task_type": "anchor_pick_place",
        "task": rospy.get_param(
            "~task", "pick the object and place it relative to the anchor"),

        "object_info": {
            "position_base": [float(v) for v in target_xyz],
            "position_semantics": "xy_center_z_bottom_surface_base",
            "z_semantics": object_z_semantics,
            "top_z_base": float(object_top_z),
            "size_m": [float(v) for v in object_size],
            "category": rospy.get_param("~object_category", "cube"),
            "color": rospy.get_param("~object_color", "green"),
        },
        "object_pose_base_frame": {
            "position_m": _position_m(target_xyz),
            "orientation_xyzw": _identity_orientation(),
            "position_semantics": "xy_center_z_bottom_surface_base",
        },
        "anchor_info": {
            "name": anchor_name,
            "category": anchor_category,
            "position_base": [float(v) for v in anchor_xyz],
            "position_semantics": str(rospy.get_param(
                "~anchor_position_semantics", "anchor_geometry_reference_base")),
            "size_m": anchor_size,
            "surface_z_offset": float(anchor_surface_z_offset),
        },
        "place_info": {
            "mode": "anchor_relation",
            "place_xyz": place_xyz,
            "offset_xyz": offset_xyz,
            "place_pose_base_frame": {"position": place_xyz},
            "source": "anchor_relation_geometry",
            "resolution_method": place_result.get("resolution_method"),
            # The demo relation already contains any surface_z_offset applied
            # during recording. Runtime replay must not add it a second time.
            "surface_z_offset_applied_in_demo_relation": float(
                anchor_surface_z_offset),
        },

        # ---- Verified RealTopGraspReplay compatibility contract ----
        "bottleneck_pose_base_frame": grasp_bn_compat,
        "grasp_pose_base_frame": grasp_close_compat,
        "top_grasp_reference": "gripper_mouth_center_required_formal_real",
        "top_grasp_mouth_center_calibration": mouth_cal,
        "top_grasp_centering_diagnostics": top_centering,
        "trajectory": top_trajectory,

        # ---- Structured full MT3 anchor-place contract ----
        "grasp_bottleneck_pose_base_frame": grasp_bn_compat,
        "grasp_close_pose_base_frame": grasp_close_compat,
        "grasp_trajectory": copy.deepcopy(grasp_traj),
        "place_bottleneck_pose_base_frame": _copy_pose_block(place_bn),
        "place_release_pose_base_frame": _copy_pose_block(place_release),
        "place_trajectory": copy.deepcopy(place_traj),
        "full_trajectory": trajectory,

        "scene": {
            "target": {
                "position_base": [float(v) for v in target_xyz],
                "method": "real_manual_param",
            },
            "anchor": {
                "position_base": [float(v) for v in anchor_xyz],
                "method": "real_manual_param",
            },
        },
        "scene_packages": {},
        "real_scene_snapshot": getattr(recorder, "snapshot", None),
        "rollout_trajectory_path": rollout_path,
        "measurement_provenance": {
            "target_xyz_source": "explicit_real_demo_metadata",
            "anchor_xyz_source": "explicit_real_demo_metadata",
            "target_top_z_source": "target_z_bottom_plus_object_height",
            "mouth_calibration_source": mouth_cal.get("source", ""),
        },
        "language_tags": [
            "anchor place", "pick and place", "top-down grasp",
            "relative placement", "抓取", "相对放置"],
        "language_description": rospy.get_param(
            "~language_description",
            "Pick up the object and place it relative to the anchor"),
    }

    _atomic_json_dump(demo, out_path)
    rospy.loginfo("=" * 68)
    rospy.loginfo("REAL MT3 ANCHOR-PLACE DEMO SAVED")
    rospy.loginfo("  demo: %s", out_path)
    rospy.loginfo("  rollout: %s", rollout_path)
    rospy.loginfo("  grasp poses: %d", len(grasp_traj.get("poses") or []))
    rospy.loginfo("  place poses: %d", len(place_traj.get("poses") or []))
    rospy.loginfo(
        "  demo anchor->place offset = [%.3f, %.3f, %.3f] m", *offset_xyz)
    rospy.loginfo("  mouth calibration source: %s", mouth_cal.get("source", ""))
    rospy.loginfo("=" * 68)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("record_anchor_place_demo_real failed: %s", exc)
        raise
