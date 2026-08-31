#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal real-Sawyer top-grasp demonstration recorder.

One successful recording writes the complete reusable MT3 demonstration set:
  * 30 Hz right_hand pose trajectory + twists + gripper/joint telemetry
  * explicit grasp-bottleneck and gripper-close events
  * bottleneck/grasp poses in base
  * object metadata with explicit XY-center / Z-bottom semantics
  * synchronous bottleneck ASC60C RGB/depth/CameraInfo + LangSAM mask
  * standard demo_<id> scene package including segmented pointcloud.npy
  * open-gripper mouth-center calibration when finger-tip TFs are available

The recorder never commands Sawyer arm motion; the human supplies interaction
motion while constrained Zero-G is active. Keyboard c/o may command the gripper.
"""

from __future__ import print_function

import json
import math
import os
import shutil
import time

import cv2
import numpy as np
import rospy

from mt3_scene_package import save_scene_package
from real_kinesthetic_recorder import (
    KinestheticRecorder, ensure_dir, event_index, pose_as_bottleneck,
)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CODE_DIR, "demo_library", "real", "recorded")
ROLLOUT_DIR = os.path.join(
    CODE_DIR, "demo_library", "real", "rollout_trajectories")
SCENE_PACKAGE_DIR = os.path.join(
    CODE_DIR, "demo_library", "real", "scene_packages")
DEFAULT_REAL_MASK = "/mnt/hgfs2/ascamera_data/current_mask.npy"
DEFAULT_CAMERA_FRAME = "ascamera_hp60c_color_0"


def required_xyz(prefix):
    keys = ["~%s_x" % prefix, "~%s_y" % prefix, "~%s_z" % prefix]
    missing = [k for k in keys if not rospy.has_param(k)]
    if missing:
        raise RuntimeError(
            "Real demo metadata requires explicit %s coordinates: %s. "
            "Do not reuse Gazebo table defaults." % (prefix, ", ".join(missing)))
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


def _validate_object_metadata(object_pos, object_size):
    if len(object_size) != 3 or any(float(v) <= 0.0 for v in object_size):
        raise RuntimeError("~object_size must contain three positive values")
    semantics = str(rospy.get_param(
        "~object_z_semantics", "bottom_surface_base")).strip().lower()
    if semantics not in ("bottom_surface_base", "bottom", "object_bottom"):
        raise RuntimeError(
            "For this top-grasp schema, ~object_z must be the object bottom/table-contact Z. "
            "Use _object_z_semantics:=bottom_surface_base")

    object_top_z = float(object_pos[2]) + float(object_size[2])
    corrected_param = rospy.get_param("~object_top_z_corrected", None)
    if corrected_param is not None:
        corrected_param = float(corrected_param)
        err = abs(corrected_param - object_top_z)
        if err > 0.006:
            raise RuntimeError(
                "object_z + height disagrees with ~object_top_z_corrected by %.1f mm "
                "(computed %.4f vs supplied %.4f)" %
                (err * 1000.0, object_top_z, corrected_param))
    return "bottom_surface_base", object_top_z


def _mouth_calibration(recorder, object_pos):
    state = recorder.bottleneck_mouth_state or {}
    threshold = float(rospy.get_param("~top_grasp_centering_warn_threshold_m", 0.003))
    out = {
        "available": bool(state.get("available", False)),
        "label": "real_top_grasp_bottleneck",
        "desired_xy": [float(object_pos[0]), float(object_pos[1])],
        "centering_warn_threshold_m": float(threshold),
    }
    for key in (
            "reason", "hand_xyz", "left_xyz", "right_xyz", "mouth_center_xyz",
            "mouth_offset_xyz", "used_mouth_offset_xy", "mouth_opening_m",
            "left_frame", "right_frame"):
        if key in state:
            out[key] = state[key]
    if state.get("available") and state.get("mouth_center_xyz"):
        center = state["mouth_center_xyz"]
        err_x = float(center[0]) - float(object_pos[0])
        err_y = float(center[1]) - float(object_pos[1])
        err_norm = math.sqrt(err_x * err_x + err_y * err_y)
        out["mouth_center_xy_error_m"] = [
            float(err_x),
            float(err_y),
        ]
        out["mouth_center_xy_error_norm_m"] = float(err_norm)
        out["centering_passed"] = bool(err_norm <= threshold)
    return out


def _top_grasp_centering_diagnostics(recorder, close_pose, object_pos):
    """Report whether the recorded close primitive is actually centered."""
    threshold = float(rospy.get_param("~top_grasp_centering_warn_threshold_m", 0.003))
    object_xy = [float(object_pos[0]), float(object_pos[1])]
    close_pose_tcp_xyz = [float(v) for v in close_pose.get("position", [0.0, 0.0, 0.0])[:3]]
    state = getattr(recorder, "close_mouth_state", None) or {}
    tcp_xyz = [float(v) for v in state.get("hand_xyz", close_pose_tcp_xyz)[:3]]
    diag = {
        "format": "mt3_top_grasp_centering_diagnostic_v1",
        "label": "real_top_grasp_before_close",
        "available": bool(state.get("available", False)),
        "object_xy": object_xy,
        "tcp_xyz": tcp_xyz,
        "close_event_tcp_xyz": close_pose_tcp_xyz,
        "centering_warn_threshold_m": float(threshold),
        "sample": "close_time_before_gripper_command",
    }
    tcp_dx = tcp_xyz[0] - object_xy[0]
    tcp_dy = tcp_xyz[1] - object_xy[1]
    diag["tcp_object_offset_xy_m"] = [float(tcp_dx), float(tcp_dy)]
    diag["tcp_object_offset_xy_norm_m"] = float(
        math.sqrt(tcp_dx * tcp_dx + tcp_dy * tcp_dy))
    if not state.get("available", False):
        diag["reason"] = state.get("reason", "close_mouth_state_unavailable")
        rospy.logwarn("TOP GRASP CENTERING DEBUG unavailable: %s", diag["reason"])
        return diag

    mouth_xyz = [float(v) for v in state.get("mouth_center_xyz", [0.0, 0.0, 0.0])[:3]]
    mouth_dx = mouth_xyz[0] - object_xy[0]
    mouth_dy = mouth_xyz[1] - object_xy[1]
    mouth_norm = math.sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy)
    diag.update({
        "mouth_center_xyz": mouth_xyz,
        "mouth_object_offset_xy_m": [float(mouth_dx), float(mouth_dy)],
        "mouth_object_offset_xy_norm_m": float(mouth_norm),
        "mouth_opening_m": state.get("mouth_opening_m"),
        "mouth_offset_xyz": state.get("mouth_offset_xyz"),
        "centering_passed": bool(mouth_norm <= threshold),
    })
    rospy.loginfo(
        "TOP GRASP CENTERING DEBUG: object_xy=[%.4f %.4f] tcp_xy=[%.4f %.4f] "
        "mouth_xy=[%.4f %.4f] tcp_obj=[%.1f %.1f]mm |tcp|=%.1fmm "
        "mouth_obj=[%.1f %.1f]mm |mouth|=%.1fmm threshold=%.1fmm result=%s",
        object_xy[0], object_xy[1],
        tcp_xyz[0], tcp_xyz[1],
        mouth_xyz[0], mouth_xyz[1],
        tcp_dx * 1000.0, tcp_dy * 1000.0,
        diag["tcp_object_offset_xy_norm_m"] * 1000.0,
        mouth_dx * 1000.0, mouth_dy * 1000.0,
        mouth_norm * 1000.0,
        threshold * 1000.0,
        "PASS" if diag["centering_passed"] else "WARN")
    if not diag["centering_passed"]:
        rospy.logwarn(
            "Recorded top grasp is off-center: mouth-object XY norm %.1f mm > %.1f mm",
            mouth_norm * 1000.0, threshold * 1000.0)
    return diag


def _scene_package_from_snapshot(recorder, demo_name, object_pos, object_size,
                                 object_top_z, bottleneck_pose, package_name=None):
    snapshot = recorder.snapshot or {}
    required = ("rgb", "depth", "camera_info", "mask")
    missing = [k for k in required if not snapshot.get(k) or not os.path.isfile(snapshot[k])]
    if missing:
        raise RuntimeError(
            "Formal real demo scene snapshot is incomplete; missing: %s" % ", ".join(missing))

    bgr = cv2.imread(snapshot["rgb"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Cannot read bottleneck RGB snapshot: %s" % snapshot["rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(snapshot["depth"])
    mask = np.load(snapshot["mask"]).astype(bool)
    with open(snapshot["camera_info"], "r") as f:
        info = json.load(f)
    Kflat = info.get("K") or []
    if len(Kflat) != 9:
        raise RuntimeError("Bottleneck CameraInfo K is missing/invalid")
    K = np.asarray(Kflat, dtype=np.float64).reshape(3, 3)
    if mask.shape != np.asarray(depth).shape[:2]:
        raise RuntimeError(
            "Bottleneck mask/depth mismatch: %s vs %s" %
            (str(mask.shape), str(np.asarray(depth).shape[:2])))
    if int(np.count_nonzero(mask)) < 20:
        raise RuntimeError("Bottleneck mask has too few pixels")

    camera_frame = str(
        info.get("camera_info_frame_id") or snapshot.get("frame_id") or DEFAULT_CAMERA_FRAME)
    scene_data = {
        "rgb": rgb,
        "depth": np.asarray(depth),
        "segmap": mask,
        "intrinsics": K,
        # Point cloud is intentionally camera-frame.  Base-frame object geometry is
        # carried separately in metadata so the +44 mm correction is never baked
        # into the camera points themselves.
        "pose": {
            "method": "real_kinesthetic_bottleneck_snapshot",
            "confidence": 1.0,
            "source_frame": camera_frame,
            "frame": camera_frame,
            "bottleneck_right_hand_base": bottleneck_pose,
        },
    }

    package_name = package_name or ("demo_%s" % demo_name)
    package_path = os.path.join(SCENE_PACKAGE_DIR, package_name)
    if os.path.isdir(package_path):
        shutil.rmtree(package_path)

    package = save_scene_package(
        scene_data,
        SCENE_PACKAGE_DIR,
        name=package_name,
        role="recorded_real_demo",
        extra_metadata={
            "demo_id": demo_name,
            "execution_environment": "real",
            "camera_frame": camera_frame,
            "object_position_base": [float(v) for v in object_pos],
            "object_position_semantics": "xy_center_z_bottom_surface_base",
            "object_size": [float(v) for v in object_size],
            "object_top_z_base": float(object_top_z),
            "object_top_z_source": "explicit_bottom_plus_height",
            "real_top_z_offset_m": float(rospy.get_param("~real_top_z_offset_m", 0.044)),
            "object_top_z_raw_camera_derived": rospy.get_param("~object_top_z_raw", None),
            "object_top_z_corrected_input": rospy.get_param("~object_top_z_corrected", None),
        })

    points_path = os.path.join(package["package_dir"], "pointcloud.npy")
    if not os.path.isfile(points_path):
        raise RuntimeError("Scene package did not produce pointcloud.npy")
    points = np.load(points_path)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise RuntimeError("Scene package pointcloud is invalid: shape=%s" % (str(points.shape),))
    rospy.loginfo(
        "Formal demo scene package ready: %s (mask=%d px, points=%d)",
        package["package_dir"], int(np.count_nonzero(mask)), int(len(points)))
    return package


def _promote_scene_package(package, demo_name, overwrite_existing=False):
    pending_dir = package.get("package_dir", "")
    if not pending_dir or not os.path.isdir(pending_dir):
        raise RuntimeError("Pending scene package directory is missing")
    final_name = "demo_%s" % demo_name
    final_dir = os.path.join(SCENE_PACKAGE_DIR, final_name)
    if os.path.abspath(pending_dir) != os.path.abspath(final_dir):
        if os.path.isdir(final_dir):
            if not overwrite_existing:
                raise RuntimeError(
                    "Scene package already exists: %s" % final_dir)
            shutil.rmtree(final_dir)
        os.replace(pending_dir, final_dir)
    meta_path = os.path.join(final_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["name"] = final_name
        meta["formal_demo_integrity"] = "PASS"
        _atomic_json_dump(meta, meta_path)
    out = dict(package)
    out["name"] = final_name
    out["package_dir"] = final_dir
    return out


def _load_snapshot_meta(recorder):
    snapshot = recorder.snapshot or {}
    path = snapshot.get("camera_info")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _integrity_report(recorder, trajectory, mouth_cal, package, top_centering=None):
    poses = trajectory.get("poses") or []
    events = trajectory.get("events") or []
    metrics = dict(trajectory.get("integrity_metrics") or {})
    close_events = [e for e in events if str(e.get("name", "")) == "gripper_close"]
    close_idx = trajectory.get("close_index", None)
    try:
        close_idx = int(close_idx)
    except Exception:
        close_idx = -1

    snapshot_meta = _load_snapshot_meta(recorder)
    mask_binding = snapshot_meta.get("mask_binding") or metrics.get("mask_binding") or {}
    rgb_depth_dt_s = snapshot_meta.get("rgb_depth_dt_s", metrics.get("rgb_depth_dt_s"))
    rgb_depth_threshold_s = snapshot_meta.get(
        "rgb_depth_sync_threshold_s", metrics.get("rgb_depth_sync_threshold_s"))
    rgb_depth_sync_passed = bool(snapshot_meta.get(
        "rgb_depth_sync_passed", metrics.get("rgb_depth_sync_passed", False)))

    package_dir = package.get("package_dir", "") if isinstance(package, dict) else ""
    required_scene_files = [
        "rgb.png", "depth.npy", "segmap.npy", "intrinsics.npy",
        "pointcloud.npy", "metadata.json",
    ]
    missing_scene_files = [
        name for name in required_scene_files
        if not package_dir or not os.path.isfile(os.path.join(package_dir, name))
    ]
    pointcloud_points = 0
    if package_dir and os.path.isfile(os.path.join(package_dir, "pointcloud.npy")):
        try:
            pts = np.load(os.path.join(package_dir, "pointcloud.npy"))
            if pts.ndim == 2 and pts.shape[1] == 3:
                pointcloud_points = int(len(pts))
        except Exception:
            pointcloud_points = 0

    joint = metrics.get("joint_telemetry") or {}
    hard_gates = {
        "pose_count_min_10": len(poses) >= 10,
        "single_close_event": len(close_events) == 1,
        "close_index_valid": 0 <= close_idx < len(poses),
        "post_close_samples": int(metrics.get("post_close_samples", 0)) >= int(
            metrics.get("min_post_close_samples", 15)),
        "post_close_lift": float(metrics.get("post_close_lift_m", 0.0)) >= float(
            metrics.get("min_post_close_lift_m", 0.040)),
        "mouth_calibration": bool(mouth_cal.get("available", False)),
        "rgb_depth_sync": rgb_depth_sync_passed,
        "mask_binding": bool(mask_binding.get("passed", False)),
        "scene_package_files": len(missing_scene_files) == 0,
        "pointcloud_min_20": pointcloud_points >= 20,
    }
    warnings = []
    joint_cov = float(joint.get("coverage", 0.0) or 0.0)
    joint_thr = float(joint.get("warn_threshold", 0.95) or 0.95)
    if joint_cov < joint_thr:
        warnings.append(
            "joint telemetry coverage %.1f%% < %.1f%%; re-record recommended" %
            (joint_cov * 100.0, joint_thr * 100.0))
    if top_centering and top_centering.get("available") and not top_centering.get("centering_passed", False):
        warnings.append(
            "top grasp mouth-object XY %.1f mm > %.1f mm; this demo is not a center primitive" %
            (float(top_centering.get("mouth_object_offset_xy_norm_m", 0.0)) * 1000.0,
             float(top_centering.get("centering_warn_threshold_m", 0.003)) * 1000.0))

    status = "PASS" if all(bool(v) for v in hard_gates.values()) else "FAIL"
    return {
        "format": "mt3_formal_demo_integrity_v1",
        "status": status,
        "hard_gates": hard_gates,
        "warnings": warnings,
        "poses": int(len(poses)),
        "velocities": int(len(trajectory.get("velocities") or [])),
        "duration_s": float(metrics.get("duration_s", trajectory.get("duration_s", 0.0)) or 0.0),
        "effective_rate_hz": float(metrics.get(
            "effective_rate_hz", trajectory.get("effective_rate_hz", 0.0)) or 0.0),
        "close_events": int(len(close_events)),
        "close_index": int(close_idx),
        "pre_close_samples": int(metrics.get("pre_close_samples", 0)),
        "post_close_samples": int(metrics.get("post_close_samples", 0)),
        "post_close_lift_m": float(metrics.get("post_close_lift_m", 0.0)),
        "post_close_lift_mm": float(metrics.get("post_close_lift_m", 0.0)) * 1000.0,
        "joint_coverage": float(joint_cov),
        "joint_coverage_percent": float(joint_cov * 100.0),
        "mouth_calibration": bool(mouth_cal.get("available", False)),
        "top_grasp_centering": top_centering or {},
        "rgb_depth_dt_s": rgb_depth_dt_s,
        "rgb_depth_dt_ms": (float(rgb_depth_dt_s) * 1000.0
                            if rgb_depth_dt_s is not None else None),
        "rgb_depth_sync_threshold_s": rgb_depth_threshold_s,
        "mask_binding": mask_binding,
        "pointcloud_points": int(pointcloud_points),
        "scene_package_missing_files": missing_scene_files,
        "scene_package_passed": len(missing_scene_files) == 0 and pointcloud_points >= 20,
    }


def _print_integrity_report(report):
    print("=" * 68)
    print("FORMAL DEMO INTEGRITY: %s" % report.get("status", "FAIL"))
    print("  poses                  %d" % int(report.get("poses", 0)))
    print("  effective rate         %.2f Hz" % float(report.get("effective_rate_hz", 0.0)))
    print("  duration               %.2f s" % float(report.get("duration_s", 0.0)))
    print("  close events           %d" % int(report.get("close_events", 0)))
    print("  close index            %d" % int(report.get("close_index", -1)))
    print("  pre-close samples      %d" % int(report.get("pre_close_samples", 0)))
    print("  post-close samples     %d" % int(report.get("post_close_samples", 0)))
    print("  post-close lift        %.1f mm" % float(report.get("post_close_lift_mm", 0.0)))
    print("  joint coverage         %.1f %%" % float(report.get("joint_coverage_percent", 0.0)))
    print("  mouth calibration      %s" % ("PASS" if report.get("mouth_calibration") else "FAIL"))
    centering = report.get("top_grasp_centering") or {}
    if centering.get("available"):
        off = centering.get("mouth_object_offset_xy_m") or [0.0, 0.0]
        norm = float(centering.get("mouth_object_offset_xy_norm_m", 0.0))
        threshold = float(centering.get("centering_warn_threshold_m", 0.003))
        print("  mouth-object XY        [%.1f, %.1f] mm |norm| %.1f mm (%s, threshold %.1f mm)" % (
            float(off[0]) * 1000.0,
            float(off[1]) * 1000.0,
            norm * 1000.0,
            "PASS" if centering.get("centering_passed") else "WARN",
            threshold * 1000.0))
    else:
        print("  mouth-object XY        unavailable")
    dt_ms = report.get("rgb_depth_dt_ms")
    print("  RGB-depth dt           %s" % (
        "%.1f ms" % float(dt_ms) if dt_ms is not None else "n/a"))
    binding = report.get("mask_binding") or {}
    print("  mask binding           %s (%s)" % (
        "PASS" if binding.get("passed") else "FAIL", binding.get("mode", "unverified")))
    print("  pointcloud points      %d" % int(report.get("pointcloud_points", 0)))
    print("  scene package          %s" % (
        "PASS" if report.get("scene_package_passed") else "FAIL"))
    for warning in report.get("warnings") or []:
        print("  WARNING                %s" % warning)
    if report.get("status") != "PASS":
        failed = [k for k, v in (report.get("hard_gates") or {}).items() if not v]
        print("  FAILED GATES           %s" % ", ".join(failed))
    print("=" * 68)


def main():
    rospy.init_node("mt3_record_real_top_demo", anonymous=True)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(ROLLOUT_DIR)
    ensure_dir(SCENE_PACKAGE_DIR)

    demo_name = str(rospy.get_param(
        "~demo_name", "cube_green_top_grasp_real")).strip()
    if not demo_name:
        raise RuntimeError("~demo_name must be non-empty")

    overwrite_existing = bool(rospy.get_param("~overwrite_existing_demo", False))
    existing_demo = os.path.join(OUTPUT_DIR, "%s.json" % demo_name)
    if os.path.isfile(existing_demo) and not overwrite_existing:
        raise RuntimeError(
            "A recorded demo with id '%s' already exists: %s. "
            "Use a new ~demo_name or explicitly set _overwrite_existing_demo:=true." %
            (demo_name, existing_demo))
    existing_package = os.path.join(SCENE_PACKAGE_DIR, "demo_%s" % demo_name)
    if os.path.isdir(existing_package) and not overwrite_existing:
        raise RuntimeError(
            "A scene package for demo id '%s' already exists: %s. "
            "Use a new ~demo_name or explicitly set _overwrite_existing_demo:=true." %
            (demo_name, existing_package))

    object_pos = required_xyz("object")
    object_size = [float(v) for v in rospy.get_param(
        "~object_size", [0.045, 0.045, 0.045])]
    object_z_semantics, object_top_z = _validate_object_metadata(object_pos, object_size)

    mask_path = os.path.expanduser(str(rospy.get_param("~mask_path", DEFAULT_REAL_MASK)))
    if not os.path.isfile(mask_path):
        raise RuntimeError("Current LangSAM mask does not exist: %s" % mask_path)
    mask = np.load(mask_path)
    if mask.ndim != 2 or int(np.count_nonzero(mask)) < 20:
        raise RuntimeError("Current LangSAM mask is invalid: %s" % mask_path)

    # Force the formal recorder settings before its constructor reads private params.
    rospy.set_param("~mask_path", mask_path)
    rospy.set_param("~capture_rgbd", True)
    rospy.set_param("~require_scene_snapshot", True)
    rospy.set_param("~require_snapshot_mask", True)
    rospy.set_param("~synchronous_bottleneck_snapshot", True)
    rospy.set_param("~formal_recording", True)
    rospy.set_param("~require_gripper", True)
    rospy.set_param("~require_limb_telemetry", True)
    rospy.set_param("~require_mouth_calibration", True)
    if not rospy.has_param("~min_post_close_samples"):
        rospy.set_param("~min_post_close_samples", 15)
    if not rospy.has_param("~min_post_close_lift_m"):
        rospy.set_param("~min_post_close_lift_m", 0.040)
    if not rospy.has_param("~max_rgb_depth_dt_s"):
        rospy.set_param("~max_rgb_depth_dt_s", 0.10)
    if not rospy.has_param("~joint_coverage_warn_threshold"):
        rospy.set_param("~joint_coverage_warn_threshold", 0.95)

    rospy.loginfo("=" * 60)
    rospy.loginfo("FORMAL REAL TOP-GRASP DEMO PREP")
    rospy.loginfo("  demo_id: %s", demo_name)
    rospy.loginfo("  object XY center: [%.4f, %.4f]", object_pos[0], object_pos[1])
    rospy.loginfo("  object Z bottom: %.4f", object_pos[2])
    rospy.loginfo("  object top Z: %.4f", object_top_z)
    rospy.loginfo("  object size: %s", object_size)
    rospy.loginfo("  LangSAM mask: %s (%d px)", mask_path, int(np.count_nonzero(mask)))
    rospy.loginfo("=" * 60)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(
        ROLLOUT_DIR, "%s_kinesthetic_%s" % (demo_name, stamp))
    recorder = KinestheticRecorder("top_grasp", session_dir)
    trajectory = recorder.record()
    if trajectory is None:
        return False

    close_idx = event_index(trajectory["events"], ["gripper_close"])
    if close_idx is None or close_idx < 0 or close_idx >= len(trajectory["poses"]):
        raise RuntimeError("Recorded gripper_close sample index is invalid")
    first = trajectory["poses"][0]
    close_pose = trajectory["poses"][close_idx]

    rollout_path = os.path.join(session_dir, "rollout.json")
    recorder.save_rollout(trajectory, rollout_path)

    # Keep the existing top-grasp replay schema used by mt3_generalize.
    top_trajectory = dict(trajectory)
    top_trajectory["format"] = "end_effector_pose_twist_gripper_v2"
    top_trajectory["close_index"] = int(close_idx)
    top_trajectory["close_event_source"] = "explicit_event"
    top_trajectory["base_index"] = 0
    top_trajectory["base_position"] = [float(v) for v in first["position"][:3]]

    bottleneck_pose = pose_as_bottleneck(first)
    mouth_cal = _mouth_calibration(recorder, object_pos)
    if not mouth_cal.get("available", False):
        raise RuntimeError(
            "FORMAL DEMO INVALID: fingertip/mouth calibration is required")
    top_centering = _top_grasp_centering_diagnostics(recorder, close_pose, object_pos)
    pending_package_name = "demo_%s__pending_%s" % (demo_name, stamp)
    package = _scene_package_from_snapshot(
        recorder, demo_name, object_pos, object_size, object_top_z, bottleneck_pose,
        package_name=pending_package_name)

    integrity = _integrity_report(
        recorder, top_trajectory, mouth_cal, package, top_centering=top_centering)
    integrity_path = os.path.join(session_dir, "integrity_report.json")
    _atomic_json_dump(integrity, integrity_path)
    _print_integrity_report(integrity)
    if integrity.get("status") != "PASS":
        try:
            if package.get("package_dir") and os.path.isdir(package["package_dir"]):
                shutil.rmtree(package["package_dir"])
        except Exception:
            pass
        raise RuntimeError(
            "FORMAL DEMO INTEGRITY FAILED; demo_library/recorded was not written")

    # Only after the integrity report passes do we promote the pending package
    # to the stable demo_<id> name consumed by mt3_generalize.
    package = _promote_scene_package(
        package, demo_name, overwrite_existing=overwrite_existing)

    demo = {
        "id": demo_name,
        "format": "mt3_recorded_v2",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recording_mode": "human_kinesthetic_zero_g",
        "execution_environment": "real",
        "arm_motion_commanded_by_recorder": False,
        "task_type": "grasp",
        "object_info": {
            "position_base": [float(v) for v in object_pos],
            "position_semantics": "xy_center_z_bottom_surface_base",
            "z_semantics": object_z_semantics,
            "top_z_base": float(object_top_z),
            "size_m": [float(v) for v in object_size],
            "category": rospy.get_param("~object_category", "cube"),
            "color": rospy.get_param("~object_color", "green"),
        },
        # Explicit compatibility field required by current mt3_generalize alignment.
        "object_pose_base_frame": {
            "position_m": _position_m(object_pos),
            "orientation_xyzw": _identity_orientation(),
            "position_semantics": "xy_center_z_bottom_surface_base",
        },
        "bottleneck_pose_base_frame": bottleneck_pose,
        "grasp_pose_base_frame": {
            "position_m": _position_m(close_pose["position"]),
            "orientation_xyzw": {
                "x": float(close_pose["orientation"][0]),
                "y": float(close_pose["orientation"][1]),
                "z": float(close_pose["orientation"][2]),
                "w": float(close_pose["orientation"][3]),
            },
            "timestamp": float(close_pose.get("timestamp", 0.0)),
            "event_sample_index": int(close_idx),
        },
        "top_grasp_reference": "gripper_mouth_center_required_formal_real",
        "top_grasp_mouth_center_calibration": mouth_cal,
        "top_grasp_centering_diagnostics": top_centering,
        "trajectory": top_trajectory,
        "formal_demo_integrity": integrity,
        "formal_demo_integrity_report_path": integrity_path,
        "language_tags": ["grasp", "pick up", "top-down grasp", "抓取"],
        "language_description": rospy.get_param(
            "~language_description", "Pick up the object from above"),
        "approach_direction": [0.0, 0.0, -1.0],
        "retract_direction": [0.0, 0.0, 1.0],
        "gripper_opening_m": (
            mouth_cal.get("mouth_opening_m")
            if mouth_cal.get("mouth_opening_m") is not None
            else rospy.get_param("~gripper_opening_m", None)),
        "real_scene_snapshot": recorder.snapshot,
        "scene_package": package.get("name"),
        "scene_package_dir": os.path.relpath(package["package_dir"], CODE_DIR),
        "scene_package_stats": package.get("stats", {}),
        "rollout_trajectory_path": rollout_path,
        "measurement_provenance": {
            "mask_path": mask_path,
            "object_top_z_raw": rospy.get_param("~object_top_z_raw", None),
            "object_top_z_corrected": rospy.get_param(
                "~object_top_z_corrected", float(object_top_z)),
            "real_top_z_offset_m": float(rospy.get_param("~real_top_z_offset_m", 0.044)),
            "object_bottom_z_derived_from_top_minus_height": True,
        },
    }

    out = os.path.join(OUTPUT_DIR, "%s.json" % demo_name)
    report_out = os.path.join(OUTPUT_DIR, "%s.integrity.json" % demo_name)
    _atomic_json_dump(demo, out)
    _atomic_json_dump(integrity, report_out)
    rospy.loginfo("=" * 60)
    rospy.loginfo("FORMAL REAL TOP-GRASP DEMO SAVED")
    rospy.loginfo("  demo json: %s", out)
    rospy.loginfo("  rollout: %s", rollout_path)
    rospy.loginfo("  scene package: %s", package["package_dir"])
    rospy.loginfo("  integrity report: %s", report_out)
    rospy.loginfo("  trajectory poses: %d", len(top_trajectory.get("poses") or []))
    rospy.loginfo("  trajectory twists: %d", len(top_trajectory.get("velocities") or []))
    rospy.loginfo("  mouth calibration available: %s", mouth_cal.get("available", False))
    rospy.loginfo("=" * 60)
    return True


if __name__ == "__main__":
    try:
        ok = main()
        raise SystemExit(0 if ok else 1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Real demo recording interrupted")
        raise SystemExit(130)
    except Exception as exc:
        rospy.logerr("record_demo_real failed: %s", exc)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
