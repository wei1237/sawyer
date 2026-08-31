#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only preflight for formal Sawyer kinesthetic demo recording.

Checks only telemetry/files; it never commands the arm or gripper.
"""
from __future__ import print_function

import importlib
import os
import sys

import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from mt3_scene_package import pointcloud_from_depth_mask
from real_kinesthetic_recorder import (
    capture_synchronized_rgbd,
    verify_mask_binding,
)

try:
    from intera_interface import Gripper, Limb
except Exception:
    Gripper = None
    Limb = None

DEFAULT_MASK = "/mnt/hgfs2/ascamera_data/current_mask.npy"
DEFAULT_MASK_RGB = "/mnt/hgfs2/ascamera_data/current_rgb.png"
RGB_TOPIC = "/ascamera_hp60c/rgb0/image"
DEPTH_TOPIC = "/ascamera_hp60c/depth0/image_raw"
INFO_TOPIC = "/ascamera_hp60c/rgb0/camera_info"


def ok(msg):
    print("[OK]   " + msg)


def warn(msg):
    print("[WARN] " + msg)


def fail(msg):
    print("[FAIL] " + msg)
    return False


def main():
    rospy.init_node("mt3_real_demo_preflight", anonymous=True)
    good = True

    print("=" * 68)
    print("MT3 REAL DEMO PREFLIGHT — READ ONLY / NO ROBOT MOTION")
    print("=" * 68)

    for module in (
            "real_kinesthetic_recorder", "record_demo_real", "mt3_scene_package",
            "mt3_generalize", "mt3_pipeline_real", "mt3_perception_real",
            "mt3_alignment_real"):
        try:
            mod = importlib.import_module(module)
            ok("import %s" % module)
            if module == "mt3_pipeline_real":
                parent_name = getattr(getattr(mod, "_sim", None), "__name__", "")
                if parent_name != "mt3_generalize":
                    good = fail("mt3_pipeline_real parent is %s, expected mt3_generalize" % parent_name) and good
                else:
                    ok("mt3_pipeline_real reuses mt3_generalize")
        except Exception as exc:
            good = fail("import %s: %s" % (module, exc)) and good

    mask_path = os.path.expanduser(str(rospy.get_param("~mask_path", DEFAULT_MASK)))
    mask_source_rgb_path = os.path.expanduser(str(rospy.get_param(
        "~mask_source_rgb_path", DEFAULT_MASK_RGB)))
    mask_metadata_path = os.path.expanduser(str(rospy.get_param(
        "~mask_metadata_path", "")))
    max_mask_age_s = float(rospy.get_param("~max_mask_age_s", 300.0))
    max_mask_rgb_mtime_delta_s = float(rospy.get_param(
        "~max_mask_rgb_mtime_delta_s", 120.0))
    allow_mask_mtime_fallback = bool(rospy.get_param(
        "~allow_mask_mtime_fallback", True))
    try:
        mask = np.load(mask_path).astype(bool)
        if mask.ndim != 2 or int(np.count_nonzero(mask)) < 20:
            raise RuntimeError("invalid shape/pixel count")
        ok("LangSAM mask %s shape=%s pixels=%d" %
           (mask_path, str(mask.shape), int(np.count_nonzero(mask))))
        binding = verify_mask_binding(
            mask_path, source_rgb_path=mask_source_rgb_path,
            metadata_path=mask_metadata_path, max_age_s=max_mask_age_s,
            max_rgb_mtime_delta_s=max_mask_rgb_mtime_delta_s,
            allow_mtime_fallback=allow_mask_mtime_fallback)
        if not binding.get("passed", False):
            good = fail("mask freshness/source binding: %s" %
                        binding.get("reason", "unknown")) and good
        else:
            ok("mask binding mode=%s reason=%s" %
               (binding.get("mode", ""), binding.get("reason", "")))
            if binding.get("mode") == "mtime_fallback":
                warn("mask binding is using mtime fallback; SHA256 metadata binding is preferred")
    except Exception as exc:
        mask = None
        good = fail("LangSAM mask %s: %s" % (mask_path, exc)) and good

    bridge = CvBridge()
    max_rgb_depth_dt_s = float(rospy.get_param("~max_rgb_depth_dt_s", 0.10))
    try:
        rgb_msg, depth_msg, dt_s, sync_diag = capture_synchronized_rgbd(
            RGB_TOPIC, DEPTH_TOPIC, timeout_s=4.0,
            max_dt_s=max_rgb_depth_dt_s)
        rgb = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        ok("RGB %s shape=%s frame=%s" %
           (RGB_TOPIC, str(np.asarray(rgb).shape), rgb_msg.header.frame_id))
        ok("Depth %s shape=%s encoding=%s frame=%s" %
           (DEPTH_TOPIC, str(np.asarray(depth).shape), depth_msg.encoding,
            depth_msg.header.frame_id))
        ok("RGB-depth concurrent pair dt %.3f ms <= %.1f ms "
           "(exact=%s, samples rgb=%d depth=%d)" %
           (dt_s * 1000.0, max_rgb_depth_dt_s * 1000.0,
            bool(sync_diag.get("exact_stamp_match", False)),
            int(sync_diag.get("rgb_samples", 0)),
            int(sync_diag.get("depth_samples", 0))))
    except Exception as exc:
        rgb = None
        depth = None
        rgb_msg = None
        depth_msg = None
        good = fail("RGB-depth synchronized capture: %s" % exc) and good

    try:
        info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=4.0)
        K = np.asarray(info.K, dtype=np.float64).reshape(3, 3)
        ok("CameraInfo %s %dx%d frame=%s fx=%.3f fy=%.3f" %
           (INFO_TOPIC, info.width, info.height, info.header.frame_id, K[0, 0], K[1, 1]))
    except Exception as exc:
        K = None
        good = fail("CameraInfo topic: %s" % exc) and good

    if mask is not None and depth is not None and K is not None:
        if mask.shape != np.asarray(depth).shape[:2]:
            good = fail("mask/depth shape mismatch: %s vs %s" %
                        (str(mask.shape), str(np.asarray(depth).shape[:2]))) and good
        else:
            points = pointcloud_from_depth_mask(depth, mask, K)
            if points is None or len(points) < 20:
                good = fail("segmented RGB-D backprojection produced too few points") and good
            else:
                ok("segmented pointcloud can be built: %d camera-frame points" % len(points))

    # Formal Top-Grasp requires robot/gripper telemetry before recording.
    if Limb is None:
        good = fail("intera_interface.Limb is unavailable") and good
    else:
        try:
            limb = Limb("right")
            joints = limb.joint_angles()
            if not isinstance(joints, dict) or len(joints) < 7:
                raise RuntimeError("expected >=7 joints, got %d" % len(joints or {}))
            ok("Limb telemetry available: %d joints" % len(joints))
        except Exception as exc:
            good = fail("Limb telemetry: %s" % exc) and good

    if Gripper is None:
        good = fail("intera_interface.Gripper is unavailable") and good
    else:
        try:
            gripper = Gripper("right_gripper")
            if not bool(gripper.is_calibrated()):
                raise RuntimeError("right_gripper is not calibrated")
            pos = float(gripper.get_position())
            ok("Gripper calibrated; position=%.4f m" % pos)
        except Exception as exc:
            good = fail("Gripper readiness: %s" % exc) and good

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.4)
    tf_points = {}
    for frame in ("right_hand",
                  "right_gripper_l_finger_tip",
                  "right_gripper_r_finger_tip"):
        try:
            tfm = tf_buffer.lookup_transform(
                "base", frame, rospy.Time(0), rospy.Duration(1.5))
            p = tfm.transform.translation
            tf_points[frame] = [float(p.x), float(p.y), float(p.z)]
            ok("TF base <- %s = [%.4f, %.4f, %.4f]" %
               (frame, p.x, p.y, p.z))
        except Exception as exc:
            good = fail("TF base <- %s: %s" % (frame, exc)) and good

    if all(k in tf_points for k in (
            "right_hand", "right_gripper_l_finger_tip",
            "right_gripper_r_finger_tip")):
        hand = np.asarray(tf_points["right_hand"], dtype=float)
        left = np.asarray(tf_points["right_gripper_l_finger_tip"], dtype=float)
        right = np.asarray(tf_points["right_gripper_r_finger_tip"], dtype=float)
        center = 0.5 * (left + right)
        offset = center - hand
        opening = float(np.linalg.norm(left - right))
        ok("mouth calibration available: opening=%.1f mm offset=[%.4f %.4f %.4f]" %
           (opening * 1000.0, offset[0], offset[1], offset[2]))
    else:
        good = fail("formal Top-Grasp mouth calibration unavailable") and good

    code_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(code_dir, "demo_library", "scene_packages")
    # When this script lives in the code tree this checks the real destination.
    # If copied elsewhere, the import checks above remain authoritative.
    try:
        if os.path.isdir(os.path.join(code_dir, "demo_library")):
            os.makedirs(package_dir, exist_ok=True)
            if not os.access(package_dir, os.W_OK):
                raise RuntimeError("not writable")
            ok("scene-package destination writable: %s" % package_dir)
        else:
            warn("preflight script is not inside the project code directory; destination write check skipped")
    except Exception as exc:
        good = fail("scene-package destination: %s" % exc) and good

    print("=" * 68)
    if good:
        print("READY: formal Top-Grasp hardware/data gates passed; recording may start.")
        return 0
    print("NOT READY: fix the FAIL items before pressing ENTER in record_demo_real.py.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
