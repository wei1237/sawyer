#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared kinesthetic/Zero-G recorder for real Sawyer demonstrations.

This module NEVER sends arm motion commands. The human moves Sawyer in Zero-G.
It only:
  * samples base -> right_hand TF (default 30 Hz),
  * optionally samples joint positions,
  * commands the physical gripper on keyboard c/o,
  * records explicit grasp/release/terminal-bottleneck events,
  * optionally captures one ASC60C RGB-D-CameraInfo snapshot,
  * converts pose samples to the pose+twist schema used by the MT3 replay code.

Keyboard while recording:
  c : close gripper + record gripper_close
  o : open gripper; after a close this becomes release_open
  t : mark terminal bottleneck (before place/insert local interaction)
  s : stop and keep recording
  x : abort/discard

The first valid pose sample is treated as the grasp bottleneck.
"""

from __future__ import print_function

import hashlib
import json
import math
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty

import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    from intera_interface import Gripper, Limb
except Exception:
    Gripper = None
    Limb = None


DEFAULT_BASE_FRAME = "base"
DEFAULT_EE_FRAME = "right_hand"
DEFAULT_RATE_HZ = 30.0
DEFAULT_RGB_TOPIC = "/ascamera_hp60c/rgb0/image"
DEFAULT_DEPTH_TOPIC = "/ascamera_hp60c/depth0/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/ascamera_hp60c/rgb0/camera_info"
DEFAULT_LEFT_FINGER_TIP_FRAME = "right_gripper_l_finger_tip"
DEFAULT_RIGHT_FINGER_TIP_FRAME = "right_gripper_r_finger_tip"


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)




def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_synchronized_rgbd(rgb_topic, depth_topic, timeout_s=4.0,
                             max_dt_s=0.10, queue_size=60):
    """Capture the nearest-time RGB/depth pair from concurrent subscriptions.

    HP60C registered RGB-D publishes matching header stamps.  Sequential
    wait_for_message(rgb) then wait_for_message(depth) can accidentally select
    frames from different cycles, so formal recording must subscribe to both
    streams at the same time and pair by header.stamp.

    Returns (rgb_msg, depth_msg, abs_dt_s, diagnostics).  An exact timestamp
    match returns immediately.  Otherwise the best pair seen before timeout is
    returned only when it satisfies max_dt_s; if no acceptable pair exists a
    RuntimeError is raised.
    """
    timeout_s = max(0.1, float(timeout_s))
    max_dt_s = max(0.0, float(max_dt_s))
    queue_size = max(4, int(queue_size))

    lock = threading.Lock()
    exact_event = threading.Event()
    rgb_msgs = []
    depth_msgs = []
    best = [None]  # (dt_s, rgb_msg, depth_msg)

    def _stamp_sec(msg):
        return float(msg.header.stamp.to_sec())

    def _consider(rgb_msg, depth_msg):
        dt_s = abs(_stamp_sec(rgb_msg) - _stamp_sec(depth_msg))
        if best[0] is None or dt_s < best[0][0]:
            best[0] = (dt_s, rgb_msg, depth_msg)
        # HP60C commonly publishes bit-identical ROS stamps.  Treat sub-ns
        # difference as exact to avoid floating-point noise.
        if dt_s <= 1.0e-9:
            exact_event.set()

    def _rgb_cb(msg):
        with lock:
            rgb_msgs.append(msg)
            if len(rgb_msgs) > queue_size:
                del rgb_msgs[:-queue_size]
            for depth_msg in depth_msgs:
                _consider(msg, depth_msg)

    def _depth_cb(msg):
        with lock:
            depth_msgs.append(msg)
            if len(depth_msgs) > queue_size:
                del depth_msgs[:-queue_size]
            for rgb_msg in rgb_msgs:
                _consider(rgb_msg, msg)

    rgb_sub = rospy.Subscriber(rgb_topic, Image, _rgb_cb, queue_size=queue_size)
    depth_sub = rospy.Subscriber(depth_topic, Image, _depth_cb, queue_size=queue_size)
    start = time.time()
    try:
        # Prefer an exact pair.  If exact matching is unavailable on another
        # camera, keep collecting until timeout and then accept the best pair
        # only if it satisfies the configured threshold.
        while not rospy.is_shutdown() and (time.time() - start) < timeout_s:
            if exact_event.wait(0.01):
                break
    finally:
        try:
            rgb_sub.unregister()
        except Exception:
            pass
        try:
            depth_sub.unregister()
        except Exception:
            pass

    with lock:
        selected = best[0]
        diagnostics = {
            "rgb_samples": int(len(rgb_msgs)),
            "depth_samples": int(len(depth_msgs)),
            "pairing_mode": "concurrent_header_stamp_nearest",
            "exact_stamp_match": bool(selected is not None and selected[0] <= 1.0e-9),
            "timeout_s": float(timeout_s),
            "max_dt_s": float(max_dt_s),
        }

    if selected is None:
        raise RuntimeError(
            "No RGB/depth messages received together within %.1fs "
            "(rgb=%d depth=%d)" %
            (timeout_s, diagnostics["rgb_samples"], diagnostics["depth_samples"]))

    dt_s, rgb_msg, depth_msg = selected
    diagnostics["selected_dt_s"] = float(dt_s)
    if dt_s > max_dt_s:
        raise RuntimeError(
            "Best concurrent RGB/depth pair dt %.1f ms exceeds %.1f ms "
            "(rgb=%d depth=%d)" %
            (dt_s * 1000.0, max_dt_s * 1000.0,
             diagnostics["rgb_samples"], diagnostics["depth_samples"]))

    return rgb_msg, depth_msg, float(dt_s), diagnostics


def verify_mask_binding(mask_path, source_rgb_path="", metadata_path="",
                        max_age_s=300.0, max_rgb_mtime_delta_s=120.0,
                        allow_mtime_fallback=True):
    """Verify that a LangSAM mask is fresh and bound to its source RGB.

    Preferred mode is SHA256 metadata binding.  When metadata is unavailable,
    formal recording can optionally fall back to a conservative mtime check
    between current_mask.npy and current_rgb.png.
    """
    result = {
        "passed": False,
        "mode": "unverified",
        "reason": "",
        "mask_path": os.path.abspath(os.path.expanduser(mask_path or "")),
        "metadata_path": "",
        "source_rgb_path": "",
        "mask_age_s": None,
        "source_rgb_age_s": None,
        "mask_rgb_mtime_delta_s": None,
        "metadata_source_rgb_sha256": "",
        "source_rgb_sha256": "",
    }
    mask_path = result["mask_path"]
    if not mask_path or not os.path.isfile(mask_path):
        result["reason"] = "mask_file_missing"
        return result

    now = time.time()
    try:
        result["mask_age_s"] = max(0.0, now - os.path.getmtime(mask_path))
    except Exception:
        pass
    if result["mask_age_s"] is not None and result["mask_age_s"] > float(max_age_s):
        result["reason"] = "mask_too_old"
        return result

    mask_dir = os.path.dirname(mask_path)
    base = os.path.splitext(os.path.basename(mask_path))[0]
    candidates = []
    if metadata_path:
        candidates.append(os.path.abspath(os.path.expanduser(metadata_path)))
    candidates.extend([
        os.path.join(mask_dir, base + "_metadata.json"),
        os.path.join(mask_dir, "current_mask_metadata.json"),
        os.path.join(mask_dir, "mask_metadata.json"),
    ])
    seen = set()
    metadata = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                result["metadata_path"] = candidate
                break
            except Exception as exc:
                result["reason"] = "mask_metadata_unreadable:%s" % exc
                return result

    configured_rgb = os.path.abspath(os.path.expanduser(source_rgb_path)) if source_rgb_path else ""
    metadata_rgb = ""
    metadata_hash = ""
    if isinstance(metadata, dict):
        metadata_hash = str(
            metadata.get("source_rgb_sha256") or
            metadata.get("rgb_sha256") or
            metadata.get("source_image_sha256") or "").strip().lower()
        metadata_rgb = str(
            metadata.get("source_rgb_path") or
            metadata.get("rgb_path") or
            metadata.get("source_image_path") or "").strip()
        result["metadata_source_rgb_sha256"] = metadata_hash

    rgb_candidates = []
    if configured_rgb:
        rgb_candidates.append(configured_rgb)
    if metadata_rgb:
        expanded = os.path.abspath(os.path.expanduser(metadata_rgb))
        rgb_candidates.append(expanded)
        rgb_candidates.append(os.path.join(mask_dir, os.path.basename(metadata_rgb.replace('\\', '/'))))
    rgb_candidates.append(os.path.join(mask_dir, "current_rgb.png"))
    rgb_path = ""
    seen = set()
    for candidate in rgb_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            rgb_path = candidate
            break
    result["source_rgb_path"] = rgb_path

    if rgb_path:
        try:
            result["source_rgb_age_s"] = max(0.0, now - os.path.getmtime(rgb_path))
            result["mask_rgb_mtime_delta_s"] = abs(
                os.path.getmtime(mask_path) - os.path.getmtime(rgb_path))
        except Exception:
            pass

    if metadata_hash:
        if not rgb_path:
            result["reason"] = "metadata_hash_present_but_source_rgb_missing"
            return result
        actual_hash = file_sha256(rgb_path).lower()
        result["source_rgb_sha256"] = actual_hash
        if actual_hash != metadata_hash:
            result["reason"] = "source_rgb_sha256_mismatch"
            return result
        result["mode"] = "sha256_metadata"
        result["passed"] = True
        result["reason"] = "sha256_match"
        return result

    if not allow_mtime_fallback:
        result["reason"] = "sha256_metadata_missing_and_fallback_disabled"
        return result
    if not rgb_path:
        result["reason"] = "mtime_fallback_source_rgb_missing"
        return result
    if (result["source_rgb_age_s"] is not None and
            result["source_rgb_age_s"] > float(max_age_s)):
        result["reason"] = "source_rgb_too_old"
        return result
    if (result["mask_rgb_mtime_delta_s"] is None or
            result["mask_rgb_mtime_delta_s"] > float(max_rgb_mtime_delta_s)):
        result["reason"] = "mask_rgb_mtime_delta_exceeded"
        return result

    result["mode"] = "mtime_fallback"
    result["passed"] = True
    result["reason"] = "fresh_mask_and_source_rgb_mtime_consistent"
    return result


def quaternion_normalize(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return (q / n).tolist()


def rotate_by_quat(q, v):
    x, y, z, w = [float(a) for a in q]
    vx, vy, vz = [float(a) for a in v]
    return [
        (1 - 2*y*y - 2*z*z) * vx + (2*x*y - 2*w*z) * vy + (2*x*z + 2*w*y) * vz,
        (2*x*y + 2*w*z) * vx + (1 - 2*x*x - 2*z*z) * vy + (2*y*z - 2*w*x) * vz,
        (2*x*z - 2*w*y) * vx + (2*y*z + 2*w*x) * vy + (1 - 2*x*x - 2*y*y) * vz,
    ]


def quat_delta_to_angular_velocity(q0, q1, dt):
    if dt <= 0:
        return [0.0, 0.0, 0.0]
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)
    dq = [
        q0[0]*q1[3] + q0[3]*q1[0] + q0[1]*q1[2] - q0[2]*q1[1],
        q0[3]*q1[1] - q0[0]*q1[2] + q0[1]*q1[3] + q0[2]*q1[0],
        q0[3]*q1[2] + q0[0]*q1[1] - q0[1]*q1[0] + q0[2]*q1[3],
        q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2],
    ]
    dq = quaternion_normalize(dq)
    if dq[3] < 0:
        dq = [-v for v in dq]
    axis_norm = math.sqrt(dq[0]**2 + dq[1]**2 + dq[2]**2)
    if axis_norm <= 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(axis_norm, max(-1.0, min(1.0, dq[3])))
    axis = [dq[0] / axis_norm, dq[1] / axis_norm, dq[2] / axis_norm]
    return [float(axis[j] * angle / dt) for j in range(3)]


def poses_to_velocities(poses):
    """Match the existing end_effector_pose_twist_gripper_v2 convention."""
    velocities = []
    for i in range(1, len(poses)):
        p_prev = poses[i - 1]
        p_cur = poses[i]
        dt = float(p_cur["timestamp"] - p_prev["timestamp"])
        if dt <= 0:
            continue
        xyz0 = np.asarray(p_prev["position"], dtype=np.float64)
        xyz1 = np.asarray(p_cur["position"], dtype=np.float64)
        linear_world = (xyz1 - xyz0) / dt
        q0 = quaternion_normalize(p_prev["orientation"])
        q1 = quaternion_normalize(p_cur["orientation"])
        q0_conj = [-q0[0], -q0[1], -q0[2], q0[3]]
        linear_ee = rotate_by_quat(q0_conj, linear_world.tolist())
        angular_ee = quat_delta_to_angular_velocity(q0, q1, dt)
        angular_world = rotate_by_quat(q0, angular_ee)
        velocities.append({
            "timestamp": float(p_cur["timestamp"]),
            "dt": dt,
            "position": [float(v) for v in p_cur["position"]],
            "orientation": [float(v) for v in p_cur["orientation"]],
            "linear_ee": [float(v) for v in linear_ee],
            "linear_world": [float(v) for v in linear_world.tolist()],
            "angular_ee": [float(v) for v in angular_ee],
            "angular_world": [float(v) for v in angular_world],
            "gripper_position": p_cur.get("gripper_position"),
            "gripper_state": p_cur.get("gripper_state"),
            "gripper_next": p_cur.get(
                "gripper_next", p_cur.get("gripper_state", p_prev.get("gripper_state"))),
        })
    return velocities


def deduplicate_poses(poses):
    seen = set()
    out = []
    for sample in poses:
        ts = round(float(sample.get("timestamp", 0.0)), 6)
        if ts in seen:
            continue
        seen.add(ts)
        out.append(sample)
    return out


def pose_as_bottleneck(sample):
    if sample is None:
        return None
    p = sample.get("position", [0.0, 0.0, 0.0])
    q = sample.get("orientation", [0.0, 0.0, 0.0, 1.0])
    return {
        "position_m": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
        "orientation_xyzw": {
            "x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
        "timestamp": float(sample.get("timestamp", 0.0)),
    }


def pose_as_relation(sample):
    if sample is None:
        return None
    return {
        "position": [float(v) for v in sample.get("position", [0.0, 0.0, 0.0])],
        "orientation": [float(v) for v in sample.get("orientation", [0.0, 0.0, 0.0, 1.0])],
        "timestamp": float(sample.get("timestamp", 0.0)),
        "frame": "base",
        "gripper_position": sample.get("gripper_position"),
        "gripper_state": sample.get("gripper_state"),
        "gripper_next": sample.get("gripper_next"),
    }


def event_index(events, names):
    names = set(names)
    found = None
    for e in events:
        if str(e.get("name", "")) in names:
            found = int(e.get("sample_index", 0))
    return found


def build_grasp_segment(full_trajectory, terminal_start_idx=None,
                        pre_samples=75, post_samples=30):
    poses = full_trajectory.get("poses") or []
    velocities = full_trajectory.get("velocities") or []
    events = full_trajectory.get("events") or []
    close_idx = event_index(events, ["gripper_close", "grasp_close", "top_grasp_close"])
    if close_idx is None or not poses:
        return None, None, None
    first_bn_idx = event_index(events, ["grasp_bottleneck"])
    if first_bn_idx is None:
        first_bn_idx = max(0, int(close_idx) - int(pre_samples))
    start_idx = max(0, min(int(first_bn_idx), int(close_idx)))
    end_idx = min(len(poses), int(close_idx) + int(post_samples) + 1)
    if terminal_start_idx is not None:
        end_idx = min(end_idx, max(int(close_idx) + 1, int(terminal_start_idx)))
    segment_poses = [dict(x) for x in poses[start_idx:end_idx]]
    # velocities correspond approximately to pose indices 1..N-1; slicing this way
    # matches the legacy recorders closely enough for the pose-first replay path.
    segment_velocities = [dict(x) for x in velocities[start_idx:end_idx]] if velocities else []
    local_close = int(close_idx) - start_idx
    for i, sample in enumerate(segment_poses):
        sample["gripper_state"] = 1 if i >= local_close else 0
        sample["gripper_next"] = 1 if i == local_close else 0
    base_position = list(segment_poses[0].get("position", [])) if segment_poses else []
    traj = {
        "format": "mt3_anchor_grasp_trajectory_v1",
        "frame": "base",
        "pose_frame": "base",
        "sample_rate_hz": full_trajectory.get("sample_rate_hz"),
        "source_rollout_format": full_trajectory.get("format"),
        "source_start_index": int(start_idx),
        "source_close_index": int(close_idx),
        "close_event_source": "explicit_event",
        "source_end_index": int(end_idx),
        "close_index": int(local_close),
        "base_index": 0,
        "base_position": [float(v) for v in base_position[:3]],
        "poses": segment_poses,
        "velocities": segment_velocities,
        "success": bool(full_trajectory.get("success", True)),
    }
    return traj, pose_as_relation(poses[start_idx]), pose_as_relation(poses[close_idx])


def build_terminal_segment(full_trajectory, kind="place",
                           pre_samples=75, post_samples=90):
    poses = full_trajectory.get("poses") or []
    velocities = full_trajectory.get("velocities") or []
    events = full_trajectory.get("events") or []
    open_idx = event_index(events, ["release_open", "place_release_open", "gripper_release_open"])
    if open_idx is None or not poses:
        return None, None, None
    marked = event_index(events, ["terminal_bottleneck"])
    if marked is None:
        start_idx = max(0, int(open_idx) - int(pre_samples))
        source = "fallback_pre_samples"
    else:
        start_idx = max(0, min(int(marked), int(open_idx)))
        source = "explicit_terminal_bottleneck"
    end_idx = min(len(poses), int(open_idx) + int(post_samples) + 1)
    segment_poses = [dict(x) for x in poses[start_idx:end_idx]]
    segment_velocities = [dict(x) for x in velocities[start_idx:end_idx]] if velocities else []
    fmt = (
        "mt3_insertion_terminal_trajectory_v1"
        if kind == "insertion"
        else "mt3_anchor_place_terminal_trajectory_v1"
    )
    traj = {
        "format": fmt,
        "frame": "base",
        "pose_frame": "base",
        "sample_rate_hz": full_trajectory.get("sample_rate_hz"),
        "source_rollout_format": full_trajectory.get("format"),
        "source_start_index": int(start_idx),
        "source_release_index": int(open_idx),
        "release_event_source": "explicit_event",
        "bottleneck_event_source": source,
        "source_end_index": int(end_idx),
        "release_index": int(open_idx) - start_idx,
        "poses": segment_poses,
        "velocities": segment_velocities,
        "success": bool(full_trajectory.get("success", True)),
    }
    return traj, pose_as_relation(poses[start_idx]), pose_as_relation(poses[open_idx])


class TerminalCbreak(object):
    def __init__(self):
        self.fd = None
        self.old = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self):
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
        return None


class KinestheticRecorder(object):
    def __init__(self, task_type, session_dir):
        self.task_type = str(task_type)
        self.session_dir = os.path.abspath(os.path.expanduser(session_dir))
        ensure_dir(self.session_dir)

        self.base_frame = rospy.get_param("~base_frame", DEFAULT_BASE_FRAME)
        self.ee_frame = rospy.get_param("~ee_frame", DEFAULT_EE_FRAME)
        self.rate_hz = float(rospy.get_param("~record_rate_hz", DEFAULT_RATE_HZ))
        self.capture_rgbd = bool(rospy.get_param("~capture_rgbd", True))
        self.rgb_topic = rospy.get_param("~rgb_topic", DEFAULT_RGB_TOPIC)
        self.depth_topic = rospy.get_param("~depth_topic", DEFAULT_DEPTH_TOPIC)
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC)
        self.mask_path = os.path.expanduser(rospy.get_param("~mask_path", ""))
        self.require_scene_snapshot = bool(rospy.get_param(
            "~require_scene_snapshot", self.task_type == "top_grasp"))
        self.require_snapshot_mask = bool(rospy.get_param(
            "~require_snapshot_mask", self.task_type == "top_grasp"))
        self.synchronous_bottleneck_snapshot = bool(rospy.get_param(
            "~synchronous_bottleneck_snapshot", True))
        self.left_finger_tip_frame = rospy.get_param(
            "~left_finger_tip_frame", DEFAULT_LEFT_FINGER_TIP_FRAME)
        self.right_finger_tip_frame = rospy.get_param(
            "~right_finger_tip_frame", DEFAULT_RIGHT_FINGER_TIP_FRAME)

        # Formal Top-Grasp integrity gates.  Other task recorders keep their
        # existing behavior unless ~formal_recording is explicitly enabled.
        self.formal_mode = bool(rospy.get_param(
            "~formal_recording", self.task_type == "top_grasp"))
        self.require_gripper = bool(rospy.get_param(
            "~require_gripper", self.formal_mode))
        self.require_limb_telemetry = bool(rospy.get_param(
            "~require_limb_telemetry", self.formal_mode))
        self.require_mouth_calibration = bool(rospy.get_param(
            "~require_mouth_calibration",
            self.formal_mode and self.task_type == "top_grasp"))
        self.min_post_close_samples = int(rospy.get_param(
            "~min_post_close_samples", 15))
        self.min_post_close_lift_m = float(rospy.get_param(
            "~min_post_close_lift_m", 0.040))
        self.joint_coverage_warn_threshold = float(rospy.get_param(
            "~joint_coverage_warn_threshold", 0.95))
        self.max_rgb_depth_dt_s = float(rospy.get_param(
            "~max_rgb_depth_dt_s", 0.10))
        self.mask_metadata_path = os.path.expanduser(str(rospy.get_param(
            "~mask_metadata_path", "")))
        default_mask_rgb = (
            os.path.join(os.path.dirname(self.mask_path), "current_rgb.png")
            if self.mask_path else "")
        self.mask_source_rgb_path = os.path.expanduser(str(rospy.get_param(
            "~mask_source_rgb_path", default_mask_rgb)))
        self.max_mask_age_s = float(rospy.get_param("~max_mask_age_s", 300.0))
        self.max_mask_rgb_mtime_delta_s = float(rospy.get_param(
            "~max_mask_rgb_mtime_delta_s", 120.0))
        self.allow_mask_mtime_fallback = bool(rospy.get_param(
            "~allow_mask_mtime_fallback", True))

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.gripper = None
        self.limb = None
        self.gripper_state = None
        self.poses = []
        self.events = []
        self.aborted = False
        self.snapshot = None
        self.bottleneck_mouth_state = None
        self.close_mouth_state = None
        self.mask_binding = None
        self.integrity_metrics = {}
        self._snapshot_thread = None

        if Limb is not None:
            try:
                self.limb = Limb("right")
            except Exception as exc:
                rospy.logwarn("Limb telemetry unavailable: %s", exc)
                self.limb = None
        if Gripper is not None:
            try:
                self.gripper = Gripper("right_gripper")
                if not self.gripper.is_calibrated():
                    rospy.logwarn(
                        "right_gripper is not calibrated; c/o commands may fail. "
                        "Calibrate it before formal demo recording.")
            except Exception as exc:
                rospy.logwarn("Gripper interface unavailable: %s", exc)
                self.gripper = None

    def get_ee_pose(self):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(0.2))
            p = tfm.transform.translation
            q = tfm.transform.rotation
            sample = {
                "position": [float(p.x), float(p.y), float(p.z)],
                "orientation": quaternion_normalize([q.x, q.y, q.z, q.w]),
                "timestamp": float(tfm.header.stamp.to_sec() or rospy.get_time()),
                "gripper_position": self.get_gripper_position(),
                "gripper_state": self.gripper_state,
                "gripper_next": self.gripper_state,
            }
            if self.limb is not None:
                try:
                    sample["joint_positions"] = {
                        str(k): float(v) for k, v in self.limb.joint_angles().items()
                    }
                except Exception:
                    pass
            return sample
        except Exception:
            return None

    def _lookup_frame_point(self, frame, timeout=0.25):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame, frame, rospy.Time(0), rospy.Duration(timeout))
            p = tfm.transform.translation
            return [float(p.x), float(p.y), float(p.z)]
        except Exception:
            return None

    def get_gripper_mouth_state(self):
        """Measure open-gripper mouth center relative to right_hand at the bottleneck.

        In formal Top-Grasp mode this calibration is required.  Other task
        recorders may still use right_hand-only fallback when configured.
        """
        hand_sample = self.get_ee_pose()
        if hand_sample is None:
            return {"available": False, "reason": "right_hand_tf_unavailable"}
        hand = [float(v) for v in hand_sample["position"]]
        left = self._lookup_frame_point(self.left_finger_tip_frame)
        right = self._lookup_frame_point(self.right_finger_tip_frame)
        if left is None or right is None:
            return {
                "available": False,
                "reason": "finger_tip_tf_unavailable",
                "hand_xyz": hand,
                "left_frame": self.left_finger_tip_frame,
                "right_frame": self.right_finger_tip_frame,
            }
        center = [0.5 * (left[i] + right[i]) for i in range(3)]
        offset = [center[i] - hand[i] for i in range(3)]
        opening = math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))
        return {
            "available": True,
            "hand_xyz": hand,
            "left_xyz": left,
            "right_xyz": right,
            "mouth_center_xyz": center,
            "mouth_offset_xyz": offset,
            "used_mouth_offset_xy": [float(offset[0]), float(offset[1])],
            "mouth_opening_m": float(opening),
            "left_frame": self.left_finger_tip_frame,
            "right_frame": self.right_finger_tip_frame,
        }

    def get_gripper_position(self):
        if self.gripper is None:
            return None
        try:
            return float(self.gripper.get_position())
        except Exception:
            return None

    def add_event(self, name):
        idx = max(0, len(self.poses) - 1)
        event = {
            "name": str(name),
            "sample_index": int(idx),
            "timestamp": float(rospy.get_time()),
        }
        self.events.append(event)
        rospy.loginfo("EVENT %-22s sample=%d", name, idx)

    def _gripper_calibrated(self):
        if self.gripper is None:
            return False
        try:
            return bool(self.gripper.is_calibrated())
        except Exception:
            return False

    def close_gripper(self):
        existing = [e for e in self.events if str(e.get("name", "")) == "gripper_close"]
        if existing:
            if self.formal_mode:
                self.aborted = True
                raise RuntimeError(
                    "FORMAL DEMO INVALID: gripper_close may occur exactly once; "
                    "a second 'c' was received")
            rospy.logwarn("Duplicate 'c' ignored; keeping the first gripper_close event.")
            return False
        if self.gripper is None:
            if self.require_gripper:
                self.aborted = True
                raise RuntimeError(
                    "FORMAL DEMO INVALID: Gripper interface unavailable; close event not recorded")
            rospy.logwarn("No gripper interface; recording close event only (non-formal mode).")
        else:
            if self.require_gripper and not self._gripper_calibrated():
                self.aborted = True
                raise RuntimeError(
                    "FORMAL DEMO INVALID: right_gripper is not calibrated")
            self.close_mouth_state = self.get_gripper_mouth_state()
            if self.require_mouth_calibration and not self.close_mouth_state.get("available", False):
                self.aborted = True
                raise RuntimeError(
                    "FORMAL DEMO INVALID: close-time mouth calibration unavailable: %s" %
                    self.close_mouth_state.get("reason", "unknown"))
            if self.close_mouth_state.get("available", False):
                mouth = self.close_mouth_state.get("mouth_center_xyz", [0.0, 0.0, 0.0])
                hand = self.close_mouth_state.get("hand_xyz", [0.0, 0.0, 0.0])
                rospy.loginfo(
                    "Close-time mouth TF: hand=[%.4f %.4f %.4f] mouth=[%.4f %.4f %.4f] "
                    "offset=[%.1f %.1f %.1f]mm opening=%.1fmm",
                    hand[0], hand[1], hand[2],
                    mouth[0], mouth[1], mouth[2],
                    float(self.close_mouth_state["mouth_offset_xyz"][0]) * 1000.0,
                    float(self.close_mouth_state["mouth_offset_xyz"][1]) * 1000.0,
                    float(self.close_mouth_state["mouth_offset_xyz"][2]) * 1000.0,
                    float(self.close_mouth_state.get("mouth_opening_m", 0.0)) * 1000.0)
            try:
                command_result = self.gripper.close()
                if command_result is False:
                    raise RuntimeError("Gripper.close() returned False")
            except Exception as exc:
                self.aborted = True
                raise RuntimeError(
                    "FORMAL DEMO INVALID: gripper close command failed: %s" % exc)
        # Only stamp the semantic event after a successful command path.
        self.gripper_state = 1
        self.add_event("gripper_close")
        return True

    def open_gripper(self):
        had_close = event_index(self.events, ["gripper_close"]) is not None
        if self.gripper is None:
            rospy.logwarn("No gripper interface; recording open event only.")
        else:
            try:
                self.gripper.open()
            except Exception as exc:
                rospy.logwarn("Gripper open failed: %s", exc)
        self.gripper_state = 0
        self.add_event("release_open" if had_close else "gripper_open")

    def mark_terminal_bottleneck(self):
        self.add_event("terminal_bottleneck")

    def _capture_snapshot_worker(self):
        if not self.capture_rgbd:
            return
        try:
            rgb_msg, depth_msg, rgb_depth_dt_s, sync_diag = capture_synchronized_rgbd(
                self.rgb_topic, self.depth_topic, timeout_s=4.0,
                max_dt_s=self.max_rgb_depth_dt_s)
            info_msg = rospy.wait_for_message(
                self.camera_info_topic, CameraInfo, timeout=4.0)
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            if np.asarray(rgb).shape[:2] != np.asarray(depth).shape[:2]:
                raise RuntimeError(
                    "ASC60C RGB/depth shape mismatch: %s vs %s" %
                    (str(np.asarray(rgb).shape[:2]), str(np.asarray(depth).shape[:2])))

            rgb_stamp = float(rgb_msg.header.stamp.to_sec())
            depth_stamp = float(depth_msg.header.stamp.to_sec())
            # capture_synchronized_rgbd already enforces the configured threshold.
            sync_passed = rgb_depth_dt_s <= float(self.max_rgb_depth_dt_s)

            snap_dir = os.path.join(self.session_dir, "scene_snapshot")
            ensure_dir(snap_dir)
            rgb_path = os.path.join(snap_dir, "rgb.png")
            if cv2 is None:
                raise RuntimeError("OpenCV unavailable; cannot save formal RGB snapshot")
            if not cv2.imwrite(rgb_path, rgb):
                raise RuntimeError("Failed to write RGB snapshot: %s" % rgb_path)
            np.save(os.path.join(snap_dir, "depth.npy"), np.asarray(depth))

            meta = {
                "rgb_topic": self.rgb_topic,
                "depth_topic": self.depth_topic,
                "camera_info_topic": self.camera_info_topic,
                "rgb_frame_id": rgb_msg.header.frame_id,
                "depth_frame_id": depth_msg.header.frame_id,
                "camera_info_frame_id": info_msg.header.frame_id,
                "width": int(info_msg.width),
                "height": int(info_msg.height),
                "K": [float(v) for v in info_msg.K],
                "D": [float(v) for v in info_msg.D],
                "distortion_model": info_msg.distortion_model,
                "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rgb_stamp": rgb_stamp,
                "depth_stamp": depth_stamp,
                "camera_info_stamp": float(info_msg.header.stamp.to_sec()),
                "rgb_depth_dt_s": float(rgb_depth_dt_s),
                "rgb_depth_sync_threshold_s": float(self.max_rgb_depth_dt_s),
                "rgb_depth_sync_passed": bool(sync_passed),
                "rgb_depth_pairing_mode": sync_diag.get("pairing_mode"),
                "rgb_depth_exact_stamp_match": bool(sync_diag.get("exact_stamp_match", False)),
                "rgb_pairing_samples": int(sync_diag.get("rgb_samples", 0)),
                "depth_pairing_samples": int(sync_diag.get("depth_samples", 0)),
                "depth_encoding": str(depth_msg.encoding),
                "snapshot_rgb_sha256": file_sha256(rgb_path),
            }

            if self.mask_path and os.path.isfile(self.mask_path):
                mask_arr = np.load(self.mask_path)
                if np.asarray(mask_arr).shape != np.asarray(depth).shape[:2]:
                    raise RuntimeError(
                        "LangSAM mask/depth shape mismatch: %s vs %s" %
                        (str(np.asarray(mask_arr).shape), str(np.asarray(depth).shape[:2])))
                if int(np.count_nonzero(mask_arr)) < 20:
                    raise RuntimeError("LangSAM mask has too few pixels")
                binding = verify_mask_binding(
                    self.mask_path,
                    source_rgb_path=self.mask_source_rgb_path,
                    metadata_path=self.mask_metadata_path,
                    max_age_s=self.max_mask_age_s,
                    max_rgb_mtime_delta_s=self.max_mask_rgb_mtime_delta_s,
                    allow_mtime_fallback=self.allow_mask_mtime_fallback)
                self.mask_binding = binding
                if self.formal_mode and not binding.get("passed", False):
                    raise RuntimeError(
                        "FORMAL DEMO INVALID: mask source binding failed: %s" %
                        binding.get("reason", "unknown"))
                shutil.copy2(self.mask_path, os.path.join(snap_dir, "mask.npy"))
                meta["mask_source"] = self.mask_path
                meta["mask_pixels"] = int(np.count_nonzero(mask_arr))
                meta["mask_binding"] = binding
            elif self.require_snapshot_mask:
                raise RuntimeError(
                    "Formal real demo requires a current LangSAM mask; set ~mask_path")

            with open(os.path.join(snap_dir, "camera_info.json"), "w") as f:
                json.dump(meta, f, indent=2)
            self.snapshot = {
                "directory": snap_dir,
                "rgb": rgb_path,
                "depth": os.path.join(snap_dir, "depth.npy"),
                "camera_info": os.path.join(snap_dir, "camera_info.json"),
                "mask": os.path.join(snap_dir, "mask.npy")
                        if os.path.isfile(os.path.join(snap_dir, "mask.npy")) else None,
                "frame_id": info_msg.header.frame_id,
                "rgb_depth_dt_s": float(rgb_depth_dt_s),
                "rgb_depth_sync_threshold_s": float(self.max_rgb_depth_dt_s),
                "rgb_depth_sync_passed": bool(sync_passed),
                "rgb_depth_pairing_mode": sync_diag.get("pairing_mode"),
                "rgb_depth_exact_stamp_match": bool(sync_diag.get("exact_stamp_match", False)),
                "rgb_pairing_samples": int(sync_diag.get("rgb_samples", 0)),
                "depth_pairing_samples": int(sync_diag.get("depth_samples", 0)),
                "mask_binding": self.mask_binding,
            }
            rospy.loginfo(
                "ASC60C snapshot saved: %s (RGB-depth dt=%.1f ms exact=%s, mask_binding=%s)",
                snap_dir, rgb_depth_dt_s * 1000.0,
                bool(sync_diag.get("exact_stamp_match", False)),
                (self.mask_binding or {}).get("mode", "n/a"))
        except Exception as exc:
            self.snapshot = None
            if self.require_scene_snapshot:
                raise
            rospy.logwarn("ASC60C snapshot skipped/failed: %s", exc)

    def validate_before_record(self):
        if self.require_snapshot_mask:
            if not self.mask_path or not os.path.isfile(self.mask_path):
                raise RuntimeError(
                    "Formal real demo requires current LangSAM mask file: %s" %
                    (self.mask_path or "<unset>"))
            try:
                mask = np.load(self.mask_path)
                if mask.ndim != 2 or int(np.count_nonzero(mask)) < 20:
                    raise RuntimeError("mask invalid or too small")
            except Exception as exc:
                raise RuntimeError("Cannot use LangSAM mask %s: %s" %
                                   (self.mask_path, exc))
            binding = verify_mask_binding(
                self.mask_path,
                source_rgb_path=self.mask_source_rgb_path,
                metadata_path=self.mask_metadata_path,
                max_age_s=self.max_mask_age_s,
                max_rgb_mtime_delta_s=self.max_mask_rgb_mtime_delta_s,
                allow_mtime_fallback=self.allow_mask_mtime_fallback)
            self.mask_binding = binding
            if self.formal_mode and not binding.get("passed", False):
                raise RuntimeError(
                    "Formal mask freshness/source binding failed: %s" %
                    binding.get("reason", "unknown"))
            rospy.loginfo(
                "Mask binding ready: mode=%s reason=%s",
                binding.get("mode", ""), binding.get("reason", ""))

        if self.require_limb_telemetry:
            if self.limb is None:
                raise RuntimeError(
                    "Formal real demo requires Limb telemetry, but Limb('right') is unavailable")
            try:
                joints = self.limb.joint_angles()
                if not isinstance(joints, dict) or len(joints) < 7:
                    raise RuntimeError("expected 7 Sawyer arm joints, got %d" % len(joints or {}))
                rospy.loginfo("Limb telemetry ready: %d joints", len(joints))
            except Exception as exc:
                raise RuntimeError("Formal Limb telemetry check failed: %s" % exc)

        if self.require_gripper:
            if self.gripper is None:
                raise RuntimeError(
                    "Formal real demo requires right_gripper interface")
            if not self._gripper_calibrated():
                raise RuntimeError(
                    "Formal real demo requires a calibrated right_gripper")
            try:
                pos = self.get_gripper_position()
                if pos is None:
                    raise RuntimeError("gripper position telemetry unavailable")
                rospy.loginfo("Gripper ready: calibrated, position=%.4f m", pos)
            except Exception as exc:
                raise RuntimeError("Formal gripper telemetry check failed: %s" % exc)

        rospy.loginfo("Waiting for TF %s <- %s ...", self.base_frame, self.ee_frame)
        deadline = time.time() + 5.0
        sample = None
        while time.time() < deadline and not rospy.is_shutdown():
            sample = self.get_ee_pose()
            if sample is not None:
                rospy.loginfo("TF ready. Current right_hand xyz = [%.3f, %.3f, %.3f]",
                              sample["position"][0], sample["position"][1], sample["position"][2])
                break
            rospy.sleep(0.1)
        if sample is None:
            raise RuntimeError("TF %s <- %s is not available" %
                               (self.base_frame, self.ee_frame))

        if self.require_mouth_calibration:
            mouth = self.get_gripper_mouth_state()
            if not mouth.get("available", False):
                raise RuntimeError(
                    "Formal Top-Grasp requires both fingertip TFs and mouth calibration: %s" %
                    mouth.get("reason", "unavailable"))
            rospy.loginfo(
                "Mouth TF ready: opening=%.1f mm offset=[%.4f %.4f %.4f]",
                float(mouth.get("mouth_opening_m", 0.0)) * 1000.0,
                float(mouth["mouth_offset_xyz"][0]),
                float(mouth["mouth_offset_xyz"][1]),
                float(mouth["mouth_offset_xyz"][2]))
        return True

    def joint_coverage_stats(self):
        total = len(self.poses)
        covered = 0
        for sample in self.poses:
            joints = sample.get("joint_positions")
            if isinstance(joints, dict) and len(joints) >= 7:
                covered += 1
        coverage = float(covered) / float(total) if total else 0.0
        return {
            "total_pose_samples": int(total),
            "samples_with_joint_positions": int(covered),
            "coverage": float(coverage),
            "warn_threshold": float(self.joint_coverage_warn_threshold),
            "passed_threshold": bool(coverage >= self.joint_coverage_warn_threshold),
        }

    def record(self):
        self.validate_before_record()
        print("\n============================================================")
        print("REAL SAWYER KINESTHETIC / ZERO-G DEMO")
        print("task:", self.task_type)
        print("This program sends NO arm motion commands.")
        print("Use Sawyer Zero-G/cuff to move the arm manually.")
        print("")
        print("Before pressing ENTER:")
        print("  1) Put the gripper at the grasp bottleneck/start pose.")
        print("  2) Make sure the robot is in Zero-G/manual guidance mode.")
        print("")
        print("During recording:")
        print("  c = close gripper / grasp event")
        print("  o = open gripper / release event")
        print("  t = mark terminal bottleneck before place/insert")
        print("  s = stop and save")
        print("  x = abort and discard")
        print("============================================================")
        input("Press ENTER to lock bottleneck and start recording... ")

        # Formal mode captures the demonstration scene BEFORE the human starts moving,
        # so RGB-D/mask/pointcloud describe the same bottleneck scene as pose[0].
        if self.capture_rgbd and self.synchronous_bottleneck_snapshot:
            rospy.loginfo("Capturing bottleneck RGB-D + mask snapshot; keep the arm/object still...")
            self._capture_snapshot_worker()

        first = self.get_ee_pose()
        if first is None:
            raise RuntimeError("No right_hand TF at recording start")
        if self.gripper_state is None:
            # Most demonstrations begin open; this only stamps metadata and does
            # not move the gripper. Press 'o' if an explicit open command is needed.
            self.gripper_state = 0
            first["gripper_state"] = 0
            first["gripper_next"] = 0
        self.poses = [first]
        self.events = []
        self.bottleneck_mouth_state = self.get_gripper_mouth_state()
        if self.require_mouth_calibration and not self.bottleneck_mouth_state.get("available", False):
            raise RuntimeError(
                "FORMAL DEMO INVALID: bottleneck mouth calibration unavailable: %s" %
                self.bottleneck_mouth_state.get("reason", "unknown"))
        self.add_event("grasp_bottleneck")

        if self.capture_rgbd and not self.synchronous_bottleneck_snapshot:
            self._snapshot_thread = threading.Thread(target=self._capture_snapshot_worker)
            self._snapshot_thread.daemon = True
            self._snapshot_thread.start()

        rospy.loginfo("Bottleneck locked. Begin the manual interaction trajectory now.")
        rate = rospy.Rate(self.rate_hz)
        start_wall = time.time()
        last_status = start_wall
        with TerminalCbreak() as kb:
            while not rospy.is_shutdown():
                sample = self.get_ee_pose()
                if sample is not None:
                    self.poses.append(sample)
                key = kb.read_key()
                if key:
                    key = key.lower()
                    if key == "c":
                        self.close_gripper()
                    elif key == "o":
                        self.open_gripper()
                    elif key == "t":
                        self.mark_terminal_bottleneck()
                    elif key == "s":
                        rospy.loginfo("Stop requested; keeping demo.")
                        break
                    elif key == "x":
                        rospy.logwarn("Abort requested; discarding demo.")
                        self.aborted = True
                        break
                if time.time() - last_status >= 2.0:
                    rospy.loginfo("recording... %.1fs, poses=%d, events=%s",
                                  time.time() - start_wall, len(self.poses),
                                  [e["name"] for e in self.events])
                    last_status = time.time()
                rate.sleep()

        if self._snapshot_thread is not None:
            self._snapshot_thread.join(timeout=0.5)

        # Keep original sample indexing so event sample_index values remain exact.
        # Zero/non-positive dt samples are simply skipped when velocities are built.
        if self.aborted:
            return None
        if len(self.poses) < 10:
            raise RuntimeError("Too few pose samples: %d" % len(self.poses))

        close_events = [
            e for e in self.events if str(e.get("name", "")) == "gripper_close"]
        if len(close_events) != 1:
            raise RuntimeError(
                "FORMAL DEMO INVALID: expected exactly one gripper_close event, got %d" %
                len(close_events))
        close_idx = int(close_events[0].get("sample_index", -1))
        if close_idx < 0 or close_idx >= len(self.poses):
            raise RuntimeError("Recorded gripper_close sample index is invalid: %d" % close_idx)

        release_idx = None
        if self.task_type in ("anchor_pick_place", "cylinder_insert_socket"):
            release_idx = event_index(self.events, ["release_open"])
            if release_idx is None or release_idx <= close_idx:
                raise RuntimeError("Relation task requires an 'o' release after the 'c' grasp.")

        post_close_samples = max(0, len(self.poses) - int(close_idx) - 1)
        close_z = float(self.poses[close_idx]["position"][2])
        post_z = [float(p["position"][2]) for p in self.poses[close_idx + 1:]]
        post_close_lift_m = (max(post_z) - close_z) if post_z else 0.0
        if self.formal_mode and self.task_type == "top_grasp":
            if post_close_samples < int(self.min_post_close_samples):
                raise RuntimeError(
                    "FORMAL DEMO INVALID: only %d post-close samples; require >= %d" %
                    (post_close_samples, self.min_post_close_samples))
            if post_close_lift_m < float(self.min_post_close_lift_m):
                raise RuntimeError(
                    "FORMAL DEMO INVALID: post-close lift %.1f mm; require >= %.1f mm" %
                    (post_close_lift_m * 1000.0, self.min_post_close_lift_m * 1000.0))
            if not (self.bottleneck_mouth_state or {}).get("available", False):
                raise RuntimeError(
                    "FORMAL DEMO INVALID: mouth calibration unavailable")

        velocities = poses_to_velocities(self.poses)
        duration_s = max(0.0, float(self.poses[-1]["timestamp"]) -
                         float(self.poses[0]["timestamp"]))
        effective_rate_hz = (
            float(len(self.poses) - 1) / duration_s
            if duration_s > 1e-6 and len(self.poses) > 1 else 0.0)
        joint_stats = self.joint_coverage_stats()
        if not joint_stats["passed_threshold"]:
            rospy.logwarn(
                "FORMAL DATA WARNING: joint telemetry coverage %.1f%% < %.1f%%; "
                "EE trajectory remains valid but re-recording is recommended.",
                joint_stats["coverage"] * 100.0,
                joint_stats["warn_threshold"] * 100.0)

        self.integrity_metrics = {
            "duration_s": float(duration_s),
            "effective_rate_hz": float(effective_rate_hz),
            "close_event_count": int(len(close_events)),
            "close_index": int(close_idx),
            "pre_close_samples": int(close_idx + 1),
            "post_close_samples": int(post_close_samples),
            "post_close_lift_m": float(post_close_lift_m),
            "min_post_close_samples": int(self.min_post_close_samples),
            "min_post_close_lift_m": float(self.min_post_close_lift_m),
            "joint_telemetry": joint_stats,
            "mouth_calibration_available": bool(
                (self.bottleneck_mouth_state or {}).get("available", False)),
            "rgb_depth_dt_s": (self.snapshot or {}).get("rgb_depth_dt_s"),
            "rgb_depth_sync_threshold_s": (self.snapshot or {}).get(
                "rgb_depth_sync_threshold_s", self.max_rgb_depth_dt_s),
            "rgb_depth_sync_passed": bool(
                (self.snapshot or {}).get("rgb_depth_sync_passed", False)),
            "mask_binding": self.mask_binding,
        }

        trajectory = {
            "format": "mt3_real_kinesthetic_rollout_v1",
            "frame": "base",
            "pose_frame": "base",
            "ee_frame": self.ee_frame,
            "sample_rate_hz": float(self.rate_hz),
            "effective_rate_hz": float(effective_rate_hz),
            "duration_s": float(duration_s),
            "num_waypoints": len(self.poses),
            "poses": self.poses,
            "velocities": velocities,
            "events": self.events,
            "close_index": int(close_idx),
            "close_event_source": "explicit_event",
            "base_index": 0,
            "base_position": [float(v) for v in self.poses[0].get("position", [])[:3]],
            "bottleneck_mouth_state": self.bottleneck_mouth_state,
            "close_mouth_state": self.close_mouth_state,
            "scene_snapshot": self.snapshot,
            "integrity_metrics": self.integrity_metrics,
            "formal_recording": bool(self.formal_mode),
            "success": True,
            "recording_mode": "human_kinesthetic_zero_g",
            "arm_motion_commanded_by_recorder": False,
            "gripper_convention": "gripper_state/gripper_next: 1=close, 0=open",
        }
        if release_idx is not None:
            trajectory["release_index"] = int(release_idx)
            trajectory["release_event_source"] = "explicit_event"
        return trajectory

    def save_rollout(self, trajectory, path):
        ensure_dir(os.path.dirname(path))
        payload = dict(trajectory)
        payload["scene_snapshot"] = self.snapshot
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        rospy.loginfo("Kinesthetic rollout saved: %s", path)
        return path
