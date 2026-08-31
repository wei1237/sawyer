#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict real-robot camera/base alignment for Sawyer + ASC60C.

This file intentionally does NOT modify mt3_alignment.py.  The simulation
module keeps its historical Sawyer head-camera/Gazebo fallbacks; this real
module requires a calibrated TF and never applies those fallbacks or empirical
Gazebo offsets.
"""

import rospy

from mt3_alignment import (
    TrajectoryAligner as _SharedTrajectoryAligner,
    pose_compose,
    pose_inverse,
    quat_conjugate,
    quat_inverse,
    quat_multiply,
    quat_rotate,
)


DEFAULT_TARGET_FRAME = "base"
DEFAULT_CAMERA_FRAME = "ascamera_hp60c_color_0"


def _global_real_param_name(name):
    return "/sawyer_auto_grasp/%s" % str(name).lstrip("~/")


def _param(name, default=None):
    private = "~%s" % str(name).lstrip("~/")
    if rospy.has_param(private):
        return rospy.get_param(private)
    return rospy.get_param(_global_real_param_name(name), default)


def _param_bool(name, default=False):
    value = _param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


class TrajectoryAligner(_SharedTrajectoryAligner):
    """Reuse demo-relative alignment math, replace only camera/base conversion."""

    def __init__(self, head_camera_extrinsics=None):
        # Deliberately do not call the simulation parent __init__ because it seeds
        # HEAD_CAMERA_IN_BASE and Gazebo-specific point-cloud correction params.
        self.target_frame = str(_param("target_frame", DEFAULT_TARGET_FRAME))
        self.camera_frame = str(_param("camera_frame", DEFAULT_CAMERA_FRAME))
        self.use_tf_camera_extrinsics = _param_bool("use_tf_camera_extrinsics", True)
        self.allow_hardcoded_camera_fallback = _param_bool(
            "allow_hardcoded_camera_fallback", False)
        self.strict_camera_frame = _param_bool("strict_camera_frame", True)
        self._tf_source = "tf_required"
        self._tf_buffer = None
        self._tf_listener = None
        self.head_camera_pose = None

        if self.allow_hardcoded_camera_fallback:
            rospy.logerr(
                "[AlignmentReal] allow_hardcoded_camera_fallback=true was requested, "
                "but real mode ignores it for safety.")
            self.allow_hardcoded_camera_fallback = False
        if not self.use_tf_camera_extrinsics:
            rospy.logerr(
                "[AlignmentReal] use_tf_camera_extrinsics=false is unsafe for the "
                "external ASC60C. TF is still required.")

        try:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        except ImportError as exc:
            rospy.logerr("[AlignmentReal] tf2_ros unavailable: %s", exc)

        rospy.loginfo(
            "[AlignmentReal] strict TF: %s <- %s (no hardcoded/Gazebo fallback)",
            self.target_frame, self.camera_frame)

    def _lookup_transform(self, source_frame, timeout_s=2.0):
        if self._tf_buffer is None:
            return None
        try:
            return self._tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(float(timeout_s)),
            )
        except Exception as exc:
            rospy.logerr_throttle(
                3.0,
                "[AlignmentReal] Missing calibrated TF %s <- %s: %s",
                self.target_frame, source_frame, exc)
            return None

    def _lookup_camera_pose_tf(self, camera_frame=None, timeout_s=2.0):
        """Compatibility helper: look up exactly the calibrated real frame."""
        source_frame = str(camera_frame or self.camera_frame)
        if self.strict_camera_frame and source_frame != self.camera_frame:
            rospy.logerr(
                "[AlignmentReal] Refusing camera frame %s; configured calibrated frame is %s",
                source_frame, self.camera_frame)
            return None
        tf_msg = self._lookup_transform(source_frame, timeout_s=timeout_s)
        if tf_msg is None:
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return {
            "position": [float(t.x), float(t.y), float(t.z)],
            "orientation": [float(q.x), float(q.y), float(q.z), float(q.w)],
            "frame": source_frame,
        }

    def _transform_source_frame_to_base(self, pose_in_source):
        source_frame = str(pose_in_source.get("source_frame") or self.camera_frame)
        if self.strict_camera_frame and source_frame != self.camera_frame:
            rospy.logerr(
                "[AlignmentReal] Pose source_frame=%s does not match calibrated camera_frame=%s",
                source_frame, self.camera_frame)
            return None
        tf_msg = self._lookup_transform(source_frame, timeout_s=2.0)
        if tf_msg is None:
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        tf_pos = [float(t.x), float(t.y), float(t.z)]
        tf_ori = [float(q.x), float(q.y), float(q.z), float(q.w)]
        pos_base, ori_base = pose_compose(
            tf_pos,
            tf_ori,
            [float(v) for v in pose_in_source["position"]],
            [float(v) for v in pose_in_source.get(
                "orientation", [0.0, 0.0, 0.0, 1.0])],
        )

        self._tf_source = "calibrated_tf_direct"
        result = {
            "position": pos_base,
            "orientation": ori_base,
            "tf_source": self._tf_source,
            "source_frame": source_frame,
        }
        for key in (
                "estimated_object_size", "estimated_object_height",
                "confidence", "method", "mask_bbox_2d", "mask_center_2d"):
            if pose_in_source.get(key) is not None:
                result[key] = pose_in_source.get(key)

        rospy.loginfo(
            "[AlignmentReal] %s <- %s : [%.4f %.4f %.4f]",
            self.target_frame, source_frame,
            pos_base[0], pos_base[1], pos_base[2])
        return result

    def transform_camera_to_base(self, pose_in_camera):
        """Strict direct TF path; no REP103 guess, no hardcoded extrinsics, no offset."""
        if pose_in_camera is None or pose_in_camera.get("position") is None:
            return None
        return self._transform_source_frame_to_base(pose_in_camera)
