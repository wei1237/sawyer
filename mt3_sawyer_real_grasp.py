#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Real Sawyer top-grasp executor aligned with the simulation MT3 replay path.

Default mode is dry-run.  Real motion requires ``--execute``.  The robot state
check is deliberately conservative: this script never resets or auto-recovers a
real Sawyer.

Perception contract, usually published by
``real_perception_patch/mt3_real_object_param_bridge.py``:

    /mt3/current_object_x
    /mt3/current_object_y
    /mt3/current_object_z
    /mt3/current_object_size_m
    /mt3/current_object_top_z_base
    /mt3/current_object_z_semantics

Execution method mirrors the simulation top-grasp path:

    demo trajectory
      -> object-relative local replay
      -> height-aware close mapping
      -> close-tail mouth-anchor XY correction
      -> close gripper at recorded event
      -> lift verification/logging
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import time
import traceback

import actionlib

ROS_WS_PYTHON = "/home/wei/ros_ws/devel/lib/python3/dist-packages"
if ROS_WS_PYTHON not in sys.path:
    sys.path.insert(0, ROS_WS_PYTHON)

import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATCH_DIR = os.path.join(CODE_DIR, "real_perception_patch")
if REAL_PATCH_DIR not in sys.path:
    sys.path.insert(0, REAL_PATCH_DIR)

DEFAULT_DEMO_PATHS = [
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/real/recorded/"
        "cube_green_top_grasp_real.json"),
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/recorded/"
        "cube_green_top_grasp_real.json"),
]

DEFAULT_LOG_DIR = "/mnt/hgfs2/learning_thousand_tasks_logs"

SIM_SAFE_START_JOINTS = {
    "right_j0": 0.0,
    "right_j1": -0.8,
    "right_j2": 0.0,
    "right_j3": 1.8,
    "right_j4": 0.0,
    "right_j5": 0.0,
    "right_j6": 0.0,
}
SAWYER_JOINT_ORDER = [
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
]


def _as_float_list(value, n=None):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return None
        out = [float(v.strip()) for v in text.split(",") if v.strip()]
    else:
        out = [float(v) for v in value]
    if n is not None and len(out) != n:
        raise RuntimeError("Expected %d floats, got %s" % (n, out))
    return out


def _pose_from_xyz_quat(xyz, quat):
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def _xyz_from_pose_msg(pose):
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]


class RealTopGraspReplay(object):

    def __init__(self, demo_path="", trial_id="", replay_velocity_scale=None,
                 pregrasp_clearance_m=None, grasp_offset_xyz=None,
                 object_offset_xyz=None, replay_offset_xyz=None,
                 vision_y_linear_calibration_enabled=None,
                 vision_y_piecewise_compensation_enabled=None,
                 move_to_start_pose=None):
        rospy.init_node("mt3_sawyer_real_top_grasp_replay", anonymous=True)
        self.emergency_stop_requested = False
        self._traj_action_client = actionlib.SimpleActionClient(
            "/robot/limb/right/follow_joint_trajectory",
            FollowJointTrajectoryAction)
        self._emergency_stop_sub = rospy.Subscriber(
            "/mt3/emergency_stop",
            Bool,
            self._emergency_stop_callback,
            queue_size=1)
        rospy.on_shutdown(self._shutdown_stop_robot)

        self.demo_path = os.path.expanduser(demo_path) if demo_path else \
            self._default_demo_path()
        self.trial_id = trial_id or time.strftime("real_top_%Y%m%d_%H%M%S")
        self.log_dir = os.path.expanduser(rospy.get_param(
            "~real_top_grasp_log_dir", DEFAULT_LOG_DIR))

        with open(self.demo_path, "r") as f:
            self.demo = json.load(f)

        self.demo_object = np.asarray(
            self.demo["object_info"]["position_base"], dtype=np.float64)
        self.demo_size = np.asarray(
            self.demo["object_info"]["size_m"], dtype=np.float64)
        self.demo_top_z = float(
            self.demo.get("object_info", {}).get(
                "top_z_base",
                float(self.demo_object[2]) + abs(float(self.demo_size[2]))))

        self.trajectory = self._load_demo_trajectory()
        self.poses = self.trajectory["poses"]
        self.velocities = self.trajectory.get("velocities", [])
        self.close_index = self._find_close_index(self.poses, self.velocities)
        self.demo_bottleneck_pose = (
            self.demo.get("bottleneck_pose_base_frame") or self.poses[0])
        self.demo_bottleneck_tcp = self._pose_block_position(
            self.demo_bottleneck_pose)
        self.demo_bottleneck_orientation = self._pose_block_orientation(
            self.demo_bottleneck_pose)

        self.demo_grasp_pose = self.demo["grasp_pose_base_frame"]
        self.demo_grasp_tcp = self._pose_block_position(self.demo_grasp_pose)
        self.demo_orientation = self._pose_block_orientation(
            self.demo_grasp_pose)
        self.demo_mouth_center, self.demo_mouth_center_source = (
            self._demo_mouth_anchor_center())
        self.tcp_to_mouth_offset = self._demo_tcp_to_mouth_offset()
        self._log_tcp_mouth_residual()
        self.demo_mouth_object_offset_xy = (
            self.demo_mouth_center[:2] - self.demo_object[:2])
        self.demo_mouth_top_offset_z = (
            float(self.demo_mouth_center[2]) - float(self.demo_top_z))
        self.demo_bottleneck_object_offset_xy = (
            self.demo_bottleneck_tcp[:2] - self.demo_object[:2])
        self.demo_bottleneck_top_offset_z = (
            float(self.demo_bottleneck_tcp[2]) - float(self.demo_top_z))

        self.tail_correction_points = int(rospy.get_param(
            "~tail_correction_points", 10))
        self.max_preclose_waypoints = int(rospy.get_param(
            "~max_preclose_waypoints", 28))
        self.max_afterclose_waypoints = int(rospy.get_param(
            "~max_afterclose_waypoints", 50))
        self.cartesian_eef_step = float(rospy.get_param(
            "~cartesian_eef_step", 0.006))
        self.cartesian_min_fraction = float(rospy.get_param(
            "~cartesian_min_fraction", 0.85))
        self.max_velocity = float(rospy.get_param(
            "~max_velocity_scaling", 0.10))
        self.max_acceleration = float(rospy.get_param(
            "~max_acceleration_scaling", 0.10))
        if replay_velocity_scale is None:
            replay_velocity_scale = rospy.get_param(
                "~replay_velocity_scale", 1.0)
        self.replay_velocity_scale = max(
            0.05, min(1.0, float(replay_velocity_scale)))
        if pregrasp_clearance_m is None:
            pregrasp_clearance_m = rospy.get_param(
                "~pregrasp_only_clearance_m", 0.020)
        self.pregrasp_only_clearance_m = float(pregrasp_clearance_m)
        if grasp_offset_xyz is None:
            grasp_offset_xyz = rospy.get_param(
                "~grasp_tcp_offset_xyz", [0.0, 0.0, 0.0])
        self.grasp_tcp_offset = np.asarray(
            _as_float_list(grasp_offset_xyz, n=3), dtype=np.float64)
        if object_offset_xyz is None:
            object_offset_xyz = rospy.get_param(
                "~object_position_offset_xyz", [0.0, 0.0, 0.0])
        self.object_position_offset = np.asarray(
            _as_float_list(object_offset_xyz, n=3), dtype=np.float64)
        if vision_y_linear_calibration_enabled is None:
            vision_y_linear_calibration_enabled = rospy.get_param(
                "~vision_y_linear_calibration_enabled", False)
        self.vision_y_linear_calibration_enabled = bool(
            vision_y_linear_calibration_enabled)
        self.vision_y_linear_calibration_coeffs = np.asarray(_as_float_list(
            rospy.get_param(
                "~vision_y_linear_calibration_coeffs",
                [-0.01788, 0.02729, 0.02702]),
            n=3), dtype=np.float64)
        self.demo_vision_y_error = (
            float(self.vision_y_linear_calibration_coeffs[0]) *
            float(self.demo_object[0]) +
            float(self.vision_y_linear_calibration_coeffs[1]) *
            float(self.demo_object[1]) +
            float(self.vision_y_linear_calibration_coeffs[2]))
        if vision_y_piecewise_compensation_enabled is None:
            vision_y_piecewise_compensation_enabled = rospy.get_param(
                "~vision_y_piecewise_compensation_enabled", True)
        self.vision_y_piecewise_compensation_enabled = bool(
            vision_y_piecewise_compensation_enabled)
        self.vision_y_piecewise_high_threshold_mm = float(rospy.get_param(
            "~vision_y_piecewise_high_threshold_mm", 200.0))
        self.vision_y_piecewise_low_threshold_mm = float(rospy.get_param(
            "~vision_y_piecewise_low_threshold_mm", -150.0))
        self.vision_y_piecewise_high_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_high_dy_mm", 28.0))
        self.vision_y_piecewise_mid_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_mid_dy_mm", 15.0))
        self.vision_y_piecewise_mid_small_x_threshold_mm = float(rospy.get_param(
            "~vision_y_piecewise_mid_small_x_threshold_mm", 400.0))
        self.vision_y_piecewise_mid_small_x_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_mid_small_x_dy_mm", 23.0))
        self.vision_y_piecewise_low_x_slope = float(rospy.get_param(
            "~vision_y_piecewise_low_x_slope", -0.01933))
        self.vision_y_piecewise_low_x_intercept_mm = float(rospy.get_param(
            "~vision_y_piecewise_low_x_intercept_mm", 20.87))
        self.vision_y_piecewise_low_min_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_low_min_dy_mm", 4.0))
        self.vision_y_piecewise_low_max_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_low_max_dy_mm", 16.0))
        self.vision_y_piecewise_low_dy_mm = float(rospy.get_param(
            "~vision_y_piecewise_low_dy_mm", 8.0))
        if replay_offset_xyz is None:
            replay_offset_xyz = rospy.get_param(
                "~replay_waypoint_offset_xyz", [0.0, 0.0, 0.0])
        self.replay_waypoint_offset = np.asarray(
            _as_float_list(replay_offset_xyz, n=3), dtype=np.float64)
        self.success_min_object_lift_m = float(rospy.get_param(
            "~success_min_object_lift_m", 0.030))
        if move_to_start_pose is None:
            move_to_start_pose = rospy.get_param(
                "~move_to_start_pose", False)
        self.move_to_start_pose_enabled = bool(move_to_start_pose)
        self.real_start_joints = self._start_joint_dict_from_param(
            rospy.get_param(
                "~real_start_joint_positions", SIM_SAFE_START_JOINTS))
        self.start_pose_joint_speed = float(rospy.get_param(
            "~start_pose_joint_speed", 0.25))
        self.start_pose_timeout_s = float(rospy.get_param(
            "~start_pose_timeout_s", 15.0))
        self.start_pose_wait_s = float(rospy.get_param(
            "~start_pose_wait_s", 0.5))
        self.start_pose_tolerance_rad = float(rospy.get_param(
            "~start_pose_tolerance_rad", 0.035))

        self.move_group = None
        self.gripper = None
        self.limb = None
        self.last_execution_debug = {}
        self.start_pose_debug = {
            "move_to_start_pose_enabled": self.move_to_start_pose_enabled,
            "start_pose_reset_success": "",
            "start_pose_reset_time_s": "",
            "start_pose_joint_error_norm_rad": "",
            "start_pose_joint_error_max_rad": "",
            "start_pose_actual_before_joints": {},
            "start_pose_actual_after_joints": {},
        }
        self.timing = {
            "retrieval_time_s": 0.0,
            "alignment_time_s": 0.0,
            "planning_time_s": 0.0,
            "robot_execution_time_s": 0.0,
            "planning_call_count": 0,
            "robot_execution_call_count": 0,
            "timing_source": "direct_real_script",
        }

        rospy.loginfo("Loaded real demo: %s", self.demo_path)
        rospy.loginfo(
            "Demo object=%s size=%s top_z=%.4f close_index=%d poses=%d",
            self.demo_object, self.demo_size, self.demo_top_z,
            self.close_index, len(self.poses))
        rospy.loginfo(
            "Demo bottleneck tcp=%s object_xy=%s top_offset_z=%.4f",
            self.demo_bottleneck_tcp,
            self.demo_bottleneck_object_offset_xy,
            self.demo_bottleneck_top_offset_z)
        rospy.loginfo(
            "Demo mouth center=%s source=%s tcp_to_mouth=%s mouth_object_xy=%s "
            "mouth_top_z=%.4f",
            self.demo_mouth_center, self.demo_mouth_center_source,
            self.tcp_to_mouth_offset,
            self.demo_mouth_object_offset_xy,
            self.demo_mouth_top_offset_z)
        if np.linalg.norm(self.grasp_tcp_offset) > 1e-9:
            rospy.logwarn(
                "GRASP OFFSET configured but ignored; use object offset "
                "for calibration: x=%.3f y=%.3f z=%.3f m",
                self.grasp_tcp_offset[0],
                self.grasp_tcp_offset[1],
                self.grasp_tcp_offset[2])
        rospy.logwarn(
            "OBJECT POSITION OFFSET configured: x=%.3f y=%.3f z=%.3f m",
            self.object_position_offset[0],
            self.object_position_offset[1],
            self.object_position_offset[2])
        rospy.logwarn(
            "VISION Y LINEAR CALIBRATION configured: enabled=%s "
            "dy=(%.5f*x + %.5f*y + %.5f) - demo_error, "
            "demo_error=%.1fmm",
            self.vision_y_linear_calibration_enabled,
            self.vision_y_linear_calibration_coeffs[0],
            self.vision_y_linear_calibration_coeffs[1],
            self.vision_y_linear_calibration_coeffs[2],
            self.demo_vision_y_error * 1000.0)
        rospy.logwarn(
            "VISION Y PIECEWISE COMPENSATION configured: enabled=%s "
            "if y>%.1fmm dy=%.1fmm; "
            "elif y<%.1fmm dy=clamp(%.5f*x_mm + %.2f, %.1f, %.1f); "
            "elif x<%.1fmm dy=%.1fmm; else dy=%.1fmm",
            self.vision_y_piecewise_compensation_enabled,
            self.vision_y_piecewise_high_threshold_mm,
            self.vision_y_piecewise_high_dy_mm,
            self.vision_y_piecewise_low_threshold_mm,
            self.vision_y_piecewise_low_x_slope,
            self.vision_y_piecewise_low_x_intercept_mm,
            self.vision_y_piecewise_low_min_dy_mm,
            self.vision_y_piecewise_low_max_dy_mm,
            self.vision_y_piecewise_mid_small_x_threshold_mm,
            self.vision_y_piecewise_mid_small_x_dy_mm,
            self.vision_y_piecewise_mid_dy_mm)
        rospy.logwarn(
            "REPLAY WAYPOINT OFFSET configured: x=%.3f y=%.3f z=%.3f m",
            self.replay_waypoint_offset[0],
            self.replay_waypoint_offset[1],
            self.replay_waypoint_offset[2])
        rospy.logwarn(
            "REAL START POSE configured: enabled=%s joints=%s "
            "speed=%.3f timeout=%.1fs",
            self.move_to_start_pose_enabled,
            self._ordered_joint_values(self.real_start_joints),
            self.start_pose_joint_speed,
            self.start_pose_timeout_s)

    def _default_demo_path(self):
        for path in DEFAULT_DEMO_PATHS:
            if os.path.exists(path):
                return path
        raise RuntimeError(
            "No default real top-grasp demo found. Pass --demo_path. Tried: %s"
            % DEFAULT_DEMO_PATHS)

    def _load_demo_trajectory(self):
        trajectory = (
            self.demo.get("trajectory") or
            self.demo.get("_recorded_trajectory") or
            self.demo.get("grasp_trajectory") or {})
        poses = trajectory.get("poses", [])
        if not isinstance(poses, list) or len(poses) < 2:
            raise RuntimeError(
                "Real top grasp requires a recorded demo trajectory. "
                "No silent single-pose fallback is allowed.")
        out = dict(trajectory)
        out["poses"] = poses
        return out

    def _start_joint_dict_from_param(self, value):
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            out = {name: float(value[name]) for name in SAWYER_JOINT_ORDER}
        else:
            values = _as_float_list(value, n=len(SAWYER_JOINT_ORDER))
            out = dict(zip(SAWYER_JOINT_ORDER, values))
        return out

    def _ordered_joint_values(self, joint_dict):
        return [float(joint_dict[name]) for name in SAWYER_JOINT_ORDER]

    def _filtered_joint_dict(self, joint_dict):
        return {
            name: float(joint_dict.get(name, float("nan")))
            for name in SAWYER_JOINT_ORDER
        }

    def _emergency_stop_callback(self, msg):
        if not bool(msg.data):
            return
        if self.emergency_stop_requested:
            return

        self.emergency_stop_requested = True
        rospy.logerr("========== MT3 SOFTWARE EMERGENCY STOP ==========")

        try:
            self._traj_action_client.cancel_all_goals()
            rospy.logerr("Emergency stop: trajectory goals cancelled.")
        except Exception as exc:
            rospy.logerr("Emergency stop: cancel trajectory failed: %s", exc)

        try:
            if getattr(self, "move_group", None) is not None:
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                rospy.logerr("Emergency stop: MoveIt stop() sent.")
        except Exception as exc:
            rospy.logerr("Emergency stop: MoveIt stop failed: %s", exc)

    def _shutdown_stop_robot(self):
        try:
            if getattr(self, "_traj_action_client", None) is not None:
                self._traj_action_client.cancel_all_goals()
        except Exception:
            pass

        try:
            if getattr(self, "move_group", None) is not None:
                self.move_group.stop()
                self.move_group.clear_pose_targets()
        except Exception:
            pass

    def _check_emergency_stop(self):
        if self.emergency_stop_requested:
            raise RuntimeError(
                "MT3 software emergency stop requested; aborting trial.")

    def _pose_block_position(self, block):
        block = block or {}
        pos = block.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            return np.asarray(pos[:3], dtype=np.float64)
        pos_m = block.get("position_m")
        if isinstance(pos_m, dict):
            return np.asarray([
                float(pos_m["x"]),
                float(pos_m["y"]),
                float(pos_m["z"]),
            ], dtype=np.float64)
        raise RuntimeError("Pose block has no position: %s" % block)

    def _pose_block_orientation(self, block):
        block = block or {}
        ori = block.get("orientation")
        if isinstance(ori, list) and len(ori) >= 4:
            return [float(v) for v in ori[:4]]
        ori = block.get("orientation_xyzw")
        if isinstance(ori, dict):
            return [
                float(ori.get("x", 0.0)),
                float(ori.get("y", 0.0)),
                float(ori.get("z", 0.0)),
                float(ori.get("w", 1.0)),
            ]
        return [0.0, 0.0, 0.0, 1.0]

    def _pose_sample_position(self, sample):
        sample = sample or {}
        pos = sample.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            return np.asarray(pos[:3], dtype=np.float64)
        pose = sample.get("pose")
        if isinstance(pose, dict):
            return self._pose_block_position(pose)
        return self._pose_block_position(sample)

    def _gripper_binary(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return 1 if float(value) >= 0.5 else 0
        except Exception:
            text = str(value).strip().lower()
            if text in ("closed", "close", "closing", "true"):
                return 1
            if text in ("open", "opening", "false"):
                return 0
        return None

    def _find_close_index(self, poses, velocities):
        for i, sample in enumerate(poses):
            state = self._gripper_binary(
                sample.get("gripper_next", sample.get("gripper_state")))
            if state == 1:
                return i
        for i, sample in enumerate(velocities or []):
            state = self._gripper_binary(
                sample.get("gripper_next", sample.get("gripper_state")))
            if state == 1:
                return min(len(poses) - 1, i + 1)
        z_values = [self._pose_sample_position(p)[2] for p in poses]
        idx = int(np.argmin(z_values))
        rospy.logwarn(
            "Demo trajectory has no gripper close event; using lowest z index %d",
            idx)
        return idx

    def _demo_tcp_to_mouth_offset(self):
        return self.demo_mouth_center - self.demo_grasp_tcp

    def _legacy_demo_mouth_offset(self):
        calib = self.demo.get("top_grasp_mouth_center_calibration") or {}
        offset = calib.get("mouth_offset_xyz")
        if offset is not None and len(offset) >= 3:
            return np.asarray(offset[:3], dtype=np.float64)
        return None

    def _log_tcp_mouth_residual(self):
        residual = (
            self.demo_grasp_tcp +
            self.tcp_to_mouth_offset -
            self.demo_mouth_center
        )
        residual_mm = residual * 1000.0
        rospy.loginfo(
            "TCP-mouth calibration residual=[%.1f %.1f %.1f]mm",
            residual_mm[0], residual_mm[1], residual_mm[2])

        legacy = self._legacy_demo_mouth_offset()
        if legacy is not None:
            legacy_mouth = self.demo_grasp_tcp + legacy
            legacy_delta_mm = (legacy_mouth - self.demo_mouth_center) * 1000.0
            rospy.loginfo(
                "Legacy JSON mouth_offset_xyz residual vs recorded mouth="
                "[%.1f %.1f %.1f]mm",
                legacy_delta_mm[0], legacy_delta_mm[1], legacy_delta_mm[2])

    def _demo_mouth_anchor_center(self):
        calib = self.demo.get("top_grasp_mouth_center_calibration") or {}

        mouth = calib.get("mouth_center_xyz")
        if mouth is not None and len(mouth) >= 3:
            rospy.loginfo("Demo recorded mouth anchor selected.")
            return (
                np.asarray(mouth[:3], dtype=np.float64),
                "top_grasp_mouth_center_calibration.mouth_center_xyz"
            )

        traj = self.demo.get("trajectory") or {}
        state = traj.get("bottleneck_mouth_state") or {}
        mouth = state.get("mouth_center_xyz")
        if mouth is not None and len(mouth) >= 3:
            rospy.logwarn(
                "Using trajectory.bottleneck_mouth_state.mouth_center_xyz "
                "as demo mouth anchor.")
            return (
                np.asarray(mouth[:3], dtype=np.float64),
                "trajectory.bottleneck_mouth_state.mouth_center_xyz"
            )

        legacy = self._legacy_demo_mouth_offset()
        if legacy is not None:
            rospy.logwarn(
                "Demo has no recorded mouth_center_xyz; falling back to "
                "grasp_tcp + legacy mouth_offset_xyz.")
            return (
                self.demo_grasp_tcp + legacy,
                "grasp_tcp_plus_legacy_mouth_offset_xyz_fallback"
            )

        raise RuntimeError(
            "Demo missing top_grasp_mouth_center_calibration.mouth_center_xyz")

    def wait_for_camera_ready(self, timeout=10.0):
        rospy.loginfo("Waiting for ASC60C topics...")

        checks = [
            (
                "/ascamera_hp60c/rgb0/image",
                Image
            ),
            (
                "/ascamera_hp60c/depth0/image_raw",
                Image
            ),
        ]

        start = time.time()

        while not rospy.is_shutdown():
            try:
                for topic, msg_type in checks:
                    rospy.wait_for_message(
                        topic,
                        msg_type,
                        timeout=3.0
                    )

                rospy.loginfo("ASC60C RGB/depth/camera_info ready.")
                return True

            except Exception as exc:
                rospy.logwarn(
                    "Waiting ASC60C data: %s",
                    str(exc)
                )

            if time.time() - start > float(timeout):
                raise RuntimeError(
                    "ASC60C topics not ready after %.1fs" %
                    float(timeout)
                )

            rospy.sleep(0.2)

        return False

    def update_real_perception_params_once(self):
        timeout = float(rospy.get_param("~camera_ready_timeout_s", 10.0))
        stabilize_s = float(rospy.get_param("~camera_stabilize_s", 0.5))

        self.wait_for_camera_ready(timeout=timeout)

        from mt3_real_object_param_bridge import RealObjectParamBridge

        bridge = RealObjectParamBridge()

        if not bridge.perception.wait_for_registered_rgbd(timeout_s=5.0):
            raise RuntimeError(
                "ASC60C bridge subscribers did not receive RGB/depth/camera_info "
                "after 5.0s")

        if stabilize_s > 0.0:
            rospy.sleep(stabilize_s)

        ok = bridge.update_once()
        if not ok:
            raise RuntimeError("ASC60C real perception bridge failed")
        return True

    def get_current_object_geometry(self):
        keys = [
            "/mt3/current_object_x",
            "/mt3/current_object_y",
            "/mt3/current_object_z",
        ]
        if not all(rospy.has_param(k) for k in keys):
            raise RuntimeError(
                "Missing real object params. Run mt3_real_object_param_bridge.py "
                "or pass --update_perception.")
        raw_position = np.asarray([rospy.get_param(k) for k in keys],
                                  dtype=np.float64)
        position = raw_position + self.object_position_offset
        fixed_offset_position = position.copy()
        if np.linalg.norm(self.object_position_offset) > 1e-9:
            rospy.logwarn(
                "OBJECT POSITION OFFSET applied: raw=%s offset=%s adjusted=%s",
                raw_position, self.object_position_offset, position)

        vision_y_piecewise_dy = 0.0
        vision_y_piecewise_region = "disabled"
        if self.vision_y_piecewise_compensation_enabled:
            x_mm = float(position[0]) * 1000.0
            y_mm = float(position[1]) * 1000.0
            if y_mm > self.vision_y_piecewise_high_threshold_mm:
                vision_y_piecewise_dy = (
                    self.vision_y_piecewise_high_dy_mm / 1000.0)
                vision_y_piecewise_region = "high_y"
            elif y_mm < self.vision_y_piecewise_low_threshold_mm:
                dy_mm = (
                    self.vision_y_piecewise_low_x_slope * x_mm +
                    self.vision_y_piecewise_low_x_intercept_mm)
                dy_mm = max(
                    self.vision_y_piecewise_low_min_dy_mm,
                    min(self.vision_y_piecewise_low_max_dy_mm, dy_mm))
                vision_y_piecewise_dy = dy_mm / 1000.0
                vision_y_piecewise_region = "low_y_x_linear_clamped"
            elif x_mm < self.vision_y_piecewise_mid_small_x_threshold_mm:
                vision_y_piecewise_dy = (
                    self.vision_y_piecewise_mid_small_x_dy_mm / 1000.0)
                vision_y_piecewise_region = "mid_y_small_x"
            else:
                vision_y_piecewise_dy = (
                    self.vision_y_piecewise_mid_dy_mm / 1000.0)
                vision_y_piecewise_region = "mid_y"

            position[1] += vision_y_piecewise_dy
            rospy.logwarn(
                "VISION PIECEWISE COMPENSATION: fixed_xy=[%.1f %.1f]mm "
                "region=%s dy=%.1fmm corrected_y=%.1fmm",
                x_mm,
                y_mm,
                vision_y_piecewise_region,
                vision_y_piecewise_dy * 1000.0,
                float(position[1]) * 1000.0)

        vision_y_linear_dy = 0.0
        live_vision_y_error = 0.0
        if self.vision_y_linear_calibration_enabled:
            x = float(position[0])
            y = float(position[1])
            coeffs = self.vision_y_linear_calibration_coeffs
            live_vision_y_error = (
                float(coeffs[0]) * x +
                float(coeffs[1]) * y +
                float(coeffs[2]))
            vision_y_linear_dy = (
                live_vision_y_error - float(self.demo_vision_y_error))
            position[1] += vision_y_linear_dy
            rospy.logwarn(
                "VISION Y LINEAR CALIBRATION: fixed_xy=[%.4f %.4f] "
                "live_error=%.1fmm demo_error=%.1fmm relative_dy=%.1fmm "
                "corrected_xy=[%.4f %.4f]",
                x,
                y,
                live_vision_y_error * 1000.0,
                self.demo_vision_y_error * 1000.0,
                vision_y_linear_dy * 1000.0,
                float(position[0]),
                float(position[1]))

        size = None
        for key in [
                "/mt3/current_object_size_m",
                "/mt3/current_object_size",
                "/mt3_current_object_size"]:
            if rospy.has_param(key):
                size = np.asarray(_as_float_list(rospy.get_param(key), 3),
                                  dtype=np.float64)
                break
        if size is None:
            size = np.asarray(_as_float_list([
                rospy.get_param("/mt3/current_object_size_x"),
                rospy.get_param("/mt3/current_object_size_y"),
                rospy.get_param("/mt3/current_object_size_z"),
            ], 3), dtype=np.float64)

        top_z = None
        top_z_from_param = False
        for key in [
                "/mt3/current_object_top_z_base",
                "/mt3/current_object_top_z",
                "/mt3_current_object_top_z"]:
            if rospy.has_param(key):
                top_z = float(rospy.get_param(key))
                top_z_from_param = True
                break
        z_semantics = str(rospy.get_param(
            "/mt3/current_object_z_semantics", "bottom_surface_base"))
        if top_z is None:
            if z_semantics in ("bottom", "bottom_surface",
                               "bottom_surface_base"):
                top_z = float(position[2]) + abs(float(size[2]))
            elif z_semantics in ("center", "center_base", "centroid",
                                 "object_center"):
                top_z = float(position[2]) + 0.5 * abs(float(size[2]))
            elif z_semantics in ("top", "top_surface", "top_surface_base"):
                top_z = float(position[2])
            else:
                raise RuntimeError("Unknown object z semantics: %s" %
                                   z_semantics)
        elif top_z_from_param:
            top_z += float(self.object_position_offset[2])

        return {
            "position": position,
            "size": size,
            "top_z": float(top_z),
            "z_semantics": z_semantics,
            "source": rospy.get_param(
                "/mt3/current_object_source_frame", "base"),
            "raw_position": raw_position,
            "fixed_offset_position": fixed_offset_position,
            "vision_y_piecewise_dy_m": float(vision_y_piecewise_dy),
            "vision_y_piecewise_region": vision_y_piecewise_region,
            "vision_y_piecewise_compensation_enabled": bool(
                self.vision_y_piecewise_compensation_enabled),
            "vision_y_linear_dy_m": float(vision_y_linear_dy),
            "vision_y_linear_live_error_m": float(live_vision_y_error),
            "vision_y_linear_demo_error_m": float(self.demo_vision_y_error),
            "vision_y_linear_calibration_enabled": bool(
                self.vision_y_linear_calibration_enabled),
            "vision_y_linear_calibration_coeffs": (
                self.vision_y_linear_calibration_coeffs.copy()),
        }

    def _selected_pose_indices(self):
        close = max(0, min(len(self.poses) - 1, int(self.close_index)))
        pre = list(range(0, close + 1))
        post = list(range(close + 1, len(self.poses)))

        if len(pre) > self.max_preclose_waypoints:
            pre = sorted(set(np.linspace(
                0, close, self.max_preclose_waypoints).round().astype(int)))
        if len(post) > self.max_afterclose_waypoints:
            post = sorted(set(np.linspace(
                close + 1, len(self.poses) - 1,
                self.max_afterclose_waypoints).round().astype(int)))
        return pre + post, close

    def make_real_replay_waypoints(self, current_geometry):
        live_obj = np.asarray(current_geometry["position"], dtype=np.float64)
        live_size = np.asarray(current_geometry["size"], dtype=np.float64)
        live_top_z = float(current_geometry["top_z"])

        live_mouth_target = np.asarray([
            live_obj[0] + self.demo_mouth_object_offset_xy[0],
            live_obj[1] + self.demo_mouth_object_offset_xy[1],
            live_top_z + self.demo_mouth_top_offset_z,
        ], dtype=np.float64)
        target_close_tcp = live_mouth_target - self.tcp_to_mouth_offset
        mapped_bottleneck_tcp = np.asarray([
            live_obj[0] + self.demo_bottleneck_object_offset_xy[0],
            live_obj[1] + self.demo_bottleneck_object_offset_xy[1],
            live_top_z + self.demo_bottleneck_top_offset_z,
        ], dtype=np.float64)
        mapped_bottleneck_pose = _pose_from_xyz_quat(
            mapped_bottleneck_tcp, self.demo_orientation)

        selected_indices, original_close = self._selected_pose_indices()
        waypoints = []
        close_waypoint_index = None
        demo_close_tcp = self._pose_sample_position(self.poses[original_close])
        mapped_close_z = (
            live_top_z + (float(demo_close_tcp[2]) - float(self.demo_top_z)))

        for out_i, src_i in enumerate(selected_indices):
            demo_tcp = self._pose_sample_position(self.poses[src_i])
            rel_xy = demo_tcp[:2] - self.demo_object[:2]
            mapped = np.asarray([
                live_obj[0] + rel_xy[0],
                live_obj[1] + rel_xy[1],
                mapped_close_z + (float(demo_tcp[2]) - float(demo_close_tcp[2])),
            ], dtype=np.float64)
            waypoints.append(_pose_from_xyz_quat(mapped, self.demo_orientation))
            if src_i == original_close:
                close_waypoint_index = out_i

        if close_waypoint_index is None:
            raise RuntimeError("Selected local replay lost close waypoint")

        close_wp = waypoints[close_waypoint_index]
        close_xyz = np.asarray(_xyz_from_pose_msg(close_wp), dtype=np.float64)
        correction = target_close_tcp - close_xyz
        correction[2] = 0.0

        tail_n = max(1, min(int(self.tail_correction_points),
                            close_waypoint_index + 1))
        start_tail = max(0, close_waypoint_index - tail_n + 1)
        denom = max(1, close_waypoint_index - start_tail + 1)
        for i, pose in enumerate(waypoints):
            if i < start_tail:
                continue
            if i <= close_waypoint_index:
                alpha = float(i - start_tail + 1) / float(denom)
            else:
                alpha = 1.0
            pose.position.x += float(correction[0]) * alpha
            pose.position.y += float(correction[1]) * alpha

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i].position
            p2 = waypoints[i + 1].position
            d = (
                (float(p2.x) - float(p1.x)) ** 2 +
                (float(p2.y) - float(p1.y)) ** 2 +
                (float(p2.z) - float(p1.z)) ** 2
            ) ** 0.5
            rospy.logwarn(
                "WP %d->%d delta %.2f mm",
                i, i + 1, d * 1000.0)

        planned_close = np.asarray(
            _xyz_from_pose_msg(waypoints[close_waypoint_index]),
            dtype=np.float64)
        planned_mouth = planned_close + self.tcp_to_mouth_offset

        rospy.loginfo("===== REAL GEOMETRY GRASP ANCHOR =====")
        rospy.loginfo("demo object=%s size=%s top_z=%.4f",
                      self.demo_object, self.demo_size, self.demo_top_z)
        rospy.loginfo("live object=%s size=%s top_z=%.4f semantics=%s",
                      live_obj, live_size, live_top_z,
                      current_geometry.get("z_semantics", ""))
        rospy.loginfo("demo mouth center=%s mouth_object_xy=%s mouth_top_z=%.4f",
                      self.demo_mouth_center,
                      self.demo_mouth_object_offset_xy,
                      self.demo_mouth_top_offset_z)
        rospy.loginfo("mapped bottleneck tcp=%s", mapped_bottleneck_tcp)
        rospy.loginfo("live mouth target=%s target_close_tcp=%s",
                      live_mouth_target, target_close_tcp)
        rospy.loginfo(
            "tail correction points=%d start=%d close=%d "
            "xy_delta=[%.1f %.1f]mm z_delta_disabled computed_z_delta=%.1fmm",
            tail_n, start_tail, close_waypoint_index,
            correction[0] * 1000.0,
            correction[1] * 1000.0,
            (target_close_tcp[2] - close_xyz[2]) * 1000.0)
        rospy.loginfo("planned close tcp=%s mouth=%s mouth-object=[%.1f %.1f]mm",
                      planned_close, planned_mouth,
                      (planned_mouth[0] - live_obj[0]) * 1000.0,
                      (planned_mouth[1] - live_obj[1]) * 1000.0)

        return {
            "waypoints": waypoints,
            "close_index": close_waypoint_index,
            "live_mouth_target": live_mouth_target,
            "live_top_z": float(live_top_z),
            "mapped_bottleneck_pose": mapped_bottleneck_pose,
            "mapped_bottleneck_tcp": mapped_bottleneck_tcp,
            "target_close_tcp": target_close_tcp,
            "planned_close_tcp": planned_close,
            "planned_close_mouth": planned_mouth,
            "selected_source_indices": selected_indices,
        }

    def apply_replay_waypoint_offset(self, replay):
        offset = np.asarray(self.replay_waypoint_offset, dtype=np.float64)
        if np.linalg.norm(offset) <= 1e-9:
            return replay

        for pose in replay["waypoints"]:
            pose.position.x += float(offset[0])
            pose.position.y += float(offset[1])
            pose.position.z += float(offset[2])

        for key in [
                "live_mouth_target",
                "mapped_bottleneck_tcp",
                "target_close_tcp",
                "planned_close_tcp",
                "planned_close_mouth"]:
            if key in replay:
                replay[key] = np.asarray(replay[key],
                                         dtype=np.float64) + offset

        if "mapped_bottleneck_pose" in replay:
            pose = replay["mapped_bottleneck_pose"]
            pose.position.x += float(offset[0])
            pose.position.y += float(offset[1])
            pose.position.z += float(offset[2])

        rospy.logwarn(
            "REPLAY WAYPOINT OFFSET applied to all %d waypoints: "
            "x=%.3f y=%.3f z=%.3f m",
            len(replay["waypoints"]),
            offset[0], offset[1], offset[2])
        return replay

    def _pregrasp_only_waypoints(self, replay):
        waypoints = [copy_pose for copy_pose in replay["waypoints"]]
        close_index = int(replay["close_index"])
        before_close = [
            _pose_from_xyz_quat(_xyz_from_pose_msg(p), [
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w
            ])
            for p in waypoints[:close_index + 1]
        ]

        safe_mouth_z = (
            float(replay["live_top_z"]) +
            max(0.0, float(self.pregrasp_only_clearance_m)))
        raised_count = 0
        for pose in before_close:
            mouth_z = float(pose.position.z) + float(self.tcp_to_mouth_offset[2])
            if mouth_z < safe_mouth_z:
                pose.position.z += safe_mouth_z - mouth_z
                raised_count += 1

        if before_close:
            planned_mouth = np.asarray(replay["planned_close_mouth"],
                                       dtype=np.float64)
            pregrasp_tcp = np.asarray(
                _xyz_from_pose_msg(before_close[-1]), dtype=np.float64)
            pregrasp_mouth = pregrasp_tcp + self.tcp_to_mouth_offset
            if float(pregrasp_mouth[2]) < safe_mouth_z:
                dz = safe_mouth_z - float(pregrasp_mouth[2])
                before_close[-1].position.z += dz
                pregrasp_tcp[2] += dz
                pregrasp_mouth[2] += dz
        else:
            pregrasp_tcp = np.asarray(replay["planned_close_tcp"],
                                      dtype=np.float64)
            pregrasp_mouth = pregrasp_tcp + self.tcp_to_mouth_offset

        rospy.logwarn("===== PREGRASP ONLY TARGET =====")
        rospy.logwarn("live_top_z=%.4f clearance=%.3f safe_mouth_z=%.4f",
                      float(replay["live_top_z"]),
                      float(self.pregrasp_only_clearance_m),
                      safe_mouth_z)
        rospy.logwarn("planned close TCP=%s mouth=%s",
                      replay["planned_close_tcp"],
                      replay["planned_close_mouth"])
        rospy.logwarn("pregrasp stop TCP=%s mouth=%s raised_waypoints=%d",
                      pregrasp_tcp, pregrasp_mouth, raised_count)

        return before_close, pregrasp_tcp, pregrasp_mouth, safe_mouth_z

    def _init_robot_interfaces(self):
        if self.move_group is not None and self.gripper is not None \
                and self.limb is not None:
            return
        self._check_real_robot_state()

        import moveit_commander
        from intera_interface import Gripper, Limb

        moveit_commander.roscpp_initialize(sys.argv)
        ns = str(rospy.get_param("~moveit_ns", "/robot"))
        robot_description = str(rospy.get_param(
            "~robot_description", "/robot/robot_description"))
        group_name = str(rospy.get_param("~move_group", "right_arm"))
        self._ensure_moveit_semantic_param(robot_description)

        self.move_group = moveit_commander.MoveGroupCommander(
            group_name, robot_description=robot_description, ns=ns)
        velocity_scale = max(
            0.01, min(1.0, self.max_velocity * self.replay_velocity_scale))
        acceleration_scale = max(
            0.01,
            min(1.0, self.max_acceleration * self.replay_velocity_scale))
        self.move_group.set_max_velocity_scaling_factor(velocity_scale)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scale)
        rospy.loginfo(
            "MoveIt replay speed scale: base_vel=%.3f base_acc=%.3f "
            "replay_velocity_scale=%.3f active_vel=%.3f active_acc=%.3f",
            self.max_velocity, self.max_acceleration,
            self.replay_velocity_scale, velocity_scale, acceleration_scale)
        self.move_group.set_planning_time(float(rospy.get_param(
            "~planning_time_s", 10.0)))
        self.gripper = Gripper("right_gripper")
        self.limb = Limb("right")

    def _move_to_real_start_pose(self):
        if not self.move_to_start_pose_enabled:
            self.start_pose_debug["move_to_start_pose_enabled"] = False
            return

        self._init_robot_interfaces()
        self._check_emergency_stop()
        rospy.logwarn(
            "MOVING TO REAL START POSE before trial timing: joints=%s",
            self._ordered_joint_values(self.real_start_joints))

        before = self._filtered_joint_dict(self.limb.joint_angles())
        t0 = time.time()
        try:
            self.gripper.open()
            rospy.sleep(0.4)
        except Exception as exc:
            rospy.logwarn("Gripper open during start-pose reset failed: %s", exc)

        try:
            self.limb.set_joint_position_speed(self.start_pose_joint_speed)
        except Exception as exc:
            rospy.logwarn("Could not set start-pose joint speed: %s", exc)

        self.limb.move_to_joint_positions(
            self.real_start_joints, timeout=self.start_pose_timeout_s)
        rospy.sleep(self.start_pose_wait_s)
        self._check_emergency_stop()

        after = self._filtered_joint_dict(self.limb.joint_angles())
        errors = np.asarray([
            float(after[name]) - float(self.real_start_joints[name])
            for name in SAWYER_JOINT_ORDER
        ], dtype=np.float64)
        error_norm = float(np.linalg.norm(errors))
        error_max = float(np.max(np.abs(errors)))
        success = error_max <= self.start_pose_tolerance_rad

        self.start_pose_debug.update({
            "move_to_start_pose_enabled": True,
            "start_pose_reset_success": bool(success),
            "start_pose_reset_time_s": time.time() - t0,
            "start_pose_joint_error_norm_rad": error_norm,
            "start_pose_joint_error_max_rad": error_max,
            "start_pose_actual_before_joints": before,
            "start_pose_actual_after_joints": after,
        })
        rospy.logwarn(
            "REAL START POSE ACTUAL joints=%s error_norm=%.4frad "
            "max=%.4frad success=%s",
            self._ordered_joint_values(after),
            error_norm,
            error_max,
            success)
        if not success:
            raise RuntimeError(
                "Start pose reset joint error too large: max=%.4f rad > %.4f rad"
                % (error_max, self.start_pose_tolerance_rad))

    def _ensure_moveit_semantic_param(self, robot_description):
        semantic_param = "%s_semantic" % robot_description
        candidates = [
            semantic_param,
            "/robot/robot_description_semantic",
            "/robot_description_semantic",
        ]

        source = None
        semantic = None
        for candidate in candidates:
            if rospy.has_param(candidate):
                source = candidate
                semantic = rospy.get_param(candidate)
                break

        if semantic is None:
            raise RuntimeError(
                "MoveIt SRDF parameter not found. Tried: %s" %
                ", ".join(candidates))

        for target in [semantic_param, "/robot_description_semantic"]:
            if not rospy.has_param(target):
                rospy.set_param(target, semantic)
                rospy.loginfo(
                    "Mirrored MoveIt SRDF param %s -> %s",
                    source, target)

    def _check_real_robot_state(self):
        from intera_core_msgs.msg import RobotAssemblyState

        state = rospy.wait_for_message("/robot/state", RobotAssemblyState,
                                       timeout=3.0)
        if hasattr(state, "homed") and not bool(state.homed):
            raise RuntimeError("Sawyer is not homed; refusing real motion.")
        if bool(state.error):
            raise RuntimeError("Sawyer reports error=True; clear manually.")
        if bool(state.stopped):
            raise RuntimeError("Sawyer reports stopped=True; check E-stop.")
        if not bool(state.ready):
            raise RuntimeError("Sawyer reports ready=False; refusing motion.")
        if not bool(state.enabled):
            raise RuntimeError(
                "Sawyer is not enabled. Enable manually before --execute.")
        rospy.loginfo("Sawyer state OK: ready/enabled and no error/stopped.")

    def _execute_cartesian_segment(self, waypoints, label):
        if not waypoints:
            return 1.0
        self._check_emergency_stop()
        self.move_group.set_start_state_to_current_state()
        t_plan = time.time()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, self.cartesian_eef_step, True)
        self.timing["planning_time_s"] += time.time() - t_plan
        self.timing["planning_call_count"] += 1
        self.last_execution_debug["last_cartesian_label"] = label
        self.last_execution_debug["last_cartesian_fraction"] = float(fraction)
        if "pregrasp-only" in label or "before-close" in label:
            self.last_execution_debug["before_close_cartesian_fraction"] = (
                float(fraction))
        elif "after-close" in label or "fallback lift" in label:
            self.last_execution_debug["after_close_cartesian_fraction"] = (
                float(fraction))
        rospy.loginfo("%s cartesian fraction %.1f%% waypoints=%d",
                      label, fraction * 100.0, len(waypoints))
        if fraction < self.cartesian_min_fraction:
            raise RuntimeError(
                "%s cartesian fraction %.1f%% < %.1f%%" %
                (label, fraction * 100.0,
                 self.cartesian_min_fraction * 100.0))
        self._check_emergency_stop()
        t_exec = time.time()
        ok = self.move_group.execute(plan, wait=True)
        self.timing["robot_execution_time_s"] += time.time() - t_exec
        self.timing["robot_execution_call_count"] += 1
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self._check_emergency_stop()
        if not ok:
            raise RuntimeError("%s execution failed" % label)
        rospy.sleep(0.3)
        return float(fraction)

    def _move_to_mapped_bottleneck(self, replay):
        if not bool(rospy.get_param("~move_to_mapped_bottleneck", True)):
            rospy.logwarn("Mapped bottleneck pre-move disabled by parameter.")
            return {
                "bottleneck_move_enabled": False,
            }
        pose = replay.get("mapped_bottleneck_pose")
        if pose is None:
            rospy.logwarn("No mapped bottleneck pose available; skipping pre-move.")
            return {
                "bottleneck_move_enabled": False,
            }
        self._check_emergency_stop()

        target_xyz = np.asarray(_xyz_from_pose_msg(pose), dtype=np.float64)
        current_pose = self.move_group.get_current_pose().pose
        current_xyz = np.asarray(
            _xyz_from_pose_msg(current_pose), dtype=np.float64)
        delta = target_xyz - current_xyz
        rospy.logwarn(
            "MOVE TO MAPPED BOTTLENECK: current=%s target=%s "
            "delta=[%.1f %.1f %.1f]mm norm=%.1fmm",
            current_xyz,
            target_xyz,
            delta[0] * 1000.0,
            delta[1] * 1000.0,
            delta[2] * 1000.0,
            np.linalg.norm(delta) * 1000.0)

        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose)
        t_exec = time.time()
        ok = self.move_group.go(wait=True)
        self.timing["robot_execution_time_s"] += time.time() - t_exec
        self.timing["robot_execution_call_count"] += 1
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self._check_emergency_stop()
        if not ok:
            raise RuntimeError("MoveIt failed to move to mapped bottleneck")
        rospy.sleep(0.3)
        actual_pose = self.move_group.get_current_pose().pose
        actual_xyz = np.asarray(
            _xyz_from_pose_msg(actual_pose), dtype=np.float64)
        error = actual_xyz - target_xyz
        rospy.logwarn(
            "MAPPED BOTTLENECK ACTUAL=%s error=[%.1f %.1f %.1f]mm "
            "norm=%.1fmm",
            actual_xyz,
            error[0] * 1000.0,
            error[1] * 1000.0,
            error[2] * 1000.0,
            np.linalg.norm(error) * 1000.0)
        return {
            "bottleneck_move_enabled": True,
            "mapped_bottleneck_xyz": target_xyz.tolist(),
            "actual_bottleneck_xyz": actual_xyz.tolist(),
            "bottleneck_error_m": error.tolist(),
            "bottleneck_error_norm_m": float(np.linalg.norm(error)),
        }

    def _start_to_first_waypoint_debug(self, initial_hand, waypoints):
        if not waypoints:
            return {
                "start_tcp_xyz": [],
                "first_waypoint_xyz": [],
                "start_to_first_waypoint_delta_m": [],
                "start_to_first_waypoint_error_m": "",
            }
        current_xyz = np.asarray(
            _xyz_from_pose_msg(initial_hand), dtype=np.float64)
        first_wp_xyz = np.asarray(
            _xyz_from_pose_msg(waypoints[0]), dtype=np.float64)
        start_delta = first_wp_xyz - current_xyz
        rospy.logwarn(
            "START TCP=%s FIRST WP=%s delta=[%.1f %.1f %.1f]mm "
            "norm=%.1fmm",
            current_xyz,
            first_wp_xyz,
            start_delta[0] * 1000.0,
            start_delta[1] * 1000.0,
            start_delta[2] * 1000.0,
            np.linalg.norm(start_delta) * 1000.0)
        return {
            "start_tcp_xyz": current_xyz.tolist(),
            "first_waypoint_xyz": first_wp_xyz.tolist(),
            "start_to_first_waypoint_delta_m": start_delta.tolist(),
            "start_to_first_waypoint_error_m": float(np.linalg.norm(start_delta)),
        }

    def execute_replay_waypoints(self, replay, dry_run=True,
                                 pregrasp_only=False):
        waypoints = replay["waypoints"]
        close_index = int(replay["close_index"])
        before_close = waypoints[:close_index + 1]
        after_close = waypoints[close_index + 1:]

        rospy.loginfo("Real replay summary: total=%d before_close=%d after=%d",
                      len(waypoints), len(before_close), len(after_close))
        rospy.loginfo("Planned close TCP=%s mouth=%s",
                      replay["planned_close_tcp"],
                      replay["planned_close_mouth"])

        if dry_run:
            rospy.logwarn("DRY RUN: no real Sawyer motion.")
            return {
                "success": True,
                "dry_run": True,
                "close_executed": False,
            }

        self._init_robot_interfaces()
        self._check_emergency_stop()
        try:
            self.gripper.open()
            rospy.sleep(0.5)
        except Exception as exc:
            rospy.logwarn("Gripper open failed before replay: %s", exc)
        self._check_emergency_stop()

        bottleneck_result = self._move_to_mapped_bottleneck(replay)
        self.last_execution_debug.update(bottleneck_result)
        self._check_emergency_stop()
        initial_hand = self.move_group.get_current_pose().pose
        if pregrasp_only:
            pregrasp_waypoints, planned_pregrasp_tcp, planned_pregrasp_mouth, \
                safe_mouth_z = self._pregrasp_only_waypoints(replay)
            start_result = self._start_to_first_waypoint_debug(
                initial_hand, pregrasp_waypoints)
            self.last_execution_debug.update(start_result)
            pregrasp_fraction = self._execute_cartesian_segment(
                pregrasp_waypoints, "real pregrasp-only before-close replay")
            self._check_emergency_stop()
            actual_pregrasp = self.move_group.get_current_pose().pose
            actual_pregrasp_xyz = np.asarray(
                _xyz_from_pose_msg(actual_pregrasp), dtype=np.float64)
            delta = actual_pregrasp_xyz - planned_pregrasp_tcp
            actual_mouth = actual_pregrasp_xyz + self.tcp_to_mouth_offset
            rospy.logwarn("===== PREGRASP ONLY DEBUG =====")
            rospy.logwarn("target pregrasp TCP=%s mouth=%s",
                          planned_pregrasp_tcp, planned_pregrasp_mouth)
            rospy.logwarn("actual pregrasp TCP=%s mouth=%s",
                          actual_pregrasp_xyz, actual_mouth)
            rospy.logwarn(
                "actual-target delta=[%.1f %.1f %.1f]mm norm=%.1fmm "
                "actual_mouth_z=%.4f safe_mouth_z=%.4f",
                delta[0] * 1000.0, delta[1] * 1000.0,
                delta[2] * 1000.0, np.linalg.norm(delta) * 1000.0,
                float(actual_mouth[2]), float(safe_mouth_z))
            rospy.logwarn(
                "PREGRASP ONLY: stopped above object; no gripper close, "
                "no after-close replay, no lift.")
            result = {
                "success": True,
                "dry_run": False,
                "pregrasp_only": True,
                "close_executed": False,
                "actual_close_tcp": actual_pregrasp_xyz.tolist(),
                "actual_close_delta_m": delta.tolist(),
                "planned_pregrasp_tcp": planned_pregrasp_tcp.tolist(),
                "planned_pregrasp_mouth": planned_pregrasp_mouth.tolist(),
                "actual_pregrasp_mouth": actual_mouth.tolist(),
                "safe_mouth_z": float(safe_mouth_z),
                "pregrasp_cartesian_fraction": float(pregrasp_fraction),
                "before_close_cartesian_fraction": float(pregrasp_fraction),
            }
            result.update(bottleneck_result)
            result.update(start_result)
            return result

        start_result = self._start_to_first_waypoint_debug(
            initial_hand, before_close)
        self.last_execution_debug.update(start_result)
        before_fraction = self._execute_cartesian_segment(
            before_close, "real before-close replay")
        self._check_emergency_stop()
        actual_close = self.move_group.get_current_pose().pose
        actual_close_xyz = np.asarray(_xyz_from_pose_msg(actual_close),
                                      dtype=np.float64)
        planned_close_xyz = np.asarray(replay["planned_close_tcp"],
                                       dtype=np.float64)
        delta = actual_close_xyz - planned_close_xyz
        rospy.logwarn("===== REAL CLOSE TCP DEBUG =====")
        rospy.logwarn("planned close TCP=%s", planned_close_xyz)
        rospy.logwarn("actual close TCP=%s", actual_close_xyz)
        rospy.logwarn("actual-planned delta=[%.1f %.1f %.1f]mm norm=%.1fmm",
                      delta[0] * 1000.0, delta[1] * 1000.0,
                      delta[2] * 1000.0, np.linalg.norm(delta) * 1000.0)

        self._check_emergency_stop()
        rospy.loginfo("Closing real gripper at recorded event.")
        self.gripper.close()
        rospy.sleep(0.8)
        self._check_emergency_stop()

        lift_start = self.move_group.get_current_pose().pose
        lift_start_z = float(lift_start.position.z)
        vertical_lift_m = float(rospy.get_param("~vertical_lift_m", 0.100))
        lift_pose = self.move_group.get_current_pose().pose
        lift_pose.position.z = lift_start_z + vertical_lift_m
        rospy.loginfo(
            "REAL TOP GRASP: recorded after-close replay skipped; "
            "pure vertical lift %.1f mm at fixed X/Y/orientation",
            vertical_lift_m * 1000.0)
        after_fraction = self._execute_cartesian_segment(
            [lift_pose], "real vertical lift")

        final_hand = self.move_group.get_current_pose().pose
        hand_lift = float(final_hand.position.z) - lift_start_z
        success = hand_lift >= self.success_min_object_lift_m
        rospy.loginfo(
            "REAL VERTICAL LIFT: start_z=%.4f target_z=%.4f "
            "actual_z=%.4f lift=%.1fmm threshold=%.1fmm success=%s",
            lift_start_z,
            lift_start_z + vertical_lift_m,
            float(final_hand.position.z),
            hand_lift * 1000.0,
            self.success_min_object_lift_m * 1000.0,
            success)
        result = {
            "success": bool(success),
            "dry_run": False,
            "pregrasp_only": False,
            "close_executed": True,
            "actual_close_tcp": actual_close_xyz.tolist(),
            "actual_close_delta_m": delta.tolist(),
            "hand_lift_m": hand_lift,
            "before_close_cartesian_fraction": float(before_fraction),
            "after_close_cartesian_fraction": float(after_fraction),
            "recorded_after_close_skipped": True,
            "recorded_after_close_waypoints": len(after_close),
            "vertical_lift_m": float(vertical_lift_m),
            "lift_start_z": float(lift_start_z),
            "lift_target_z": float(lift_start_z + vertical_lift_m),
            "replay_success": bool(success),
        }
        result.update(bottleneck_result)
        result.update(start_result)
        return result

    def _write_log(self, current_geometry, replay, result):
        os.makedirs(self.log_dir, exist_ok=True)

        def dumps_float_list(value):
            if value is None:
                return json.dumps([])
            if isinstance(value, str):
                return value
            return json.dumps([float(v) for v in value])

        def dumps_joint_dict(value):
            if not isinstance(value, dict):
                return json.dumps({})
            return json.dumps({
                name: float(value.get(name, float("nan")))
                for name in SAWYER_JOINT_ORDER
            })

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": "%.3f" % time.time(),
            "trial_id": self.trial_id,
            "demo_path": self.demo_path,
            "success": bool(result.get("success", False)),
            "dry_run": bool(result.get("dry_run", True)),
            "pregrasp_only": bool(result.get("pregrasp_only", False)),
            "failure_reason": result.get("failure_reason", ""),
            "emergency_stop_requested": bool(self.emergency_stop_requested),
            "move_to_start_pose_enabled": bool(
                self.start_pose_debug.get(
                    "move_to_start_pose_enabled", False)),
            "real_start_joint_targets": dumps_joint_dict(
                self.real_start_joints),
            "start_pose_reset_success": self.start_pose_debug.get(
                "start_pose_reset_success", ""),
            "start_pose_reset_time_s": self.start_pose_debug.get(
                "start_pose_reset_time_s", ""),
            "start_pose_joint_error_norm_rad": self.start_pose_debug.get(
                "start_pose_joint_error_norm_rad", ""),
            "start_pose_joint_error_max_rad": self.start_pose_debug.get(
                "start_pose_joint_error_max_rad", ""),
            "start_pose_actual_before_joints": dumps_joint_dict(
                self.start_pose_debug.get(
                    "start_pose_actual_before_joints", {})),
            "start_pose_actual_after_joints": dumps_joint_dict(
                self.start_pose_debug.get(
                    "start_pose_actual_after_joints", {})),
            "raw_object_xyz": dumps_float_list(
                current_geometry.get("raw_position", [])),
            "fixed_offset_object_xyz": dumps_float_list(
                current_geometry.get("fixed_offset_position", [])),
            "live_object_xyz": json.dumps(
                [float(v) for v in current_geometry["position"]]),
            "live_object_size": json.dumps(
                [float(v) for v in current_geometry["size"]]),
            "live_top_z": float(current_geometry["top_z"]),
            "z_semantics": current_geometry.get("z_semantics", ""),
            "object_position_offset_xyz": dumps_float_list(
                self.object_position_offset),
            "vision_y_piecewise_compensation_enabled": bool(
                current_geometry.get(
                    "vision_y_piecewise_compensation_enabled", False)),
            "vision_y_piecewise_region": current_geometry.get(
                "vision_y_piecewise_region", ""),
            "vision_y_piecewise_dy_m": current_geometry.get(
                "vision_y_piecewise_dy_m", ""),
            "vision_y_linear_calibration_enabled": bool(
                current_geometry.get(
                    "vision_y_linear_calibration_enabled", False)),
            "vision_y_linear_calibration_coeffs": dumps_float_list(
                current_geometry.get(
                    "vision_y_linear_calibration_coeffs", [])),
            "vision_y_linear_dy_m": current_geometry.get(
                "vision_y_linear_dy_m", ""),
            "vision_y_linear_live_error_m": current_geometry.get(
                "vision_y_linear_live_error_m", ""),
            "vision_y_linear_demo_error_m": current_geometry.get(
                "vision_y_linear_demo_error_m", ""),
            "replay_waypoint_offset_xyz": dumps_float_list(
                self.replay_waypoint_offset),
            "demo_bottleneck_xyz": dumps_float_list(
                self.demo_bottleneck_tcp),
            "mapped_bottleneck_xyz": dumps_float_list(
                replay.get("mapped_bottleneck_tcp", [])),
            "actual_bottleneck_xyz": dumps_float_list(
                result.get("actual_bottleneck_xyz", [])),
            "bottleneck_error_m": dumps_float_list(
                result.get("bottleneck_error_m", [])),
            "bottleneck_error_norm_m": result.get(
                "bottleneck_error_norm_m", ""),
            "bottleneck_move_enabled": result.get(
                "bottleneck_move_enabled", ""),
            "start_tcp_xyz": dumps_float_list(
                result.get("start_tcp_xyz", [])),
            "first_waypoint_xyz": dumps_float_list(
                result.get("first_waypoint_xyz", [])),
            "start_to_first_waypoint_delta_m": dumps_float_list(
                result.get("start_to_first_waypoint_delta_m", [])),
            "start_to_first_waypoint_error_m": result.get(
                "start_to_first_waypoint_error_m", ""),
            "planned_close_tcp": json.dumps(
                [float(v) for v in replay["planned_close_tcp"]]),
            "planned_close_mouth": json.dumps(
                [float(v) for v in replay["planned_close_mouth"]]),
            "actual_close_tcp": json.dumps(
                result.get("actual_close_tcp", [])),
            "actual_close_delta_m": json.dumps(
                result.get("actual_close_delta_m", [])),
            "planned_pregrasp_tcp": json.dumps(
                result.get("planned_pregrasp_tcp", [])),
            "planned_pregrasp_mouth": json.dumps(
                result.get("planned_pregrasp_mouth", [])),
            "actual_pregrasp_mouth": json.dumps(
                result.get("actual_pregrasp_mouth", [])),
            "safe_mouth_z": result.get("safe_mouth_z", ""),
            "hand_lift_m": result.get("hand_lift_m", ""),
            "recorded_after_close_skipped": result.get(
                "recorded_after_close_skipped", ""),
            "recorded_after_close_waypoints": result.get(
                "recorded_after_close_waypoints", ""),
            "vertical_lift_m": result.get("vertical_lift_m", ""),
            "lift_start_z": result.get("lift_start_z", ""),
            "lift_target_z": result.get("lift_target_z", ""),
            "pregrasp_cartesian_fraction": result.get(
                "pregrasp_cartesian_fraction", ""),
            "before_close_cartesian_fraction": result.get(
                "before_close_cartesian_fraction", ""),
            "after_close_cartesian_fraction": result.get(
                "after_close_cartesian_fraction", ""),
            "last_cartesian_label": result.get("last_cartesian_label", ""),
            "last_cartesian_fraction": result.get(
                "last_cartesian_fraction", ""),
            "replay_success": result.get("replay_success", ""),
            "close_executed": bool(result.get("close_executed", False)),
            "total_time_s": self.timing.get("total_time_s", ""),
            "perception_time_s": self.timing.get("perception_time_s", ""),
            "retrieval_time_s": self.timing.get("retrieval_time_s", ""),
            "alignment_time_s": self.timing.get("alignment_time_s", ""),
            "planning_time_s": self.timing.get("planning_time_s", ""),
            "robot_execution_time_s": self.timing.get(
                "robot_execution_time_s", ""),
            "execution_wall_time_s": self.timing.get(
                "execution_wall_time_s", ""),
            "execution_time_s": self.timing.get("execution_time_s", ""),
            "planning_call_count": self.timing.get(
                "planning_call_count", ""),
            "robot_execution_call_count": self.timing.get(
                "robot_execution_call_count", ""),
            "timing_source": self.timing.get("timing_source", ""),
            "selected_source_indices": json.dumps(
                [int(v) for v in replay["selected_source_indices"]]),
        }
        def write_row_to_dir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, "mt3_real_top_grasp_trials.csv")
            fieldnames = list(row.keys())
            exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            if exists:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_fieldnames = reader.fieldnames or []
                    if old_fieldnames != fieldnames:
                        old_rows = list(reader)
                        merged_fieldnames = list(old_fieldnames)
                        for name in fieldnames:
                            if name not in merged_fieldnames:
                                merged_fieldnames.append(name)
                        with open(csv_path, "w", newline="",
                                  encoding="utf-8") as wf:
                            writer = csv.DictWriter(
                                wf, fieldnames=merged_fieldnames)
                            writer.writeheader()
                            for old_row in old_rows:
                                writer.writerow(old_row)
                        fieldnames = merged_fieldnames
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            jsonl_path = os.path.join(
                log_dir, "mt3_real_top_grasp_trials.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return csv_path

        try:
            csv_path = write_row_to_dir(self.log_dir)
            rospy.loginfo("Real top grasp trial logged: %s", csv_path)
        except Exception as exc:
            fallback_dir = os.path.expanduser(
                "~/code/learning_thousand_tasks/demo_library/real/"
                "experiment_logs/top_grasp")
            rospy.logerr(
                "Primary real top grasp log dir failed: %s; "
                "falling back to %s",
                exc, fallback_dir)
            try:
                csv_path = write_row_to_dir(fallback_dir)
                rospy.loginfo(
                    "Real top grasp trial logged to fallback: %s", csv_path)
            except Exception as fallback_exc:
                rospy.logerr(
                    "Fallback real top grasp logging failed: %s",
                    fallback_exc)

    def run(self, dry_run=True, update_perception=False,
            pregrasp_only=False):
        if not dry_run:
            self._move_to_real_start_pose()

        run_start = time.time()
        self.timing["run_start"] = run_start

        if update_perception:
            t_perception = time.time()
            self.update_real_perception_params_once()
            self.timing["perception_time_s"] = time.time() - t_perception
        else:
            self.timing["perception_time_s"] = 0.0

        current_geometry = self.get_current_object_geometry()
        rospy.loginfo("Current real object position=%s size=%s top_z=%.4f",
                      current_geometry["position"], current_geometry["size"],
                      current_geometry["top_z"])
        t_alignment = time.time()
        replay = self.make_real_replay_waypoints(current_geometry)
        replay = self.apply_replay_waypoint_offset(replay)
        self.timing["alignment_time_s"] = time.time() - t_alignment
        t_execution = time.time()
        try:
            result = self.execute_replay_waypoints(
                replay, dry_run=dry_run, pregrasp_only=pregrasp_only)
            execution_time_s = time.time() - t_execution
            self.timing["execution_time_s"] = execution_time_s
            self.timing["execution_wall_time_s"] = execution_time_s
            self.timing["total_time_s"] = time.time() - run_start
            self._write_log(current_geometry, replay, result)
            return bool(result.get("success", False))
        except Exception as exc:
            execution_time_s = time.time() - t_execution
            self.timing["execution_time_s"] = execution_time_s
            self.timing["execution_wall_time_s"] = execution_time_s
            self.timing["total_time_s"] = time.time() - run_start
            result = {
                "success": False,
                "dry_run": bool(dry_run),
                "pregrasp_only": bool(pregrasp_only),
                "close_executed": False,
                "replay_success": False,
                "failure_reason": str(exc),
            }
            result.update(self.last_execution_debug)
            try:
                self._write_log(current_geometry, replay, result)
            except Exception as log_exc:
                rospy.logerr("Failed to write failed-trial log: %s", log_exc)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Force dry-run even if --execute is present.")
    parser.add_argument("--execute", action="store_true",
                        help="Allow real Sawyer motion.")
    parser.add_argument("--pregrasp_only", action="store_true",
                        help="Execute only to a safe stop above object top; "
                             "do not close or lift.")
    parser.add_argument("--pregrasp_clearance_m", type=float, default=None,
                        help="Mouth-center clearance above live object top for "
                             "--pregrasp_only. Default 0.020 m.")
    parser.add_argument("--update_perception", action="store_true",
                        help="Run ASC60C perception bridge once before planning.")
    parser.add_argument("--replay_velocity_scale", type=float, default=None,
                        help="Extra multiplier for MoveIt replay velocity and "
                             "acceleration scaling. Default keeps current speed.")
    parser.add_argument("--grasp_offset_x", type=float, default=0.0,
                        help="Deprecated no-op. Use --object_offset_x instead.")
    parser.add_argument("--grasp_offset_y", type=float, default=0.0,
                        help="Deprecated no-op. Use --object_offset_y instead.")
    parser.add_argument("--grasp_offset_z", type=float, default=0.0,
                        help="Deprecated no-op. Use --object_offset_z instead.")
    parser.add_argument("--object_offset_x", type=float, default=0.0,
                        help="Fixed object-position offset in Sawyer base X, meters.")
    parser.add_argument("--object_offset_y", type=float, default=0.0,
                        help="Fixed object-position offset in Sawyer base Y, meters.")
    parser.add_argument("--object_offset_z", type=float, default=0.0,
                        help="Fixed object-position offset in Sawyer base Z, meters.")
    parser.add_argument("--replay_offset_x", type=float, default=0.0,
                        help="Rigid translation applied to every final replay waypoint X, meters.")
    parser.add_argument("--replay_offset_y", type=float, default=0.0,
                        help="Rigid translation applied to every final replay waypoint Y, meters.")
    parser.add_argument("--replay_offset_z", type=float, default=0.0,
                        help="Rigid translation applied to every final replay waypoint Z, meters.")
    parser.add_argument("--disable_vision_y_linear_calibration",
                        action="store_true",
                        help="Disable the relative linear Y vision calibration.")
    parser.add_argument("--enable_vision_y_linear_calibration",
                        action="store_true",
                        help="Enable the relative linear Y vision calibration.")
    parser.add_argument("--disable_vision_y_piecewise_compensation",
                        action="store_true",
                        help="Disable the piecewise Y vision compensation.")
    parser.add_argument("--move_to_start_pose", action="store_true",
                        help="Move Sawyer to the fixed simulation start "
                             "joint pose before trial timing.")
    parser.add_argument("--disable_move_to_start_pose", action="store_true",
                        help="Disable start-pose reset even if ROS param "
                             "~move_to_start_pose is true.")
    parser.add_argument("--demo_path", default="",
                        help="Recorded real top-grasp demo JSON path.")
    parser.add_argument("--trial_id", default="",
                        help="Experiment trial id.")
    args = parser.parse_args()

    dry_run = bool(args.dry_run or not args.execute)
    move_to_start_pose = None
    if bool(args.move_to_start_pose):
        move_to_start_pose = True
    if bool(args.disable_move_to_start_pose):
        move_to_start_pose = False
    robot = RealTopGraspReplay(
        demo_path=args.demo_path,
        trial_id=args.trial_id,
        replay_velocity_scale=args.replay_velocity_scale,
        pregrasp_clearance_m=args.pregrasp_clearance_m,
        grasp_offset_xyz=[
            args.grasp_offset_x,
            args.grasp_offset_y,
            args.grasp_offset_z,
        ],
        object_offset_xyz=[
            args.object_offset_x,
            args.object_offset_y,
            args.object_offset_z,
        ],
        replay_offset_xyz=[
            args.replay_offset_x,
            args.replay_offset_y,
            args.replay_offset_z,
        ],
        vision_y_linear_calibration_enabled=(
            bool(args.enable_vision_y_linear_calibration) and
            not bool(args.disable_vision_y_linear_calibration)),
        vision_y_piecewise_compensation_enabled=(
            not bool(args.disable_vision_y_piecewise_compensation)),
        move_to_start_pose=move_to_start_pose)
    ok = robot.run(
        dry_run=dry_run,
        update_perception=bool(args.update_perception),
        pregrasp_only=bool(args.pregrasp_only))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("mt3_sawyer_real_grasp failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)
