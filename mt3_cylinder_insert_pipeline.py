#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MT3-style generalization for inserting a cylinder into a shallow socket."""

import json
import os
import subprocess
import sys
import time
import csv
import math

import rospy

from mt3_demo_library import DemoLibrary
from mt3_anchor_perception import DualMaskAnchorPerception
from mt3_cylinder_insert_generalization import (
    DEFAULT_CYLINDER_SIZE,
    DEFAULT_SOCKET_PROFILE,
    compute_insert_target,
)
from mt3_relation_scene_package import save_dual_object_scene_packages


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CODE_DIR, "demo_library", "recorded")
ROLLOUT_DIR = os.path.join(CODE_DIR, "demo_library", "rollout_trajectories")
DEFAULT_INSERT_GRASP_FLANGE_Z_OFFSET = 0.050
INSERT_PIPELINE_DIAG_VERSION = "2026-08-17_joint_contact_perception_diag_v4"
SHARED_EXPERIMENT_LOG_DIR = (
    "/mnt/hgfs2/code/learning_thousand_tasks/demo_library/experiment_logs"
)
EXPERIMENT_LOG_DIR = (
    os.path.join(SHARED_EXPERIMENT_LOG_DIR, "vertical_insert")
    if os.path.isdir(os.path.dirname(SHARED_EXPERIMENT_LOG_DIR))
    else os.path.join(
        CODE_DIR, "demo_library", "experiment_logs", "vertical_insert")
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


def _forward_insert_motion_params():
    pairs = [
        ("~insert_descent_velocity_scale",
         "/sawyer_auto_grasp/insert_descent_velocity_scale", 0.025),
        ("~insert_descent_acceleration_scale",
         "/sawyer_auto_grasp/insert_descent_acceleration_scale", 0.025),
        ("~insert_descent_eef_step",
         "/sawyer_auto_grasp/insert_descent_eef_step", 0.002),
        ("~insert_descent_fallback_step_z",
         "/sawyer_auto_grasp/insert_descent_fallback_step_z", 0.006),
        ("~insert_descent_step_sleep",
         "/sawyer_auto_grasp/insert_descent_step_sleep", 0.35),
        ("~insert_replay_velocity_scale",
         "/sawyer_auto_grasp/insert_replay_velocity_scale", 0.025),
        ("~insert_replay_acceleration_scale",
         "/sawyer_auto_grasp/insert_replay_acceleration_scale", 0.025),
        ("~insert_replay_eef_step",
         "/sawyer_auto_grasp/insert_replay_eef_step", 0.002),
        ("~insert_replay_post_sleep",
         "/sawyer_auto_grasp/insert_replay_post_sleep", 0.40),
        ("~insert_replay_tracking_error_max_m",
         "/sawyer_auto_grasp/insert_replay_tracking_error_max_m", 0.020),
        ("~insert_transport_extra_clearance",
         "/sawyer_auto_grasp/insert_transport_extra_clearance", 0.030),
        ("~insert_transport_xy_step",
         "/sawyer_auto_grasp/insert_transport_xy_step", 0.025),
        ("~insert_transport_eef_step",
         "/sawyer_auto_grasp/insert_transport_eef_step", 0.004),
        ("~insert_transport_velocity_scale",
         "/sawyer_auto_grasp/insert_transport_velocity_scale", 0.060),
        ("~insert_transport_acceleration_scale",
         "/sawyer_auto_grasp/insert_transport_acceleration_scale", 0.060),
        ("~insert_transport_post_sleep",
         "/sawyer_auto_grasp/insert_transport_post_sleep", 0.30),
    ]
    for src, dst, default in pairs:
        rospy.set_param(dst, float(rospy.get_param(src, default)))


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



def _reset_executor_status_params():
    defaults = [
        ("/sawyer_auto_grasp/grasp_replay_attempted", False),
        ("/sawyer_auto_grasp/grasp_replay_success", ""),
        ("/sawyer_auto_grasp/grasp_replay_stage", ""),
        ("/sawyer_auto_grasp/grasp_replay_failure_stage", ""),
        ("/sawyer_auto_grasp/insertion_replay_attempted", False),
        ("/sawyer_auto_grasp/insertion_replay_success", ""),
        ("/sawyer_auto_grasp/insertion_replay_stage", ""),
        ("/sawyer_auto_grasp/insertion_replay_failure_stage", ""),
        ("/sawyer_auto_grasp/insertion_interaction_success", False),
        ("/sawyer_auto_grasp/post_release_retreat_attempted", False),
        ("/sawyer_auto_grasp/post_release_retreat_success", ""),
        ("/sawyer_auto_grasp/scripted_fallback_used", False),
        ("/sawyer_auto_grasp/pure_replay_success", False),
        ("/sawyer_auto_grasp/failure_stage_detail", ""),
        ("/sawyer_auto_grasp/replay_failure_stage_detail", ""),
        ("/sawyer_auto_grasp/grasp_post_close_motion_max_m", ""),
        ("/sawyer_auto_grasp/grasp_post_close_motion_max_xy_m", ""),
        ("/sawyer_auto_grasp/grasp_post_close_motion_max_z_m", ""),
        ("/sawyer_auto_grasp/grasp_post_close_mode", ""),
        ("/sawyer_auto_grasp/grasp_post_close_dwell_s", ""),
        ("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_x_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_y_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_xy_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_x_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_y_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_xy_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_x_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_y_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_xy_m", ""),
        ("/sawyer_auto_grasp/diag_grasp_complete_before_transport_cylinder_hand_offset_xy_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_x_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_y_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_xy_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_offset_x_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_offset_y_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_replay_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_hand_tracking_error_xyz_m", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_hand_tracking_error_xyz_m", ""),
        ("/sawyer_auto_grasp/diag_post_release_cylinder_socket_error_xy_m", ""),
        ("/sawyer_auto_grasp/insert_tracking_failure", ""),
        ("/sawyer_auto_grasp/insert_tracking_failure_chunk", ""),
        ("/sawyer_auto_grasp/insert_tracking_failure_error_m", ""),
        ("/sawyer_auto_grasp/insert_replay_tracking_error_max_m_active", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_translational_jacobian_condition_number", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_translational_jacobian_sigma_min", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_vertical_joint_velocity_gain", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_joint_limit_min_margin_normalized", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_speed_xy_m_s", ""),
        ("/sawyer_auto_grasp/diag_pre_step_f_cylinder_tilt_deg", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_translational_jacobian_condition_number", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_translational_jacobian_condition_number", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_translational_jacobian_condition_number", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_translational_jacobian_sigma_min", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_translational_jacobian_sigma_min", ""),
        ("/sawyer_auto_grasp/diag_step_f_min_translational_jacobian_sigma_min", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_vertical_joint_velocity_gain", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_vertical_joint_velocity_gain", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_vertical_joint_velocity_gain", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_joint_limit_min_margin_normalized", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_joint_limit_min_margin_normalized", ""),
        ("/sawyer_auto_grasp/diag_step_f_min_joint_limit_min_margin_normalized", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_cylinder_speed_xy_m_s", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_cylinder_speed_xy_m_s", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_cylinder_speed_xy_m_s", ""),
        ("/sawyer_auto_grasp/diag_step_f_first_cylinder_tilt_deg", ""),
        ("/sawyer_auto_grasp/diag_step_f_last_cylinder_tilt_deg", ""),
        ("/sawyer_auto_grasp/diag_step_f_max_cylinder_tilt_deg", ""),
    ]
    for name, value in defaults:
        try:
            rospy.set_param(name, value)
        except Exception:
            pass


def _read_executor_status_params():
    def _value(name, default=""):
        try:
            return rospy.get_param(name, default)
        except Exception:
            return default

    return {
        "grasp_replay_attempted": _value(
            "/sawyer_auto_grasp/grasp_replay_attempted", False),
        "grasp_replay_success": _value(
            "/sawyer_auto_grasp/grasp_replay_success", ""),
        "grasp_replay_stage": _value(
            "/sawyer_auto_grasp/grasp_replay_stage", ""),
        "grasp_replay_failure_stage": _value(
            "/sawyer_auto_grasp/grasp_replay_failure_stage", ""),
        "insertion_replay_attempted": _value(
            "/sawyer_auto_grasp/insertion_replay_attempted", False),
        "insertion_replay_success": _value(
            "/sawyer_auto_grasp/insertion_replay_success", ""),
        "insertion_replay_stage": _value(
            "/sawyer_auto_grasp/insertion_replay_stage", ""),
        "insertion_replay_failure_stage": _value(
            "/sawyer_auto_grasp/insertion_replay_failure_stage", ""),
        "insertion_interaction_success": _value(
            "/sawyer_auto_grasp/insertion_interaction_success", False),
        "post_release_retreat_attempted": _value(
            "/sawyer_auto_grasp/post_release_retreat_attempted", False),
        "post_release_retreat_success": _value(
            "/sawyer_auto_grasp/post_release_retreat_success", ""),
        "scripted_fallback_used": _value(
            "/sawyer_auto_grasp/scripted_fallback_used", False),
        "pure_replay_success": _value(
            "/sawyer_auto_grasp/pure_replay_success", False),
        "failure_stage_detail": _value(
            "/sawyer_auto_grasp/failure_stage_detail", ""),
        "replay_failure_stage_detail": _value(
            "/sawyer_auto_grasp/replay_failure_stage_detail", ""),
        "grasp_post_close_motion_max_m": _value(
            "/sawyer_auto_grasp/grasp_post_close_motion_max_m", ""),
        "grasp_post_close_motion_max_xy_m": _value(
            "/sawyer_auto_grasp/grasp_post_close_motion_max_xy_m", ""),
        "grasp_post_close_motion_max_z_m": _value(
            "/sawyer_auto_grasp/grasp_post_close_motion_max_z_m", ""),
        "grasp_post_close_mode": _value(
            "/sawyer_auto_grasp/grasp_post_close_mode", ""),
        "grasp_post_close_dwell_s": _value(
            "/sawyer_auto_grasp/grasp_post_close_dwell_s", ""),
        "diag_grasp_before_close_cylinder_hand_offset_x_m": _value("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_x_m", ""),
        "diag_grasp_before_close_cylinder_hand_offset_y_m": _value("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_y_m", ""),
        "diag_grasp_before_close_cylinder_hand_offset_xy_m": _value("/sawyer_auto_grasp/diag_grasp_before_close_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_x_m": _value("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_x_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_y_m": _value("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_y_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_xy_m": _value("/sawyer_auto_grasp/diag_grasp_after_close_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_x_m": _value("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_x_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_y_m": _value("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_y_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_xy_m": _value("/sawyer_auto_grasp/diag_grasp_after_lift_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_complete_before_transport_cylinder_hand_offset_xy_m": _value("/sawyer_auto_grasp/diag_grasp_complete_before_transport_cylinder_hand_offset_xy_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_x_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_x_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_y_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_y_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_xy_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_hand_offset_xy_m", ""),
        "diag_pre_step_f_cylinder_socket_offset_x_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_offset_x_m", ""),
        "diag_pre_step_f_cylinder_socket_offset_y_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_offset_y_m", ""),
        "diag_pre_step_f_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_socket_error_xy_m", ""),
        "diag_pre_step_f_replay_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_pre_step_f_replay_cylinder_socket_error_xy_m", ""),
        "diag_step_f_first_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_step_f_first_cylinder_socket_error_xy_m", ""),
        "diag_step_f_last_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_step_f_last_cylinder_socket_error_xy_m", ""),
        "diag_step_f_max_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_step_f_max_cylinder_socket_error_xy_m", ""),
        "diag_step_f_last_hand_tracking_error_xyz_m": _value("/sawyer_auto_grasp/diag_step_f_last_hand_tracking_error_xyz_m", ""),
        "diag_step_f_max_hand_tracking_error_xyz_m": _value("/sawyer_auto_grasp/diag_step_f_max_hand_tracking_error_xyz_m", ""),
        "diag_post_release_cylinder_socket_error_xy_m": _value("/sawyer_auto_grasp/diag_post_release_cylinder_socket_error_xy_m", ""),
        "insert_tracking_failure": _value("/sawyer_auto_grasp/insert_tracking_failure", ""),
        "insert_tracking_failure_chunk": _value("/sawyer_auto_grasp/insert_tracking_failure_chunk", ""),
        "insert_tracking_failure_error_m": _value("/sawyer_auto_grasp/insert_tracking_failure_error_m", ""),
        "insert_replay_tracking_error_max_m_active": _value("/sawyer_auto_grasp/insert_replay_tracking_error_max_m_active", ""),
        "diag_pre_step_f_translational_jacobian_condition_number": _value("/sawyer_auto_grasp/diag_pre_step_f_translational_jacobian_condition_number", ""),
        "diag_pre_step_f_translational_jacobian_sigma_min": _value("/sawyer_auto_grasp/diag_pre_step_f_translational_jacobian_sigma_min", ""),
        "diag_pre_step_f_vertical_joint_velocity_gain": _value("/sawyer_auto_grasp/diag_pre_step_f_vertical_joint_velocity_gain", ""),
        "diag_pre_step_f_joint_limit_min_margin_normalized": _value("/sawyer_auto_grasp/diag_pre_step_f_joint_limit_min_margin_normalized", ""),
        "diag_pre_step_f_cylinder_speed_xy_m_s": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_speed_xy_m_s", ""),
        "diag_pre_step_f_cylinder_tilt_deg": _value("/sawyer_auto_grasp/diag_pre_step_f_cylinder_tilt_deg", ""),
        "diag_step_f_first_translational_jacobian_condition_number": _value("/sawyer_auto_grasp/diag_step_f_first_translational_jacobian_condition_number", ""),
        "diag_step_f_last_translational_jacobian_condition_number": _value("/sawyer_auto_grasp/diag_step_f_last_translational_jacobian_condition_number", ""),
        "diag_step_f_max_translational_jacobian_condition_number": _value("/sawyer_auto_grasp/diag_step_f_max_translational_jacobian_condition_number", ""),
        "diag_step_f_first_translational_jacobian_sigma_min": _value("/sawyer_auto_grasp/diag_step_f_first_translational_jacobian_sigma_min", ""),
        "diag_step_f_last_translational_jacobian_sigma_min": _value("/sawyer_auto_grasp/diag_step_f_last_translational_jacobian_sigma_min", ""),
        "diag_step_f_min_translational_jacobian_sigma_min": _value("/sawyer_auto_grasp/diag_step_f_min_translational_jacobian_sigma_min", ""),
        "diag_step_f_first_vertical_joint_velocity_gain": _value("/sawyer_auto_grasp/diag_step_f_first_vertical_joint_velocity_gain", ""),
        "diag_step_f_last_vertical_joint_velocity_gain": _value("/sawyer_auto_grasp/diag_step_f_last_vertical_joint_velocity_gain", ""),
        "diag_step_f_max_vertical_joint_velocity_gain": _value("/sawyer_auto_grasp/diag_step_f_max_vertical_joint_velocity_gain", ""),
        "diag_step_f_first_joint_limit_min_margin_normalized": _value("/sawyer_auto_grasp/diag_step_f_first_joint_limit_min_margin_normalized", ""),
        "diag_step_f_last_joint_limit_min_margin_normalized": _value("/sawyer_auto_grasp/diag_step_f_last_joint_limit_min_margin_normalized", ""),
        "diag_step_f_min_joint_limit_min_margin_normalized": _value("/sawyer_auto_grasp/diag_step_f_min_joint_limit_min_margin_normalized", ""),
        "diag_step_f_first_cylinder_speed_xy_m_s": _value("/sawyer_auto_grasp/diag_step_f_first_cylinder_speed_xy_m_s", ""),
        "diag_step_f_last_cylinder_speed_xy_m_s": _value("/sawyer_auto_grasp/diag_step_f_last_cylinder_speed_xy_m_s", ""),
        "diag_step_f_max_cylinder_speed_xy_m_s": _value("/sawyer_auto_grasp/diag_step_f_max_cylinder_speed_xy_m_s", ""),
        "diag_step_f_first_cylinder_tilt_deg": _value("/sawyer_auto_grasp/diag_step_f_first_cylinder_tilt_deg", ""),
        "diag_step_f_last_cylinder_tilt_deg": _value("/sawyer_auto_grasp/diag_step_f_last_cylinder_tilt_deg", ""),
        "diag_step_f_max_cylinder_tilt_deg": _value("/sawyer_auto_grasp/diag_step_f_max_cylinder_tilt_deg", ""),
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
    if any(k in text for k in [
            "insert", "socket", "descent", "depth", "rim",
            "verification", "postcheck"]):
        return "insertion_failure"
    return "other_execution_failure"


def _insert_failure_detail(failure_category, reason_text, target_error_xy, socket_error_xy):
    text = str(reason_text or "").lower()
    if not failure_category:
        return ""
    try:
        if target_error_xy != "" and float(target_error_xy) > 0.025:
            return "cylinder_misaligned"
    except Exception:
        pass
    try:
        if socket_error_xy != "" and float(socket_error_xy) > 0.025:
            return "socket_misaligned"
    except Exception:
        pass
    if any(k in text for k in ["z error", "descent", "depth", "too high"]):
        return "insufficient_insert_depth"
    if any(k in text for k in ["socket", "rim", "collision", "path_tolerance", "control_failed", "aborted"]):
        return "socket_rim_collision_or_controller_abort"
    if "planning" in text or "no motion plan" in text or "plan failed" in text:
        return "motion_planning_failure"
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


def _param_float_list(name, default_value):
    value = rospy.get_param(name, default_value)
    return [float(v) for v in value]


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


def _geometry_method(entry):
    try:
        return (((entry or {}).get("pose_base") or {})
                .get("geometry_center_correction") or {}).get("method", "")
    except Exception:
        return ""


def _safe_float(value, default=""):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def _safe_xyz(values):
    if values is None:
        return None
    try:
        if len(values) < 3:
            return None
        xyz = [float(values[0]), float(values[1]), float(values[2])]
        if not all(math.isfinite(v) for v in xyz):
            return None
        return xyz
    except Exception:
        return None


def _percentile(values, percentile):
    """Small dependency-free linear percentile helper for diagnostics only."""
    vals = []
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value):
                vals.append(value)
        except Exception:
            continue
    if not vals:
        return ""
    vals.sort()
    if len(vals) == 1:
        return vals[0]
    p = min(100.0, max(0.0, float(percentile))) / 100.0
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    alpha = pos - lo
    return vals[lo] * (1.0 - alpha) + vals[hi] * alpha


def _target_perception_diagnostics(target):
    """Extract read-only target point-cloud / geometry-center diagnostics.

    This function deliberately does NOT modify the perception result used by
    grasping or trajectory transfer.  It only summarizes data already present
    in the DualMaskAnchorPerception scene dictionary.
    """
    target = target or {}
    pose_source = target.get("pose_source") or {}
    pose_base = target.get("pose_base") or {}
    correction = pose_base.get("geometry_center_correction") or {}

    points = pose_source.get("object_points", None)
    if points is None:
        points = target.get("object_points", None)
    point_rows = []
    if points is not None:
        try:
            for point in points:
                xyz = _safe_xyz(point)
                if xyz is not None:
                    point_rows.append(xyz)
        except Exception:
            point_rows = []

    diag = {
        "version": "perception_bias_diag_v4",
        "source_frame": str(pose_source.get("source_frame", "") or ""),
        "source_point_count": len(point_rows),
        "geometry_method": str(correction.get("method", "") or ""),
        "raw_center_base_xyz": _safe_xyz(correction.get("old_position")),
        "corrected_center_base_xyz": _safe_xyz(
            correction.get("new_position")) or _safe_xyz(target.get("position_base")),
        "geometry_correction_delta_xyz": _safe_xyz(correction.get("delta_xyz")),
        "base_bounds_low_xyz": _safe_xyz(correction.get("bounds_low_xyz")),
        "base_bounds_high_xyz": _safe_xyz(correction.get("bounds_high_xyz")),
    }

    if point_rows:
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            vals = [p[axis_idx] for p in point_rows]
            for label, pct in [
                    ("min", 0), ("p05", 5), ("p10", 10), ("p25", 25),
                    ("p50", 50), ("p75", 75), ("p90", 90),
                    ("p95", 95), ("max", 100)]:
                diag["source_%s_%s_m" % (axis_name, label)] = _percentile(vals, pct)

    low = diag.get("base_bounds_low_xyz")
    high = diag.get("base_bounds_high_xyz")
    if low is not None and high is not None:
        diag["base_bounds_mid_xyz"] = [
            0.5 * (low[i] + high[i]) for i in range(3)]
        diag["base_bounds_extent_xyz"] = [
            high[i] - low[i] for i in range(3)]
    else:
        diag["base_bounds_mid_xyz"] = None
        diag["base_bounds_extent_xyz"] = None
    return diag


def _attach_and_log_target_perception_diagnostics(scene):
    """Attach diagnostics to scene metadata and print concise terminal lines."""
    target = (scene or {}).get("target") or {}
    diag = _target_perception_diagnostics(target)
    target["perception_bias_diagnostics_v4"] = diag

    rospy.loginfo(
        "TARGET PERCEPTION DIAG v4: frame=%s points=%d geometry_method=%s",
        diag.get("source_frame", ""),
        int(diag.get("source_point_count", 0) or 0),
        diag.get("geometry_method", ""))
    if diag.get("source_point_count", 0):
        rospy.loginfo(
            "TARGET PC source X [min p05 p10 p50 p90 p95 max]="
            "[%.4f %.4f %.4f %.4f %.4f %.4f %.4f]",
            diag.get("source_x_min_m", float("nan")),
            diag.get("source_x_p05_m", float("nan")),
            diag.get("source_x_p10_m", float("nan")),
            diag.get("source_x_p50_m", float("nan")),
            diag.get("source_x_p90_m", float("nan")),
            diag.get("source_x_p95_m", float("nan")),
            diag.get("source_x_max_m", float("nan")))
        rospy.loginfo(
            "TARGET PC source Y [min p05 p10 p50 p90 p95 max]="
            "[%.4f %.4f %.4f %.4f %.4f %.4f %.4f]",
            diag.get("source_y_min_m", float("nan")),
            diag.get("source_y_p05_m", float("nan")),
            diag.get("source_y_p10_m", float("nan")),
            diag.get("source_y_p50_m", float("nan")),
            diag.get("source_y_p90_m", float("nan")),
            diag.get("source_y_p95_m", float("nan")),
            diag.get("source_y_max_m", float("nan")))

    raw = diag.get("raw_center_base_xyz")
    corrected = diag.get("corrected_center_base_xyz")
    delta = diag.get("geometry_correction_delta_xyz")
    low = diag.get("base_bounds_low_xyz")
    high = diag.get("base_bounds_high_xyz")
    mid = diag.get("base_bounds_mid_xyz")
    if raw is not None or corrected is not None:
        rospy.loginfo(
            "TARGET center diag base: raw=%s corrected=%s correction_mm=%s",
            "[%.4f %.4f %.4f]" % tuple(raw) if raw is not None else "n/a",
            "[%.4f %.4f %.4f]" % tuple(corrected) if corrected is not None else "n/a",
            "[%.1f %.1f %.1f]" % tuple(v * 1000.0 for v in delta)
            if delta is not None else "n/a")
    if low is not None and high is not None:
        rospy.loginfo(
            "TARGET geometry bounds base: low=[%.4f %.4f %.4f] "
            "high=[%.4f %.4f %.4f] midpoint=[%.4f %.4f %.4f]",
            low[0], low[1], low[2], high[0], high[1], high[2],
            mid[0], mid[1], mid[2])
    return diag


def _xy_delta(est, gt):
    try:
        if est is None or gt is None:
            return "", "", ""
        dx = float(est[0]) - float(gt[0])
        dy = float(est[1]) - float(gt[1])
        return dx, dy, math.sqrt(dx * dx + dy * dy)
    except Exception:
        return "", "", ""


def _gazebo_pose(model_param, fallback_keywords):
    try:
        from gazebo_msgs.msg import ModelStates
        explicit = str(rospy.get_param(model_param, "")).strip()
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=0.5)
        names = list(msg.name)
        chosen = None
        if explicit and explicit in names:
            chosen = explicit
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
            return None
        pose = msg.pose[names.index(chosen)]
        return {
            "name": chosen,
            "xyz": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
        }
    except Exception:
        return None


def _relation_xy_error(target_xyz, anchor_xyz, desired_offset_xyz):
    try:
        if target_xyz is None or anchor_xyz is None or desired_offset_xyz is None:
            return ""
        dx = (float(target_xyz[0]) - float(anchor_xyz[0])) - float(desired_offset_xyz[0])
        dy = (float(target_xyz[1]) - float(anchor_xyz[1])) - float(desired_offset_xyz[1])
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return ""


def _validate_post_insert_success(insert_result, cylinder_size, socket_profile,
                                  initial_socket_xyz):
    """Verify final cylinder/socket relation and approximate insertion depth."""
    target_final = _gazebo_pose("~target_gt_model", ["green", "cylinder"])
    socket_final = _gazebo_pose("~socket_gt_model", ["blue", "socket"])
    target_xyz = (target_final or {}).get("xyz")
    socket_xyz = (socket_final or {}).get("xyz") or initial_socket_xyz
    insert_xyz = insert_result.get("place_xyz", [])
    desired_offset = insert_result.get("offset_xyz", [])
    if target_xyz is None or socket_xyz is None:
        rospy.logwarn(
            "POST-INSERT CHECK: Gazebo cylinder/socket pose unavailable; "
            "keeping executor result.")
        return {
            "ok": True,
            "failure_stage": "",
            "failure_reason": "",
            "postcheck_success": "",
            "postcheck_reason": "gazebo_pose_unavailable",
            "final_object_model_name": (target_final or {}).get("name", ""),
            "final_socket_model_name": (socket_final or {}).get("name", ""),
            "final_object_xyz": target_xyz,
            "final_socket_xyz": socket_xyz,
            "final_target_error_xy_m": "",
            "final_relation_error_xy_m": "",
            "insert_depth_m": "",
        }

    final_target_error = _xy_error(target_xyz, insert_xyz)
    final_relation_error = _relation_xy_error(target_xyz, socket_xyz, desired_offset)
    opening = socket_profile.get("opening_m") or DEFAULT_SOCKET_PROFILE["opening_m"]
    default_xy_limit = max(0.010, min(float(opening[0]), float(opening[1])) * 0.45)
    max_xy_error = float(rospy.get_param(
        "~post_insert_max_xy_error_m", default_xy_limit))

    insert_depth = ""
    try:
        cylinder_height = float(cylinder_size[2])
        socket_height = float((socket_profile.get("size_m") or DEFAULT_SOCKET_PROFILE["size_m"])[2])
        socket_top_z = float(socket_xyz[2]) + 0.5 * socket_height
        cylinder_bottom_z = float(target_xyz[2]) - 0.5 * cylinder_height
        insert_depth = max(0.0, socket_top_z - cylinder_bottom_z)
    except Exception:
        insert_depth = ""
    min_depth = float(rospy.get_param("~post_insert_min_depth_m", 0.010))

    xy_basis = final_relation_error if final_relation_error != "" else final_target_error
    xy_ok = xy_basis != "" and float(xy_basis) <= max_xy_error
    depth_ok = insert_depth != "" and float(insert_depth) >= min_depth
    ok = bool(xy_ok and depth_ok)
    if ok:
        reason = ""
    elif not xy_ok:
        reason = "圆柱与插孔最终XY对准误差过大"
    else:
        reason = "圆柱插入深度不足"

    rospy.loginfo(
        "POST-INSERT CHECK: target_err=%s relation_err=%s max_xy=%.3f "
        "depth=%s min_depth=%.3f ok=%s",
        ("%.3f" % final_target_error if final_target_error != "" else "n/a"),
        ("%.3f" % final_relation_error if final_relation_error != "" else "n/a"),
        max_xy_error,
        ("%.3f" % insert_depth if insert_depth != "" else "n/a"),
        min_depth,
        ok)
    return {
        "ok": ok,
        "failure_stage": "" if ok else "insertion_verification",
        "failure_reason": reason,
        "postcheck_success": ok,
        "postcheck_reason": reason,
        "final_object_model_name": (target_final or {}).get("name", ""),
        "final_socket_model_name": (socket_final or {}).get("name", ""),
        "final_object_xyz": target_xyz,
        "final_socket_xyz": socket_xyz,
        "final_target_error_xy_m": final_target_error,
        "final_relation_error_xy_m": final_relation_error,
        "insert_depth_m": insert_depth,
    }


def _event_sample_index(events, names):
    for event in events or []:
        if event.get("name") in names:
            try:
                return int(event.get("sample_index"))
            except Exception:
                return None
    return None


def _rollout_insert_process_metrics(rollout_path, cylinder_size, socket_profile):
    """Measure insertion-window relation error from the saved rollout.

    Gazebo contacts are not available here, so rim contact is an inferred
    diagnostic: during the insertion descent, if the cylinder axis deviates
    from the socket axis beyond the usable radial clearance, flag it.
    """
    empty = {
        "max_insert_relation_error_xy_m": "",
        "rim_contact_or_collision_flag": "",
        "rim_contact_xy_threshold_m": "",
    }
    if not rollout_path or not os.path.exists(rollout_path):
        return empty
    try:
        with open(rollout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return empty

    samples = data.get("poses") or []
    if not samples:
        return empty
    events = data.get("events") or []
    start_idx = _event_sample_index(events, ["insert_replay_start"])
    end_idx = _event_sample_index(events, ["insert_release_open"])
    if end_idx is None:
        end_idx = _event_sample_index(events, ["insert_replay_end"])
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx < start_idx:
        end_idx = len(samples) - 1

    errors = []
    for sample in samples[start_idx:end_idx + 1]:
        value = sample.get("target_socket_relation_error_xy_m")
        if value is None:
            target_xyz = sample.get("target_model_xyz")
            socket_xyz = sample.get("socket_model_xyz")
            offset_xy = data.get("desired_relation_offset_xy") or [0.0, 0.0]
            if target_xyz is None or socket_xyz is None:
                continue
            try:
                dx = (
                    float(target_xyz[0]) - float(socket_xyz[0]) -
                    float(offset_xy[0]))
                dy = (
                    float(target_xyz[1]) - float(socket_xyz[1]) -
                    float(offset_xy[1]))
                value = math.sqrt(dx * dx + dy * dy)
            except Exception:
                continue
        try:
            errors.append(float(value))
        except Exception:
            pass

    if not errors:
        return empty
    max_error = max(errors)
    try:
        opening = socket_profile.get("opening_m") or DEFAULT_SOCKET_PROFILE["opening_m"]
        cyl_diameter = max(float(cylinder_size[0]), float(cylinder_size[1]))
        radial_clearance = (
            min(float(opening[0]), float(opening[1])) - cyl_diameter) * 0.5
    except Exception:
        radial_clearance = 0.005
    default_threshold = max(0.0025, radial_clearance * 0.80)
    threshold = float(rospy.get_param(
        "~rim_contact_xy_threshold_m", default_threshold))
    return {
        "max_insert_relation_error_xy_m": max_error,
        "rim_contact_or_collision_flag": bool(max_error >= threshold),
        "rim_contact_xy_threshold_m": threshold,
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
    cylinder_size = rospy.get_param("~cylinder_size", DEFAULT_CYLINDER_SIZE)
    socket_size = _param_float_list(
        "~socket_size", DEFAULT_SOCKET_PROFILE["size_m"])
    if demo is not None and not rospy.has_param("~anchor_plane_z"):
        demo_socket = _demo_position(demo, "anchor_info", fallback=None)
        if demo_socket is not None and len(demo_socket) >= 3:
            rospy.set_param("~anchor_plane_z", float(demo_socket[2]))
    if _param_bool("~use_perception", True):
        detector = DualMaskAnchorPerception(
            target_mask_path=rospy.get_param(
                "~target_mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy"),
            anchor_mask_path=rospy.get_param(
                "~socket_mask_path", "/mnt/hgfs2/tmp_vision/current_anchor_mask.npy"),
            target_size=[float(v) for v in cylinder_size],
            anchor_size=socket_size,
        )
        scene = detector.detect_scene(timeout_s=float(rospy.get_param(
            "~perception_timeout_s", 8.0)))
        if scene is None:
            raise RuntimeError("cylinder/socket perception failed")
        _attach_and_log_target_perception_diagnostics(scene)
        cylinder_xyz = scene["target"]["position_base"]
        socket_xyz = scene["anchor"]["position_base"]
        return scene, cylinder_xyz, socket_xyz, cylinder_size

    cylinder_xyz = _param_xyz("target", [0.60, 0.00, -0.58])
    socket_xyz = _param_xyz("socket", [0.60, -0.18, -0.58])
    scene = {
        "target": {"position_base": cylinder_xyz, "method": "manual_param"},
        "socket": {"position_base": socket_xyz, "method": "manual_param"},
    }
    return scene, cylinder_xyz, socket_xyz, cylinder_size


def _demo_position(demo, key, fallback=None):
    block = demo.get(key, {}) or {}
    if "position_base" in block:
        return [float(v) for v in block["position_base"][:3]]
    if "position" in block:
        return [float(v) for v in block["position"][:3]]
    pos_m = block.get("position_m", {})
    if pos_m:
        return [float(pos_m["x"]), float(pos_m["y"]), float(pos_m["z"])]
    if fallback is not None:
        return [float(v) for v in fallback[:3]]
    return None



def _demo_size(demo, key="object_info", fallback=None):
    block = (demo or {}).get(key, {}) or {}
    for field in ("size_m", "size", "dimensions_m", "dimensions"):
        value = block.get(field)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return [float(v) for v in value[:3]]
    geom = (demo or {}).get("geometric_features", {}) or {}
    for field in ("dimensions_m", "dimensions", "size_m", "size"):
        value = geom.get(field)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return [float(v) for v in value[:3]]
    if fallback is not None:
        return [float(v) for v in fallback[:3]]
    return None


def _stabilize_cylinder_z_from_demo_relation(demo, scene, cylinder_xyz,
                                              socket_xyz):
    """Stabilize live cylinder Z from the selected demo target-anchor relation.

    Perception XY is preserved.  Only Z is replaced by the live socket Z plus
    the selected demo's cylinder-to-socket Z relation, with a safety clamp.
    """
    if not _param_bool("~stabilize_cylinder_z_from_demo_relation", True):
        return [float(v) for v in cylinder_xyz]
    demo_cylinder = _demo_position(demo, "object_info", fallback=None)
    demo_socket = _demo_position(demo, "anchor_info", fallback=None)
    if demo_cylinder is None or demo_socket is None or socket_xyz is None:
        rospy.logwarn(
            "Insert cylinder Z stabilization skipped: demo/live relation missing")
        return [float(v) for v in cylinder_xyz]

    raw_z = float(cylinder_xyz[2])
    demo_rel_z = float(demo_cylinder[2]) - float(demo_socket[2])
    corrected_z = float(socket_xyz[2]) + demo_rel_z
    correction = corrected_z - raw_z
    max_correction = abs(float(rospy.get_param(
        "~max_cylinder_z_stabilization_m", 0.020)))
    if abs(correction) > max_correction:
        rospy.logwarn(
            "Insert cylinder Z stabilization rejected: correction=%.2fmm "
            "exceeds max=%.2fmm",
            correction * 1000.0, max_correction * 1000.0)
        return [float(v) for v in cylinder_xyz]

    corrected = [
        float(cylinder_xyz[0]),
        float(cylinder_xyz[1]),
        corrected_z,
    ]
    try:
        target = (scene or {}).get("target", {})
        target["z_stabilization"] = {
            "enabled": True,
            "raw_z_m": raw_z,
            "corrected_z_m": corrected_z,
            "correction_m": correction,
            "demo_target_anchor_rel_z_m": demo_rel_z,
            "socket_z_m": float(socket_xyz[2]),
            "method": "selected_demo_target_anchor_z_relation",
        }
        target["position_base_raw_before_z_stabilization"] = [
            float(v) for v in cylinder_xyz]
        target["position_base"] = [float(v) for v in corrected]
    except Exception:
        pass

    rospy.loginfo(
        "Insert cylinder Z stabilized from demo target-anchor relation: "
        "raw=%.6f socket=%.6f demo_rel_z=%.6f -> %.6f correction=%+.2fmm",
        raw_z, float(socket_xyz[2]), demo_rel_z, corrected_z,
        correction * 1000.0)
    return corrected


def _place_xyz_from_demo(demo):
    place_info = (demo or {}).get("place_info", {}) or {}
    place_xyz = (
        place_info.get("place_xyz")
        or (place_info.get("place_pose_base_frame") or {}).get("position")
    )
    if place_xyz and len(place_xyz) >= 3:
        return [float(v) for v in place_xyz[:3]]
    return None


def _orientation_from_pose_payload(pose):
    pose = pose or {}
    ori = pose.get("orientation")
    if isinstance(ori, list) and len(ori) >= 4:
        return [float(v) for v in ori[:4]]
    ori = pose.get("orientation_xyzw")
    if isinstance(ori, dict):
        return [
            float(ori.get("x", 0.0)),
            float(ori.get("y", 0.0)),
            float(ori.get("z", 0.0)),
            float(ori.get("w", 1.0)),
        ]
    return [0.0, 0.0, 0.0, 1.0]


def _mapped_insertion_bottleneck_pose(demo, insert_result):
    demo_insert = _place_xyz_from_demo(demo)
    demo_bn = _demo_position(demo, "insertion_bottleneck_pose_base_frame")
    live_insert = (insert_result or {}).get("place_xyz")
    if demo_insert is None or demo_bn is None or not live_insert:
        return {}
    mapped = [
        float(live_insert[0]) + (float(demo_bn[0]) - float(demo_insert[0])),
        float(live_insert[1]) + (float(demo_bn[1]) - float(demo_insert[1])),
        float(live_insert[2]) + (float(demo_bn[2]) - float(demo_insert[2])),
    ]
    return {
        "position": mapped,
        "orientation": _orientation_from_pose_payload(
            (demo or {}).get("insertion_bottleneck_pose_base_frame", {})),
        "relative_to_demo_insert": [
            float(demo_bn[0]) - float(demo_insert[0]),
            float(demo_bn[1]) - float(demo_insert[1]),
            float(demo_bn[2]) - float(demo_insert[2]),
        ],
        "mapping_method": "live_insert_plus_demo_insertion_bottleneck_offset",
    }


def _mapped_grasp_close_pose(demo, grasp):
    """Map live grasp close while preserving the demonstrated close-Z relation.

    XY stays anchored to the live semantic grasp, matching the previous behavior.
    Z uses the *recorded actual* demo right_hand close relative to the demo
    semantic grasp. This avoids reconstructing the close pose from the fixed
    expected flange offset when an actual recorded close pose is available.
    """
    live_grasp = (grasp or {}).get("position")
    if not live_grasp:
        return {}

    demo_grasp, _ = _demo_grasp_pose(demo)
    demo_close = _demo_position(
        demo, "grasp_close_pose_base_frame", fallback=None)

    if demo_close is None:
        z_offset = None
        close_z = (
            float(live_grasp[2]) +
            float(rospy.get_param(
                "~insert_grasp_flange_z_offset",
                DEFAULT_INSERT_GRASP_FLANGE_Z_OFFSET))
        )
        mapping_method = "fixed_flange_z_offset_fallback"
        rospy.logwarn(
            "Demo has no recorded grasp close pose; falling back to fixed "
            "insert_grasp_flange_z_offset for close Z.")
    else:
        z_offset = float(demo_close[2]) - float(demo_grasp[2])
        close_z = float(live_grasp[2]) + z_offset
        mapping_method = "demo_actual_close_relative_to_semantic_grasp"

    rospy.loginfo(
        "Insert grasp close Z mapped: live_grasp_z=%.6f close_z=%.6f "
        "demo_actual_offset=%s method=%s",
        float(live_grasp[2]),
        close_z,
        ("%.2fmm" % (z_offset * 1000.0))
        if z_offset is not None else "fallback",
        mapping_method)

    return {
        "position": [
            float(live_grasp[0]),
            float(live_grasp[1]),
            close_z,
        ],
        "orientation": [float(v) for v in (
            (grasp or {}).get("orientation") or [-1.0, 0.0, 0.0, 0.0])],
        "frame": "base",
        "mapping_method": mapping_method,
    }


def _mapped_grasp_bottleneck_pose(demo, grasp):
    demo_bn = _demo_position(demo, "grasp_bottleneck_pose_base_frame")
    demo_close = _demo_position(demo, "grasp_close_pose_base_frame")
    if demo_bn is None or demo_close is None or not grasp:
        return {}

    mapped_close = _mapped_grasp_close_pose(demo, grasp)
    if not mapped_close:
        return {}
    live_close = mapped_close["position"]

    mapped = [
        float(live_close[0]) + (float(demo_bn[0]) - float(demo_close[0])),
        float(live_close[1]) + (float(demo_bn[1]) - float(demo_close[1])),
        float(live_close[2]) + (float(demo_bn[2]) - float(demo_close[2])),
    ]
    return {
        "position": mapped,
        "orientation": _orientation_from_pose_payload(
            (demo or {}).get("grasp_bottleneck_pose_base_frame", {})),
        "relative_to_demo_close": [
            float(demo_bn[0]) - float(demo_close[0]),
            float(demo_bn[1]) - float(demo_close[1]),
            float(demo_bn[2]) - float(demo_close[2]),
        ],
        "mapping_method": (
            "mapped_demo_actual_close_plus_demo_bottleneck_close_offset"),
    }


def _write_grasp_replay_input(demo, trial_id, grasp=None,
                              live_object_position=None,
                              live_object_size=None):
    grasp_trajectory = (demo or {}).get("grasp_trajectory")
    if not isinstance(grasp_trajectory, dict):
        rospy.logwarn(
            "Insert grasp replay requested, but demo has no structured "
            "grasp_trajectory.")
        return ""
    poses = grasp_trajectory.get("poses", [])
    if len(poses) < 10:
        rospy.logwarn(
            "Insert grasp replay requested, but grasp_trajectory has only "
            "%d poses.", len(poses))
        return ""
    try:
        close_idx = int(grasp_trajectory.get("close_index"))
    except Exception:
        close_idx = None
    if close_idx is None or not (0 <= close_idx < len(poses)):
        rospy.logwarn(
            "Insert grasp replay requested, but grasp_trajectory close_index "
            "is invalid.")
        return ""

    trajectory = dict(grasp_trajectory)
    trajectory["poses"] = [dict(sample) for sample in poses]
    velocities = grasp_trajectory.get("velocities", [])
    trajectory["velocities"] = (
        [dict(sample) for sample in velocities]
        if isinstance(velocities, list) else [])
    trajectory["base_index"] = int(trajectory.get("base_index", 0))
    trajectory["close_index"] = close_idx

    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    path = os.path.join(
        ROLLOUT_DIR, "insert_grasp_replay_input_%s.json" % trial_id)
    mapped_bn = _mapped_grasp_bottleneck_pose(demo, grasp)
    mapped_close = _mapped_grasp_close_pose(demo, grasp)
    demo_object_position = _demo_position(
        demo, "object_info", fallback=live_object_position)
    demo_object_size = _demo_size(
        demo, "object_info", fallback=live_object_size)
    if live_object_position is None or live_object_size is None:
        rospy.logwarn(
            "Insert grasp replay object anchors missing live object metadata; "
            "live_object_position=%s live_object_size=%s",
            live_object_position, live_object_size)
    if demo_object_position is None or demo_object_size is None:
        rospy.logwarn(
            "Insert grasp replay object anchors missing demo object metadata; "
            "demo_object_position=%s demo_object_size=%s",
            demo_object_position, demo_object_size)
    payload = {
        "format": "mt3_demo_replay_input_v1",
        "source_demo": demo.get("id", ""),
        "trajectory": trajectory,
        "close_index": close_idx,
        "source_close_index": int(
            trajectory.get("source_close_index", close_idx)),
        "replay_base_position": trajectory.get("base_position"),
        "use_aligned_bottleneck_pose": bool(mapped_bn),
        "aligned_grasp_pose": mapped_close,
        "aligned_bottleneck_pose": mapped_bn,
        "replay_source": "structured_grasp_trajectory",
    }
    if (demo_object_position is not None and demo_object_size is not None and
            live_object_position is not None and live_object_size is not None):
        payload.update({
            "demo_object_position": [
                float(v) for v in demo_object_position[:3]],
            "demo_object_size": [
                float(v) for v in demo_object_size[:3]],
            "live_object_position": [
                float(v) for v in live_object_position[:3]],
            "live_object_size": [
                float(v) for v in live_object_size[:3]],
        })
        rospy.loginfo(
            "Insert grasp replay object anchors: demo_obj=%s demo_size=%s "
            "live_obj=%s live_size=%s",
            payload["demo_object_position"], payload["demo_object_size"],
            payload["live_object_position"], payload["live_object_size"])
    if mapped_bn:
        rospy.loginfo(
            "Insert grasp replay bottleneck mapped: live=[%.3f %.3f %.3f] "
            "demo_offset=[%.3f %.3f %.3f]",
            mapped_bn["position"][0], mapped_bn["position"][1],
            mapped_bn["position"][2],
            mapped_bn["relative_to_demo_close"][0],
            mapped_bn["relative_to_demo_close"][1],
            mapped_bn["relative_to_demo_close"][2])
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    rospy.loginfo(
        "Insert grasp replay input saved: %s poses=%d close_index=%d",
        path, len(poses), close_idx)
    return path


def _demo_grasp_pose(demo):
    gp = demo.get("grasp_pose_base_frame", {}) or {}
    pos_m = gp.get("position_m", {})
    ori = gp.get("orientation_xyzw", {})
    return (
        [
            float(pos_m.get("x", 0.60)),
            float(pos_m.get("y", 0.00)),
            float(pos_m.get("z", -0.50)),
        ],
        [
            float(ori.get("x", -1.0)),
            float(ori.get("y", 0.0)),
            float(ori.get("z", 0.0)),
            float(ori.get("w", 0.0)),
        ],
    )


def _aligned_grasp_from_demo(demo, live_cylinder_xyz):
    demo_obj = _demo_position(demo, "object_info", fallback=live_cylinder_xyz)
    demo_grasp, demo_q = _demo_grasp_pose(demo)
    delta = [
        demo_grasp[0] - demo_obj[0],
        demo_grasp[1] - demo_obj[1],
        demo_grasp[2] - demo_obj[2],
    ]
    return {
        "position": [
            float(live_cylinder_xyz[0]) + delta[0],
            float(live_cylinder_xyz[1]) + delta[1],
            float(live_cylinder_xyz[2]) + delta[2],
        ],
        "orientation": demo_q,
        "relative_to_cylinder_xyz": delta,
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


def _detected_features(cylinder_size):
    dims = [float(v) for v in cylinder_size[:3]]
    max_dim = max(max(dims), 0.001)
    return {
        "shape": rospy.get_param("~object_shape", "cylinder"),
        "dimensions_m": dims,
        "aspect_ratio": [v / max_dim for v in dims],
        "color_rgb": [0.0, 1.0, 0.0],
        "object_label": rospy.get_param("~target_label", "green_cylinder"),
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
        "~query", "insert the green cylinder into the blue socket")
    library = DemoLibrary(execution_environment=_execution_environment())
    demo, score, metadata = library.full_query(
        query,
        detected_features or _detected_features(rospy.get_param(
            "~cylinder_size", DEFAULT_CYLINDER_SIZE)),
        task_type="cylinder_insert_socket",
        retrieval_mode=rospy.get_param("~retrieval_mode", "hierarchical"),
        return_metadata=True)
    path = metadata.get("selected_demo_path") or _demo_path_by_id(demo.get("id", ""))
    if not path:
        raise RuntimeError("retrieved insert demo has no recorded JSON: %s" % demo.get("id", ""))
    metadata["selected_demo_path"] = path
    metadata["selected_score"] = float(score)
    metadata["query"] = query
    return path, metadata


def _append_csv_row(csv_path, row):
    existing_rows = []
    existing_fields = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [
                {k: v for k, v in old.items() if k is not None}
                for old in reader
            ]
    fieldnames = []
    for name in existing_fields + list(row.keys()):
        if name and name not in fieldnames:
            fieldnames.append(name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing_rows:
            writer.writerow({k: old.get(k, "") for k in fieldnames})
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _log_experiment_trial(trial_id, outcome, demo, retrieval_meta, scene,
                          grasp, insert_result, cylinder_size, socket_profile,
                          rollout_path, replay_path, scene_packages=None,
                          timing=None, failure_stage="", failure_reason="",
                          initial_target_gt=None, initial_socket_gt=None,
                          postcheck=None, execution_success=None,
                          executor_status=None):
    if not _param_bool("~auto_log_experiment", True):
        return
    os.makedirs(EXPERIMENT_LOG_DIR, exist_ok=True)
    perception_diag_only = bool(
        outcome == "dry_run" and _param_bool("~perception_diag_only", False))
    csv_name = (
        "mt3_perception_diagnostics.csv"
        if perception_diag_only else "mt3_relation_trials.csv")
    jsonl_name = (
        "mt3_perception_diagnostics.jsonl"
        if perception_diag_only else "mt3_relation_trials.jsonl")
    csv_path = os.path.join(EXPERIMENT_LOG_DIR, csv_name)
    target = scene.get("target", {})
    anchor = scene.get("anchor", scene.get("socket", {}))
    target_est = target.get("position_base")
    socket_est = anchor.get("position_base")
    target_gt = initial_target_gt or _gazebo_pose("~target_gt_model", ["green", "cylinder"])
    socket_gt = initial_socket_gt or _gazebo_pose("~socket_gt_model", ["blue", "socket"])
    target_gt_xyz = (target_gt or {}).get("xyz")
    socket_gt_xyz = (socket_gt or {}).get("xyz")
    replay_type, release_index = _replay_info(replay_path)
    replay_used = _param_bool("~use_demo_replay", True) and bool(replay_path)
    relation_alignment_mode = rospy.get_param(
        "~relation_alignment_mode", "target_anchor")
    timing = timing or {}
    postcheck = postcheck or {}
    executor_status = executor_status or {}
    execution_ok = (
        bool(outcome == "success") if execution_success is None
        else bool(execution_success))
    relation_path = ((scene_packages or {}).get("relation") or {}).get("relation_path", "")
    target_error_xy = _xy_error(target_est, target_gt_xyz)
    socket_error_xy = _xy_error(socket_est, socket_gt_xyz)
    perception_diag = target.get("perception_bias_diagnostics_v4") or \
        _target_perception_diagnostics(target)
    raw_center = perception_diag.get("raw_center_base_xyz")
    corrected_center = perception_diag.get("corrected_center_base_xyz") or target_est
    raw_dx, raw_dy, raw_error_xy = _xy_delta(raw_center, target_gt_xyz)
    corrected_dx, corrected_dy, corrected_error_xy = _xy_delta(
        corrected_center, target_gt_xyz)
    correction_error_change = ""
    try:
        if raw_error_xy != "" and corrected_error_xy != "":
            correction_error_change = float(corrected_error_xy) - float(raw_error_xy)
    except Exception:
        correction_error_change = ""
    failure_category = (
        _failure_category(failure_stage, failure_reason)
        if outcome == "failed" else "")
    failure_stage_detail = str(executor_status.get(
        "failure_stage_detail", "") or "")
    if outcome == "failed" and not failure_stage_detail:
        failure_stage_detail = str(failure_stage or "execution")
    pure_replay_execution_success = bool(executor_status.get(
        "pure_replay_success", False))
    pure_replay_success = bool(
        outcome == "success" and pure_replay_execution_success)
    insert_failure_detail = (
        _insert_failure_detail(
            failure_category, failure_reason, target_error_xy, socket_error_xy)
        if outcome == "failed" else "")
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_id": trial_id,
        "task_type": "cylinder_insert_socket",
        "query": rospy.get_param(
            "~query", "insert the green cylinder into the blue socket"),
        "condition_id": rospy.get_param("~condition_id", ""),
        "repeat_id": rospy.get_param("~repeat_id", ""),
        "method_variant": rospy.get_param("~method_variant", "full"),
        "relation_alignment_mode": relation_alignment_mode,
        "insert_replay_anchor_mode": rospy.get_param(
            "~insert_replay_anchor_mode", "bottleneck"),
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
        "insert_failure_detail": insert_failure_detail,
        "retrieval_mode": retrieval_meta.get("retrieval_mode", ""),
        "retrieved_demo_id": demo.get("id", ""),
        "language_score": retrieval_meta.get("language_score", ""),
        "geometry_score": retrieval_meta.get("geometry_score", retrieval_meta.get("selected_score", "")),
        "target_shape": rospy.get_param("~object_shape", "cylinder"),
        "target_size_xyz": _json_vec(cylinder_size),
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
        "target_perception_error_xy_m": target_error_xy,
        "target_perception_error_xyz_m": "",
        "target_perception_error_metric": "xy_base_vs_gazebo_world_model_xy",
        "target_perception_method": target.get("method", ""),
        "target_geometry_center_method": _geometry_method(target),
        "target_pc_diag_version": perception_diag.get("version", ""),
        "target_pc_source_frame": perception_diag.get("source_frame", ""),
        "target_pc_source_point_count": perception_diag.get("source_point_count", ""),
        "target_pc_source_x_min_m": perception_diag.get("source_x_min_m", ""),
        "target_pc_source_x_p05_m": perception_diag.get("source_x_p05_m", ""),
        "target_pc_source_x_p10_m": perception_diag.get("source_x_p10_m", ""),
        "target_pc_source_x_p25_m": perception_diag.get("source_x_p25_m", ""),
        "target_pc_source_x_p50_m": perception_diag.get("source_x_p50_m", ""),
        "target_pc_source_x_p75_m": perception_diag.get("source_x_p75_m", ""),
        "target_pc_source_x_p90_m": perception_diag.get("source_x_p90_m", ""),
        "target_pc_source_x_p95_m": perception_diag.get("source_x_p95_m", ""),
        "target_pc_source_x_max_m": perception_diag.get("source_x_max_m", ""),
        "target_pc_source_y_min_m": perception_diag.get("source_y_min_m", ""),
        "target_pc_source_y_p05_m": perception_diag.get("source_y_p05_m", ""),
        "target_pc_source_y_p10_m": perception_diag.get("source_y_p10_m", ""),
        "target_pc_source_y_p25_m": perception_diag.get("source_y_p25_m", ""),
        "target_pc_source_y_p50_m": perception_diag.get("source_y_p50_m", ""),
        "target_pc_source_y_p75_m": perception_diag.get("source_y_p75_m", ""),
        "target_pc_source_y_p90_m": perception_diag.get("source_y_p90_m", ""),
        "target_pc_source_y_p95_m": perception_diag.get("source_y_p95_m", ""),
        "target_pc_source_y_max_m": perception_diag.get("source_y_max_m", ""),
        "target_pc_source_z_p05_m": perception_diag.get("source_z_p05_m", ""),
        "target_pc_source_z_p50_m": perception_diag.get("source_z_p50_m", ""),
        "target_pc_source_z_p95_m": perception_diag.get("source_z_p95_m", ""),
        "target_raw_center_base_xyz": _json_vec(raw_center),
        "target_corrected_center_base_xyz_diag": _json_vec(corrected_center),
        "target_geometry_correction_dx_m": ((perception_diag.get("geometry_correction_delta_xyz") or ["", "", ""])[0]),
        "target_geometry_correction_dy_m": ((perception_diag.get("geometry_correction_delta_xyz") or ["", "", ""])[1]),
        "target_pc_base_bound_low_x_m": ((perception_diag.get("base_bounds_low_xyz") or ["", "", ""])[0]),
        "target_pc_base_bound_low_y_m": ((perception_diag.get("base_bounds_low_xyz") or ["", "", ""])[1]),
        "target_pc_base_bound_high_x_m": ((perception_diag.get("base_bounds_high_xyz") or ["", "", ""])[0]),
        "target_pc_base_bound_high_y_m": ((perception_diag.get("base_bounds_high_xyz") or ["", "", ""])[1]),
        "target_pc_base_bound_mid_x_m": ((perception_diag.get("base_bounds_mid_xyz") or ["", "", ""])[0]),
        "target_pc_base_bound_mid_y_m": ((perception_diag.get("base_bounds_mid_xyz") or ["", "", ""])[1]),
        "target_pc_base_bound_extent_x_m": ((perception_diag.get("base_bounds_extent_xyz") or ["", "", ""])[0]),
        "target_pc_base_bound_extent_y_m": ((perception_diag.get("base_bounds_extent_xyz") or ["", "", ""])[1]),
        "target_raw_center_dx_gt_m": raw_dx,
        "target_raw_center_dy_gt_m": raw_dy,
        "target_raw_center_error_xy_m": raw_error_xy,
        "target_corrected_center_dx_gt_m": corrected_dx,
        "target_corrected_center_dy_gt_m": corrected_dy,
        "target_corrected_center_error_xy_m_diag": corrected_error_xy,
        "target_geometry_correction_error_change_m": correction_error_change,
        "target_x": (target_est or ["", "", ""])[0],
        "target_y": (target_est or ["", "", ""])[1],
        "target_z": (target_est or ["", "", ""])[2],
        "anchor_gt_xyz": "",
        "anchor_est_xyz": "",
        "anchor_error_xy_m": "",
        "socket_gt_model": (socket_gt or {}).get("name", ""),
        "socket_gt_xyz": _xy_only_xyz(socket_gt_xyz),
        "socket_gt_world_xyz": _json_vec(socket_gt_xyz),
        "initial_socket_xyz": _json_vec(socket_gt_xyz),
        "initial_anchor_xyz": _json_vec(socket_gt_xyz),
        "socket_gt_frame": "gazebo_world_xy_only",
        "socket_est_frame": "base",
        "socket_est_xyz": _json_vec(socket_est),
        "socket_error_xy_m": socket_error_xy,
        "socket_perception_error_xy_m": socket_error_xy,
        "socket_perception_error_xyz_m": "",
        "socket_perception_error_metric": "xy_base_vs_gazebo_world_model_xy",
        "socket_perception_method": anchor.get("method", ""),
        "socket_geometry_center_method": _geometry_method(anchor),
        "socket_error_z_m": "",
        "socket_error_xyz_m": "",
        "anchor_x": (socket_est or ["", "", ""])[0],
        "anchor_y": (socket_est or ["", "", ""])[1],
        "anchor_z": (socket_est or ["", "", ""])[2],
        "target_yaw_gt_deg": "",
        "target_yaw_est_deg": "",
        "yaw_error_deg": "",
        "grasp_x": grasp["position"][0],
        "grasp_y": grasp["position"][1],
        "grasp_z": grasp["position"][2],
        "grasp_xyz": _json_vec(grasp.get("position")),
        "bottleneck_xyz": _json_vec((demo.get("insertion_bottleneck_pose_base_frame") or {}).get("position", [])),
        "place_x": insert_result["place_xyz"][0],
        "place_y": insert_result["place_xyz"][1],
        "place_z": insert_result["place_xyz"][2],
        "place_or_insert_xyz": _json_vec(insert_result.get("place_xyz")),
        "cylinder_size": json.dumps([float(v) for v in cylinder_size]),
        "socket_size": json.dumps([float(v) for v in socket_profile.get("size_m", [])]),
        "socket_opening": json.dumps([float(v) for v in socket_profile.get("opening_m", [])]),
        "place_offset_xyz": json.dumps(insert_result.get("offset_xyz", [])),
        "final_object_model_name": postcheck.get("final_object_model_name", ""),
        "final_socket_model_name": postcheck.get("final_socket_model_name", ""),
        "final_object_xyz": _json_vec(postcheck.get("final_object_xyz")),
        "final_socket_xyz": _json_vec(postcheck.get("final_socket_xyz")),
        "final_anchor_xyz": _json_vec(postcheck.get("final_socket_xyz")),
        "final_target_error_xy_m": postcheck.get("final_target_error_xy_m", ""),
        "final_relation_error_xy_m": postcheck.get("final_relation_error_xy_m", ""),
        "max_insert_relation_error_xy_m": postcheck.get(
            "max_insert_relation_error_xy_m", ""),
        "rim_contact_or_collision_flag": postcheck.get(
            "rim_contact_or_collision_flag", ""),
        "rim_contact_xy_threshold_m": postcheck.get(
            "rim_contact_xy_threshold_m", ""),
        "insert_depth_m": postcheck.get("insert_depth_m", ""),
        "postcheck_reason": postcheck.get("postcheck_reason", ""),
        "replay_used": replay_used,
        "replay_type": replay_type,
        "release_index": release_index,
        "grasp_replay_used": _param_bool(
            "/sawyer_auto_grasp/use_grasp_replay", False),
        "grasp_replay_path": rospy.get_param(
            "/sawyer_auto_grasp/grasp_replay_trajectory_path", ""),
        "mask_pixels": _mask_pixels(target.get("mask_path")),
        "anchor_mask_pixels": _mask_pixels(anchor.get("mask_path")),
        "pointcloud_points": _point_count(target),
        "anchor_pointcloud_points": _point_count(anchor),
        "icp_mean_error_m": "",
        "icp_median_error_m": "",
        "icp_p90_error_m": "",
        "planning_success": "",
        "execution_success": execution_ok,
        "grasp_replay_attempted": executor_status.get(
            "grasp_replay_attempted", False),
        "grasp_replay_success": executor_status.get(
            "grasp_replay_success", ""),
        "grasp_replay_stage": executor_status.get(
            "grasp_replay_stage", ""),
        "grasp_replay_failure_stage": executor_status.get(
            "grasp_replay_failure_stage", ""),
        "insertion_replay_attempted": executor_status.get(
            "insertion_replay_attempted", False),
        "insertion_replay_success": executor_status.get(
            "insertion_replay_success", ""),
        "insertion_replay_stage": executor_status.get(
            "insertion_replay_stage", ""),
        "insertion_replay_failure_stage": executor_status.get(
            "insertion_replay_failure_stage", ""),
        "insertion_interaction_success": executor_status.get(
            "insertion_interaction_success", False),
        "post_release_retreat_attempted": executor_status.get(
            "post_release_retreat_attempted", False),
        "post_release_retreat_success": executor_status.get(
            "post_release_retreat_success", ""),
        "scripted_fallback_used": executor_status.get(
            "scripted_fallback_used", False),
        "pure_replay_execution_success": pure_replay_execution_success,
        "pure_replay_success": pure_replay_success,
        "failure_stage_detail": failure_stage_detail,
        "replay_failure_stage_detail": executor_status.get(
            "replay_failure_stage_detail", ""),
        "grasp_post_close_motion_max_m": executor_status.get(
            "grasp_post_close_motion_max_m", ""),
        "grasp_post_close_motion_max_xy_m": executor_status.get(
            "grasp_post_close_motion_max_xy_m", ""),
        "grasp_post_close_motion_max_z_m": executor_status.get(
            "grasp_post_close_motion_max_z_m", ""),
        "grasp_post_close_mode": executor_status.get(
            "grasp_post_close_mode", ""),
        "grasp_post_close_dwell_s": executor_status.get(
            "grasp_post_close_dwell_s", ""),
        "diag_grasp_before_close_cylinder_hand_offset_x_m": executor_status.get("diag_grasp_before_close_cylinder_hand_offset_x_m", ""),
        "diag_grasp_before_close_cylinder_hand_offset_y_m": executor_status.get("diag_grasp_before_close_cylinder_hand_offset_y_m", ""),
        "diag_grasp_before_close_cylinder_hand_offset_xy_m": executor_status.get("diag_grasp_before_close_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_x_m": executor_status.get("diag_grasp_after_close_cylinder_hand_offset_x_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_y_m": executor_status.get("diag_grasp_after_close_cylinder_hand_offset_y_m", ""),
        "diag_grasp_after_close_cylinder_hand_offset_xy_m": executor_status.get("diag_grasp_after_close_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_x_m": executor_status.get("diag_grasp_after_lift_cylinder_hand_offset_x_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_y_m": executor_status.get("diag_grasp_after_lift_cylinder_hand_offset_y_m", ""),
        "diag_grasp_after_lift_cylinder_hand_offset_xy_m": executor_status.get("diag_grasp_after_lift_cylinder_hand_offset_xy_m", ""),
        "diag_grasp_complete_before_transport_cylinder_hand_offset_xy_m": executor_status.get("diag_grasp_complete_before_transport_cylinder_hand_offset_xy_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_x_m": executor_status.get("diag_pre_step_f_cylinder_hand_offset_x_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_y_m": executor_status.get("diag_pre_step_f_cylinder_hand_offset_y_m", ""),
        "diag_pre_step_f_cylinder_hand_offset_xy_m": executor_status.get("diag_pre_step_f_cylinder_hand_offset_xy_m", ""),
        "diag_pre_step_f_cylinder_socket_offset_x_m": executor_status.get("diag_pre_step_f_cylinder_socket_offset_x_m", ""),
        "diag_pre_step_f_cylinder_socket_offset_y_m": executor_status.get("diag_pre_step_f_cylinder_socket_offset_y_m", ""),
        "diag_pre_step_f_cylinder_socket_error_xy_m": executor_status.get("diag_pre_step_f_cylinder_socket_error_xy_m", ""),
        "diag_pre_step_f_replay_cylinder_socket_error_xy_m": executor_status.get("diag_pre_step_f_replay_cylinder_socket_error_xy_m", ""),
        "diag_step_f_first_cylinder_socket_error_xy_m": executor_status.get("diag_step_f_first_cylinder_socket_error_xy_m", ""),
        "diag_step_f_last_cylinder_socket_error_xy_m": executor_status.get("diag_step_f_last_cylinder_socket_error_xy_m", ""),
        "diag_step_f_max_cylinder_socket_error_xy_m": executor_status.get("diag_step_f_max_cylinder_socket_error_xy_m", ""),
        "diag_step_f_last_hand_tracking_error_xyz_m": executor_status.get("diag_step_f_last_hand_tracking_error_xyz_m", ""),
        "diag_step_f_max_hand_tracking_error_xyz_m": executor_status.get("diag_step_f_max_hand_tracking_error_xyz_m", ""),
        "diag_post_release_cylinder_socket_error_xy_m": executor_status.get("diag_post_release_cylinder_socket_error_xy_m", ""),
        "insert_tracking_failure": executor_status.get("insert_tracking_failure", ""),
        "insert_tracking_failure_chunk": executor_status.get("insert_tracking_failure_chunk", ""),
        "insert_tracking_failure_error_m": executor_status.get("insert_tracking_failure_error_m", ""),
        "insert_replay_tracking_error_max_m_active": executor_status.get("insert_replay_tracking_error_max_m_active", ""),
        "diag_pre_step_f_translational_jacobian_condition_number": executor_status.get("diag_pre_step_f_translational_jacobian_condition_number", ""),
        "diag_pre_step_f_translational_jacobian_sigma_min": executor_status.get("diag_pre_step_f_translational_jacobian_sigma_min", ""),
        "diag_pre_step_f_vertical_joint_velocity_gain": executor_status.get("diag_pre_step_f_vertical_joint_velocity_gain", ""),
        "diag_pre_step_f_joint_limit_min_margin_normalized": executor_status.get("diag_pre_step_f_joint_limit_min_margin_normalized", ""),
        "diag_pre_step_f_cylinder_speed_xy_m_s": executor_status.get("diag_pre_step_f_cylinder_speed_xy_m_s", ""),
        "diag_pre_step_f_cylinder_tilt_deg": executor_status.get("diag_pre_step_f_cylinder_tilt_deg", ""),
        "diag_step_f_first_translational_jacobian_condition_number": executor_status.get("diag_step_f_first_translational_jacobian_condition_number", ""),
        "diag_step_f_last_translational_jacobian_condition_number": executor_status.get("diag_step_f_last_translational_jacobian_condition_number", ""),
        "diag_step_f_max_translational_jacobian_condition_number": executor_status.get("diag_step_f_max_translational_jacobian_condition_number", ""),
        "diag_step_f_first_translational_jacobian_sigma_min": executor_status.get("diag_step_f_first_translational_jacobian_sigma_min", ""),
        "diag_step_f_last_translational_jacobian_sigma_min": executor_status.get("diag_step_f_last_translational_jacobian_sigma_min", ""),
        "diag_step_f_min_translational_jacobian_sigma_min": executor_status.get("diag_step_f_min_translational_jacobian_sigma_min", ""),
        "diag_step_f_first_vertical_joint_velocity_gain": executor_status.get("diag_step_f_first_vertical_joint_velocity_gain", ""),
        "diag_step_f_last_vertical_joint_velocity_gain": executor_status.get("diag_step_f_last_vertical_joint_velocity_gain", ""),
        "diag_step_f_max_vertical_joint_velocity_gain": executor_status.get("diag_step_f_max_vertical_joint_velocity_gain", ""),
        "diag_step_f_first_joint_limit_min_margin_normalized": executor_status.get("diag_step_f_first_joint_limit_min_margin_normalized", ""),
        "diag_step_f_last_joint_limit_min_margin_normalized": executor_status.get("diag_step_f_last_joint_limit_min_margin_normalized", ""),
        "diag_step_f_min_joint_limit_min_margin_normalized": executor_status.get("diag_step_f_min_joint_limit_min_margin_normalized", ""),
        "diag_step_f_first_cylinder_speed_xy_m_s": executor_status.get("diag_step_f_first_cylinder_speed_xy_m_s", ""),
        "diag_step_f_last_cylinder_speed_xy_m_s": executor_status.get("diag_step_f_last_cylinder_speed_xy_m_s", ""),
        "diag_step_f_max_cylinder_speed_xy_m_s": executor_status.get("diag_step_f_max_cylinder_speed_xy_m_s", ""),
        "diag_step_f_first_cylinder_tilt_deg": executor_status.get("diag_step_f_first_cylinder_tilt_deg", ""),
        "diag_step_f_last_cylinder_tilt_deg": executor_status.get("diag_step_f_last_cylinder_tilt_deg", ""),
        "diag_step_f_max_cylinder_tilt_deg": executor_status.get("diag_step_f_max_cylinder_tilt_deg", ""),
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
    _append_csv_row(csv_path, row)
    jsonl_path = os.path.join(EXPERIMENT_LOG_DIR, jsonl_name)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if perception_diag_only:
        rospy.loginfo("  Perception diagnostic trial logged: %s", csv_path)
    else:
        rospy.loginfo("  Relation experiment trial logged: %s", csv_path)


def _write_replay_input(demo, trial_id, insert_result=None):
    trajectory = demo.get("insertion_trajectory") or demo.get("trajectory")
    if not trajectory:
        return ""
    anchor_mode = str(rospy.get_param(
        "~insert_replay_anchor_mode", "bottleneck")).strip().lower()
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    path = os.path.join(ROLLOUT_DIR, "insert_replay_input_%s.json" % trial_id)
    payload = {
        "format": "mt3_cylinder_insert_replay_input_v1",
        "source_demo": demo.get("id", ""),
        "trajectory": trajectory,
        "trajectory_source": (
            "insertion_trajectory"
            if demo.get("insertion_trajectory") else "full_trajectory"),
        "insertion_bottleneck_pose_base_frame": demo.get(
            "insertion_bottleneck_pose_base_frame"),
        "place_info": demo.get("place_info", {}),
        "anchor_info": demo.get("anchor_info", {}),
        "insert_replay_anchor_mode": anchor_mode,
    }
    if anchor_mode in ("bottleneck", "mapped_bottleneck", "mapped"):
        mapped_bn = _mapped_insertion_bottleneck_pose(demo, insert_result or {})
        if mapped_bn:
            payload["place_bottleneck_pose_base_frame"] = demo.get(
                "insertion_bottleneck_pose_base_frame")
            payload["aligned_place_bottleneck_pose"] = mapped_bn
            rospy.loginfo(
                "Insert replay bottleneck mapped: live=[%.3f %.3f %.3f] "
                "demo_offset=[%.3f %.3f %.3f]",
                mapped_bn["position"][0], mapped_bn["position"][1],
                mapped_bn["position"][2],
                mapped_bn["relative_to_demo_insert"][0],
                mapped_bn["relative_to_demo_insert"][1],
                mapped_bn["relative_to_demo_insert"][2])
        else:
            rospy.logwarn(
                "Insert replay anchor mode requested bottleneck, but demo "
                "insert/bottleneck or live insert pose is missing; keeping "
                "legacy place-anchored replay.")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _write_execution_params(grasp, insert_result, cylinder_size, rollout_path,
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
    rospy.set_param("/sawyer_auto_grasp/object_size", cylinder_size)

    insert_xyz = insert_result["place_xyz"]
    socket_profile = insert_result.get("anchor_profile", {})
    socket_height = float(rospy.get_param(
        "~socket_height", (socket_profile.get("size_m") or [0.0, 0.0, 0.100])[2]))
    rospy.set_param("/sawyer_auto_grasp/place_x", float(insert_xyz[0]))
    rospy.set_param("/sawyer_auto_grasp/place_y", float(insert_xyz[1]))
    rospy.set_param("/sawyer_auto_grasp/place_z", float(insert_xyz[2]))
    rospy.set_param("/sawyer_auto_grasp/place_direction", "insert_into_socket")
    rospy.set_param("/sawyer_auto_grasp/record_model_state_samples", True)
    rospy.set_param("/sawyer_auto_grasp/target_model_name", str(
        rospy.get_param("~target_gt_model", "green_insert_cylinder")))
    rospy.set_param("/sawyer_auto_grasp/socket_model_name", str(
        rospy.get_param("~socket_gt_model", "blue_insert_socket")))
    offset_xyz = insert_result.get("offset_xyz") or [0.0, 0.0, 0.0]
    rospy.set_param("/sawyer_auto_grasp/insert_desired_offset_x",
                    float(offset_xyz[0]))
    rospy.set_param("/sawyer_auto_grasp/insert_desired_offset_y",
                    float(offset_xyz[1]))
    rospy.set_param("/sawyer_auto_grasp/place_clearance", float(rospy.get_param(
        "~insert_clearance", 0.020)))
    rospy.set_param("/sawyer_auto_grasp/insert_socket_height", socket_height)
    rospy.set_param("/sawyer_auto_grasp/insert_release_clearance", float(
        rospy.get_param("~insert_release_clearance", 0.006)))
    rospy.set_param("/sawyer_auto_grasp/place_lift_height", float(rospy.get_param(
        "~place_lift_height", 0.160)))
    _forward_insert_motion_params()

    replay_requested = _param_bool("~use_demo_replay", True)
    if replay_requested and not replay_path:
        raise RuntimeError(
            "Demo replay requested but insertion replay trajectory is unavailable; "
            "scripted fallback is disabled.")
    use_replay = bool(replay_requested)
    use_grasp_replay = (
        use_replay and
        _param_bool("~use_grasp_replay", True) and
        bool(grasp_replay_path))
    rospy.set_param("/sawyer_auto_grasp/use_demo_replay", use_replay)
    rospy.set_param("/sawyer_auto_grasp/use_place_release_replay", use_replay)
    rospy.set_param("/sawyer_auto_grasp/use_grasp_replay", use_grasp_replay)
    rospy.set_param(
        "/sawyer_auto_grasp/insert_require_grasp_replay",
        _param_bool("~insert_require_grasp_replay", True))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_trajectory_path",
                    grasp_replay_path if use_grasp_replay else "")
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_prefer_pose_replay",
                    _param_bool("~grasp_replay_prefer_pose_replay", True))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_use_segmented_replay",
                    _param_bool("~grasp_replay_use_segmented_replay", True))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_close_on_blocked",
                    _param_bool("~grasp_replay_close_on_blocked", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_close_on_blocked_min_progress",
        float(rospy.get_param(
            "~grasp_replay_close_on_blocked_min_progress", 0.35)))
    rospy.set_param("/sawyer_auto_grasp/grasp_replay_anchor_close_waypoint",
                    _param_bool("~grasp_replay_anchor_close_waypoint", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_use_aligned_bottleneck_pose",
        _param_bool("~grasp_replay_use_aligned_bottleneck_pose", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_use_top_mouth_center_final_correction",
        _param_bool(
            "~grasp_replay_use_top_mouth_center_final_correction", False))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_close_anchor_offset_x",
        float(rospy.get_param("~grasp_replay_close_anchor_offset_x", 0.0)))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_replay_close_anchor_offset_y",
        float(rospy.get_param("~grasp_replay_close_anchor_offset_y", 0.0)))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_post_close_dwell_detection",
        _param_bool("~grasp_post_close_dwell_detection", True))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_post_close_dwell_threshold_m",
        float(rospy.get_param("~grasp_post_close_dwell_threshold_m", 0.001)))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_post_close_dwell_max_s",
        float(rospy.get_param("~grasp_post_close_dwell_max_s", 5.0)))
    rospy.set_param(
        "/sawyer_auto_grasp/grasp_close_settle_s",
        float(rospy.get_param("~grasp_close_settle_s", 1.0)))
    rospy.set_param("/sawyer_auto_grasp/demo_replay_trajectory_path", replay_path)
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_path", rollout_path)
    rospy.set_param("/sawyer_auto_grasp/trajectory_record_rate_hz", float(
        rospy.get_param("~trajectory_record_rate_hz", 10.0)))


def _run_executor():
    script = _executor_path()
    if not os.path.exists(script):
        raise RuntimeError("executor script not found: %s" % script)
    detected_failure = None
    error_patterns = [
        ("no motion plan", "motion_planning", "moveit_no_motion_plan"),
        ("planning failed", "motion_planning", "moveit_planning_failed"),
        ("plan failed", "motion_planning", "moveit_planning_failed"),
        ("controller", "trajectory_execution", "controller_execution_failed"),
        ("control_failed", "trajectory_execution", "controller_control_failed"),
        ("path_tolerance", "trajectory_execution", "path_tolerance_violated"),
        ("aborted", "trajectory_execution", "trajectory_execution_aborted"),
        ("z error", "insertion_execution", "insert_depth_insufficient"),
        ("descent still too high", "insertion_execution", "insert_depth_insufficient"),
        ("insert_into_socket", "insertion_execution", "insert_execution_failed"),
        ("replay execute failed", "replay_execution", "insert_replay_execute_failed"),
        ("insertion/place replay failed", "replay_execution", "insert_replay_failed_terminal"),
        ("replay path is empty", "replay_execution", "insert_replay_path_missing"),
        ("scripted fallback disabled", "replay_execution", "insert_replay_failed_terminal"),
        ("failed/skipped", "replay_execution", "insert_replay_failed_or_skipped"),
        ("pick-place execution failed", "insertion_execution", "insert_execution_failed"),
    ]
    proc = subprocess.Popen(
        ["python", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            rospy.loginfo("[insert-pipeline executor] %s", line)
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
    rospy.init_node("mt3_cylinder_insert_pipeline", anonymous=True)
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    run_start = time.time()

    warm_demo = None
    try:
        with open(_latest_demo_path("cylinder_insert_socket"), "r") as f:
            warm_demo = json.load(f)
    except Exception:
        warm_demo = None

    t_perception = time.time()
    scene, cylinder_xyz, socket_xyz, cylinder_size = _load_scene(warm_demo)
    perception_time_s = time.time() - t_perception
    t_retrieval = time.time()
    demo_path, retrieval_meta = _find_demo_path(_detected_features(cylinder_size))
    retrieval_time_s = time.time() - t_retrieval
    t_alignment = time.time()
    with open(demo_path, "r") as f:
        demo = json.load(f)

    cylinder_xyz = _stabilize_cylinder_z_from_demo_relation(
        demo, scene, cylinder_xyz, socket_xyz)

    grasp = _aligned_grasp_from_demo(demo, cylinder_xyz)
    socket_profile = {
        "name": rospy.get_param("~socket_name", "blue_insert_socket"),
        "category": rospy.get_param(
            "~socket_category", "shallow_circular_socket"),
        "size_m": _param_float_list(
            "~socket_size", DEFAULT_SOCKET_PROFILE["size_m"]),
        "opening_m": _param_float_list(
            "~socket_opening", DEFAULT_SOCKET_PROFILE["opening_m"]),
        "surface_z_offset": float(rospy.get_param("~socket_surface_z_offset", 0.0)),
    }
    insert_result = compute_insert_target(
        socket_xyz,
        cylinder_position_base=cylinder_xyz,
        cylinder_size=cylinder_size,
        demo_entry=demo,
        socket_profile=socket_profile,
        override_offset_xyz=rospy.get_param("~socket_insert_offset_xyz", None),
        relation_alignment_mode=rospy.get_param(
            "~relation_alignment_mode", "target_anchor"))
    relation_alignment_mode = rospy.get_param(
        "~relation_alignment_mode", "target_anchor")
    alignment_time_s = time.time() - t_alignment

    trial_id = rospy.get_param(
        "~trial_id", "insert_%s" % time.strftime("%Y%m%d_%H%M%S"))
    rollout_path = os.path.join(ROLLOUT_DIR, "rollout_%s.json" % trial_id)
    replay_path = _write_replay_input(demo, trial_id, insert_result=insert_result)
    grasp_replay_path = (
        _write_grasp_replay_input(
            demo, trial_id, grasp=grasp,
            live_object_position=cylinder_xyz,
            live_object_size=cylinder_size)
        if _param_bool("~use_grasp_replay", True) else "")
    scene_packages = save_dual_object_scene_packages(
        scene,
        trial_id,
        role="live_insert_trial",
        target_label="green_cylinder",
        anchor_label="blue_insert_socket",
        relation_kind="cylinder_insert_socket",
        extra_metadata={
            "trial_id": trial_id,
            "source_demo": demo.get("id", ""),
            "retrieval": retrieval_meta,
            "task_type": "cylinder_insert_socket",
            "relation_alignment_mode": relation_alignment_mode,
            "cylinder_size": [float(v) for v in cylinder_size],
            "socket_size": [float(v) for v in socket_profile["size_m"]],
            "socket_opening": [float(v) for v in socket_profile["opening_m"]],
            "socket_profile": socket_profile,
            "insert_target_xyz": [float(v) for v in insert_result["place_xyz"]],
            "insert_offset_xyz": [float(v) for v in insert_result["offset_xyz"]],
            "target_perception_bias_diagnostics_v4":
                ((scene.get("target") or {}).get("perception_bias_diagnostics_v4") or {}),
        })
    _write_execution_params(
        grasp, insert_result, cylinder_size, rollout_path, replay_path,
        grasp_replay_path=grasp_replay_path)

    rospy.loginfo("=" * 60)
    rospy.loginfo("Cylinder insertion MT3 pipeline")
    rospy.loginfo("  diagnostic build: %s", INSERT_PIPELINE_DIAG_VERSION)
    rospy.loginfo("  query: %s", retrieval_meta.get("query", ""))
    rospy.loginfo("  retrieval: %s selected=%s",
                  retrieval_meta.get("retrieval_mode", ""),
                  retrieval_meta.get("selected_demo_id", demo.get("id", "")))
    rospy.loginfo("  relation alignment: %s", relation_alignment_mode)
    rospy.loginfo("  insert replay anchor mode: %s",
                  rospy.get_param("~insert_replay_anchor_mode", "bottleneck"))
    rospy.loginfo("  demo: %s", demo_path)
    rospy.loginfo("  cylinder: [%.3f %.3f %.3f]",
                  cylinder_xyz[0], cylinder_xyz[1], cylinder_xyz[2])
    rospy.loginfo("  socket:   [%.3f %.3f %.3f]",
                  socket_xyz[0], socket_xyz[1], socket_xyz[2])
    rospy.loginfo("  grasp:    [%.3f %.3f %.3f]",
                  grasp["position"][0], grasp["position"][1],
                  grasp["position"][2])
    rospy.loginfo("  insert:   [%.3f %.3f %.3f]",
                  insert_result["place_xyz"][0], insert_result["place_xyz"][1],
                  insert_result["place_xyz"][2])
    rospy.loginfo("  scene packages: target=%s anchor=%s relation=%s",
                  (scene_packages.get("target_package") or {}).get(
                      "package_dir", "(none)"),
                  (scene_packages.get("anchor_package") or {}).get(
                      "package_dir", "(none)"),
                  scene_packages.get("relation", {}).get(
                      "relation_path", "(none)"))
    rospy.loginfo("  replay:   %s", replay_path or "(disabled/no trajectory)")
    rospy.loginfo("  grasp replay: %s",
                  grasp_replay_path or "(disabled/no grasp trajectory)")
    rospy.loginfo("=" * 60)

    initial_target_gt = _gazebo_pose("~target_gt_model", ["green", "cylinder"])
    initial_socket_gt = _gazebo_pose("~socket_gt_model", ["blue", "socket"])

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
            grasp, insert_result, cylinder_size, socket_profile,
            rollout_path, replay_path, scene_packages=scene_packages,
            timing=timing, initial_target_gt=initial_target_gt,
            initial_socket_gt=initial_socket_gt)
        return True

    _reset_executor_timing_params()
    _reset_executor_status_params()
    t_execution = time.time()
    ok, failure_info = _run_executor()
    execution_time_s = time.time() - t_execution
    executor_status = _read_executor_status_params()
    if (ok and
            executor_status.get("insertion_interaction_success", False) and
            executor_status.get("post_release_retreat_success", "") is False):
        rospy.logwarn(
            "Executor reached insertion release but Step H retreat failed. "
            "Running final Gazebo postcheck to decide task_success; "
            "pure_replay_success will remain False.")
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
        process_metrics = _rollout_insert_process_metrics(
            rollout_path, cylinder_size, socket_profile)
        _log_experiment_trial(
            trial_id, "failed", demo, retrieval_meta, scene,
            grasp, insert_result, cylinder_size, socket_profile,
            rollout_path, replay_path, scene_packages=scene_packages,
            timing=timing, failure_stage=failure_stage or "execution",
            failure_reason=failure_reason or "cylinder insertion execution failed",
            initial_target_gt=initial_target_gt,
            initial_socket_gt=initial_socket_gt,
            postcheck=process_metrics,
            execution_success=False,
            executor_status=executor_status)
        raise RuntimeError(failure_reason or "cylinder insertion execution failed")
    postcheck = _validate_post_insert_success(
        insert_result, cylinder_size, socket_profile,
        (initial_socket_gt or {}).get("xyz"))
    postcheck.update(_rollout_insert_process_metrics(
        rollout_path, cylinder_size, socket_profile))
    if not postcheck.get("ok", False):
        failure_stage = postcheck.get("failure_stage", "insertion_verification")
        failure_reason = postcheck.get(
            "failure_reason", "圆柱插入后验检查失败")
        _log_experiment_trial(
            trial_id, "failed", demo, retrieval_meta, scene,
            grasp, insert_result, cylinder_size, socket_profile,
            rollout_path, replay_path, scene_packages=scene_packages,
            timing=timing, failure_stage=failure_stage,
            failure_reason=failure_reason,
            initial_target_gt=initial_target_gt,
            initial_socket_gt=initial_socket_gt,
            postcheck=postcheck,
            execution_success=True,
            executor_status=executor_status)
        raise RuntimeError(failure_reason)
    _log_experiment_trial(
        trial_id, "success", demo, retrieval_meta, scene,
        grasp, insert_result, cylinder_size, socket_profile,
        rollout_path, replay_path, scene_packages=scene_packages,
        timing=timing, initial_target_gt=initial_target_gt,
        initial_socket_gt=initial_socket_gt,
        postcheck=postcheck, execution_success=True,
        executor_status=executor_status)
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("mt3_cylinder_insert_pipeline failed: %s", exc)
        sys.exit(1)
