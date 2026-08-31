#!/usr/bin/env python3
"""
MT3 Pipeline — Following the official deploy_mt3.py 7-step flow.

Official MT3 pipeline (from README):
  1. Load test image + segmentation
  2. Initialize live scene state (point cloud from RGB-D+seg)
  3. Retrieve similar demonstration
  4. Load retrieved demonstration
  5. Estimate relative pose (PointNet++ → we use PnP+Alignment)
  6. Transform bottleneck pose to live scene
  7. Load end-effector twists for interaction

Adapted for Sawyer+Gazebo with red cube (monocular RGB, no depth camera).
Generates official-style visualizations saved to ~/.mt3_debug/
"""
import rospy
import sys
import os
import math
import json
import time
import shutil
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt3_demo_library import DemoLibrary
from mt3_perception import PerceptionNode
from mt3_alignment import TrajectoryAligner, pose_compose, quat_multiply
from mt3_visualization import generate_all_official
from mt3_scene_package import save_scene_package
from mt3_icp_registration import save_icp_outputs
from mt3_place_generalization import (
    compute_place_target,
    parse_place_direction,
)

try:
    from llm_retriever import LLMSemanticRetriever, _contains_any, ACTION_SYNONYMS
except Exception:
    LLMSemanticRetriever = None
    _contains_any = None
    ACTION_SYNONYMS = {}


# Path to official MT3 demo data (if available)
OFFICIAL_DEMO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "mt3", "learning_thousand_tasks",
    "assets", "demonstrations")


SHARED_EXPERIMENT_LOG_DIR = (
    "/mnt/hgfs2/code/learning_thousand_tasks/demo_library/experiment_logs"
)


def _default_experiment_log_dir(code_dir):
    """Keep compact trial tables on the Windows shared folder when mounted."""
    shared_parent = os.path.dirname(SHARED_EXPERIMENT_LOG_DIR)
    if os.path.isdir(shared_parent):
        return SHARED_EXPERIMENT_LOG_DIR
    return os.path.join(code_dir, "demo_library", "experiment_logs")


class MT3Pipeline:
    def __init__(self):
        rospy.init_node("mt3_pipeline", anonymous=True)

        self.language_query = rospy.get_param("~query", "pick up the green cube")
        self.lang_weight = rospy.get_param("~lang_weight", 0.3)
        self.geo_weight = rospy.get_param("~geo_weight", 0.7)
        self.retrieval_mode = rospy.get_param("~retrieval_mode", "hierarchical")
        self.use_perception = rospy.get_param("~use_perception", True)
        self.dry_run = rospy.get_param("~dry_run", False)
        self.require_depth_pose = rospy.get_param("~require_depth_pose", True)
        self.export_scene_packages = rospy.get_param("~export_scene_packages", True)
        self.run_icp = rospy.get_param("~run_icp", True)
        self.use_icp_object_pose = rospy.get_param("~use_icp_object_pose", False)
        self.use_demo_replay = rospy.get_param("~use_demo_replay", False)
        self.use_top_grasp_replay = rospy.get_param(
            "~use_top_grasp_replay", False)
        self.use_side_staged_replay = rospy.get_param(
            "~use_side_staged_replay", False)
        self.prefer_pose_replay = rospy.get_param("~prefer_pose_replay", True)
        self.use_segmented_replay = rospy.get_param("~use_segmented_replay", False)
        self.close_on_replay_blocked = rospy.get_param("~close_on_replay_blocked", False)
        self.replay_close_on_blocked_min_progress = float(rospy.get_param(
            "~replay_close_on_blocked_min_progress", 0.35))
        self.auto_record_success = rospy.get_param("~auto_record_success", True)
        self.auto_log_experiment = rospy.get_param("~auto_log_experiment", True)
        self.archive_trial_scene = rospy.get_param("~archive_trial_scene", True)
        self.execution_environment = str(rospy.get_param(
            "~execution_environment",
            os.environ.get("MT3_EXECUTION_ENVIRONMENT", "simulation"))
        ).strip().lower() or "simulation"
        self.object_shape = rospy.get_param(
            "~object_shape",
            rospy.get_param("/mt3_current_object_shape", "unknown"))
        self.object_label = rospy.get_param(
            "~object_label",
            rospy.get_param("/mt3_current_object_label", self.object_shape))
        self.trial_note = rospy.get_param("~trial_note", "")
        self.condition_id = rospy.get_param("~condition_id", "")
        self.repeat_id = rospy.get_param("~repeat_id", "")
        self.method_variant = rospy.get_param("~method_variant", "full")
        self.object_shape, self.object_label = self._resolve_object_metadata(
            self.object_shape, self.object_label)
        self.object_long_axis_local = str(rospy.get_param(
            "~object_long_axis_local", "")).strip().lower()
        self.real_demo_package_name = rospy.get_param(
            "~real_demo_package_name", "demo_cube_top_grasp_v2_real")
        self.scene_package_dir = rospy.get_param(
            "~scene_package_dir",
            os.path.join(os.path.dirname(__file__), "demo_library", "scene_packages"))
        self.experiment_log_dir = rospy.get_param(
            "~experiment_log_dir",
            _default_experiment_log_dir(os.path.dirname(__file__)))
        experiment_group = str(rospy.get_param(
            "~experiment_group", "")).strip()
        self.experiment_group = ""
        if experiment_group:
            safe_group = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in experiment_group)
            self.experiment_group = safe_group
            self.experiment_log_dir = os.path.join(
                self.experiment_log_dir, safe_group)
        self.rollout_trajectory_dir = rospy.get_param(
            "~rollout_trajectory_dir",
            os.path.join(os.path.dirname(__file__), "demo_library", "rollout_trajectories"))
        self.auto_record_dir = rospy.get_param(
            "~auto_record_dir",
            os.path.join(
                os.path.dirname(__file__), "demo_library",
                self.execution_environment, "auto_recorded"))
        self.use_height_aware_top_grasp = rospy.get_param("~use_height_aware_top_grasp", True)
        self.use_pointcloud_yaw = rospy.get_param("~use_pointcloud_yaw", True)
        self.pointcloud_yaw_shapes = rospy.get_param(
            "~pointcloud_yaw_shapes",
            ["rectangular_prism", "cuboid"])
        self.top_grasp_clearance = float(rospy.get_param("~top_grasp_clearance", 0.030))
        self.object_z_min = rospy.get_param("~object_z_min", -0.63)
        self.object_z_max = rospy.get_param("~object_z_max", -0.40)
        self.object_size = rospy.get_param(
            "~object_size",
            rospy.get_param("/mt3_current_object_size", [0.045, 0.045, 0.045])
        )
        if len(self.object_size) != 3:
            rospy.logwarn("~object_size should have 3 values; using 4.5cm cube default")
            self.object_size = [0.045, 0.045, 0.045]
        self.object_size = [float(v) for v in self.object_size]
        rospy.set_param("~object_height", self.object_size[2])
        safe_label = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in str(self.object_label or "object"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.condition_id:
            condition_code = str(self.condition_id).split("_", 1)[0]
            safe_condition = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in condition_code)
            repeat_text = str(self.repeat_id or "NA")
            safe_repeat = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in repeat_text)
            self.trial_id = "%s_R%s_%s_%s" % (
                safe_condition, safe_repeat, stamp, safe_label)
        else:
            self.trial_id = "%s_%s" % (stamp, safe_label)

        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 Pipeline (official 7-step flow)")
        rospy.loginfo(f"  Query: '{self.language_query}'")
        rospy.loginfo(f"  Dry run: {self.dry_run}")
        rospy.loginfo(f"  Demo replay: {self.use_demo_replay}")
        rospy.loginfo(f"  Top grasp replay: {self.use_top_grasp_replay}")
        rospy.loginfo(f"  Prefer pose replay: {self.prefer_pose_replay}")
        rospy.loginfo(f"  Segmented replay: {self.use_segmented_replay}")
        rospy.loginfo(f"  Close on replay blocked: {self.close_on_replay_blocked}")
        rospy.loginfo(f"  Object size: {self.object_size}")
        rospy.loginfo(f"  Object label: {self.object_label} shape={self.object_shape}")
        rospy.loginfo(f"  Execution environment: {self.execution_environment}")
        rospy.loginfo("=" * 60)

        self.library = DemoLibrary(
            execution_environment=self.execution_environment)
        self.aligner = TrajectoryAligner()
        self.perception = PerceptionNode() if self.use_perception else None
        self.frame_no = 0
        self.last_rollout_trajectory_path = None
        self.last_demo_replay_path = None
        self.last_execution_failure_stage = ""
        self.last_execution_failure_reason = ""
        self.last_planning_success = ""
        self.last_executor_success = ""
        self.pre_execution_gazebo_pose = None
        self.last_postcheck_info = {}
        self._timing = {"init_time": time.time()}
        self.task_type = self._infer_task_type()
        rospy.loginfo(f"  Task type: {self.task_type}")

    def _reset_executor_timing_params(self):
        for name, value in [
                ("/sawyer_auto_grasp/planning_time_s", 0.0),
                ("/sawyer_auto_grasp/robot_execution_time_s", 0.0),
                ("/sawyer_auto_grasp/planning_call_count", 0),
                ("/sawyer_auto_grasp/robot_execution_call_count", 0),
                ("/sawyer_auto_grasp/timing_source", "parent_reset"),
                ("/sawyer_auto_grasp/used_recovery_logic", False),
                ("/sawyer_auto_grasp/replay_recovery_progress", ""),
                ("/sawyer_auto_grasp/replay_recovery_stage", ""),
                ("/sawyer_auto_grasp/execution_variant", "standard_replay"),
                ("/sawyer_auto_grasp/before_close_mouth_center_xy", ["", ""]),
                ("/sawyer_auto_grasp/before_close_mouth_error_xy", ["", ""]),
                ("/sawyer_auto_grasp/before_close_mouth_x", ""),
                ("/sawyer_auto_grasp/before_close_mouth_y", ""),
                ("/sawyer_auto_grasp/before_close_mouth_error_x_m", ""),
                ("/sawyer_auto_grasp/before_close_mouth_error_y_m", ""),
                ("/sawyer_auto_grasp/before_close_mouth_error_xy_m", "")]:
            try:
                rospy.set_param(name, value)
            except Exception:
                pass

    def _read_executor_timing_params(self):
        def _get_float(name):
            try:
                return float(rospy.get_param(name, ""))
            except Exception:
                return ""

        def _get_int(name):
            try:
                return int(rospy.get_param(name, ""))
            except Exception:
                return ""

        def _get_bool(name):
            try:
                value = rospy.get_param(name, "")
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in ("true", "1", "yes")
            except Exception:
                return ""

        def _get_vec(name, length=2):
            try:
                value = rospy.get_param(name, "")
                if isinstance(value, (list, tuple)):
                    out = []
                    for i in range(length):
                        try:
                            out.append(float(value[i]))
                        except Exception:
                            out.append("")
                    return out
            except Exception:
                pass
            return ["" for _ in range(length)]

        return {
            "planning_time_s": _get_float("/sawyer_auto_grasp/planning_time_s"),
            "robot_execution_time_s": _get_float(
                "/sawyer_auto_grasp/robot_execution_time_s"),
            "planning_call_count": _get_int(
                "/sawyer_auto_grasp/planning_call_count"),
            "robot_execution_call_count": _get_int(
                "/sawyer_auto_grasp/robot_execution_call_count"),
            "timing_source": str(rospy.get_param(
                "/sawyer_auto_grasp/timing_source", "")),
            "used_recovery_logic": _get_bool(
                "/sawyer_auto_grasp/used_recovery_logic"),
            "replay_recovery_progress": _get_float(
                "/sawyer_auto_grasp/replay_recovery_progress"),
            "replay_recovery_stage": str(rospy.get_param(
                "/sawyer_auto_grasp/replay_recovery_stage", "")),
            "execution_variant": str(rospy.get_param(
                "/sawyer_auto_grasp/execution_variant", "")),
            "before_close_mouth_center_xy": _get_vec(
                "/sawyer_auto_grasp/before_close_mouth_center_xy", 2),
            "before_close_mouth_error_xy": _get_vec(
                "/sawyer_auto_grasp/before_close_mouth_error_xy", 2),
            "before_close_mouth_x": _get_float(
                "/sawyer_auto_grasp/before_close_mouth_x"),
            "before_close_mouth_y": _get_float(
                "/sawyer_auto_grasp/before_close_mouth_y"),
            "before_close_mouth_error_x_m": _get_float(
                "/sawyer_auto_grasp/before_close_mouth_error_x_m"),
            "before_close_mouth_error_y_m": _get_float(
                "/sawyer_auto_grasp/before_close_mouth_error_y_m"),
            "before_close_mouth_error_xy_m": _get_float(
                "/sawyer_auto_grasp/before_close_mouth_error_xy_m"),
        }

        # ── 任务类型判断 ────────────────────────────────────────
    def _infer_task_type(self):
        """从语言指令推断任务类型: grasp / pick_place."""
        if _contains_any and _contains_any(self.language_query, ACTION_SYNONYMS.get("place", set())):
            return "pick_place"
        if any(k in (self.language_query or "").lower()
               for k in ["place", "put", "放", "放置", "放到", "放在", "搬到", "移到"]):
            return "pick_place"
        return "grasp"

    def _resolve_object_metadata(self, object_shape, object_label):
        """Infer experiment labels conservatively when the user did not pass them."""
        shape = str(object_shape or "unknown")
        label = str(object_label or shape or "unknown")
        if shape != "unknown" and label != "unknown":
            return shape, label

        inferred_label = None
        try:
            from gazebo_msgs.msg import ModelStates
            msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
            candidates = []
            for name in msg.name:
                low = name.lower()
                if any(k in low for k in [
                        "green_cube", "cube", "rectangular", "prism",
                        "cylinder", "sphere", "ellipsoid", "block"]):
                    if "sawyer" not in low and "table" not in low and "ground" not in low:
                        candidates.append(name)
            if candidates:
                inferred_label = candidates[-1]
        except Exception:
            inferred_label = None

        text = (self.language_query + " " + (inferred_label or "")).lower()
        inferred_shape = "unknown"
        if any(k in text for k in ["ellipsoid", "ellipse", "oval", "椭圆"]):
            inferred_shape = "ellipsoid"
        elif any(k in text for k in ["cylinder", "圆柱"]):
            inferred_shape = "cylinder"
        elif any(k in text for k in ["sphere", "ball", "球"]):
            inferred_shape = "sphere"
        elif any(k in text for k in ["rectangular", "prism", "cuboid", "长方体"]):
            inferred_shape = "rectangular_prism"
        elif any(k in text for k in ["cube", "block", "方块", "正方体"]):
            inferred_shape = "cube"

        if shape == "unknown":
            shape = inferred_shape
        if label == "unknown":
            label = inferred_label or (
                "green_%s" % shape if shape != "unknown" else "unknown")
        return shape, label

    def _record_experiment_result(
            self, outcome, test_data=None, aligned=None, best_demo=None, score=None,
            live_package=None, icp_result=None, reason=""):
        """Append one compact row for thesis/experiment bookkeeping."""
        if not self.auto_log_experiment:
            return None
        # dry_run still logs to CSV (perception validation is useful data),
        # but the outcome is always "dry_run" — never "success" or "failed".
        if self.dry_run:
            outcome = "dry_run"
            reason = ""

        def _safe_read_param(name, default=""):
            try:
                return float(rospy.get_param(name, default))
            except Exception:
                return default

        def _safe_read_str_param(name, default=""):
            try:
                if rospy.has_param(name):
                    return str(rospy.get_param(name))
            except Exception:
                pass
            return default

        def _vec(values, length=3):
            if values is None:
                return ["" for _ in range(length)]
            out = []
            for i in range(length):
                try:
                    out.append(float(values[i]))
                except Exception:
                    out.append("")
            return out

        def _json_vec(values, length=3):
            return json.dumps(_vec(values, length), ensure_ascii=False)

        def _replay_info():
            replay_requested = bool(rospy.get_param('/sawyer_auto_grasp/use_demo_replay', False))
            if rospy.has_param('/sawyer_auto_grasp/replay_executed'):
                replay_used = bool(rospy.get_param('/sawyer_auto_grasp/replay_executed', False))
            else:
                replay_used = replay_requested
            replay_type = str(rospy.get_param('/sawyer_auto_grasp/replay_type', ""))
            release_index = ""
            path = self.last_demo_replay_path or str(rospy.get_param(
                '/sawyer_auto_grasp/demo_replay_trajectory_path', ''))
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if not replay_type:
                        replay_type = payload.get(
                            "trajectory_source", payload.get("format", ""))
                    traj = payload.get("trajectory", {}) or {}
                    release_index = payload.get(
                        "release_index", traj.get("release_index", ""))
                except Exception:
                    pass
            return replay_used, replay_type, release_index

        def _normalize_deg(angle):
            angle = float(angle)
            while angle > 180.0:
                angle -= 360.0
            while angle <= -180.0:
                angle += 360.0
            return angle

        def _axis_diff_deg(a, b):
            try:
                diff = _normalize_deg(float(a) - float(b))
                if diff > 90.0:
                    diff -= 180.0
                elif diff < -90.0:
                    diff += 180.0
                return abs(diff)
            except Exception:
                return ""

        def _failure_category(stage, reason_text):
            text = ("%s %s" % (stage or "", reason_text or "")).lower()
            if not text.strip():
                return ""
            if any(k in text for k in ["perception", "mask", "object not detected", "safety_gate"]):
                return "perception_or_pose_failure"
            if any(k in text for k in ["icp", "pose", "yaw", "alignment"]):
                return "pose_estimation_failure"
            if any(k in text for k in ["no_motion_plan", "planning", "plan failed"]):
                return "motion_planning_failure"
            if any(k in text for k in ["controller", "control_failed", "path_tolerance", "aborted"]):
                return "controller_execution_failure"
            if "replay" in text or "bottleneck" in text or "cartesian" in text:
                return "replay_failure"
            if any(k in text for k in ["grasp", "rollout_reported_success_false"]):
                return "grasp_failure"
            if any(k in text for k in ["place", "placement", "release"]):
                return "placement_failure"
            if any(k in text for k in ["postcheck", "post-check", "verification", "后验"]):
                return "task_verification_failure"
            return "other_execution_failure"

        def _gazebo_long_axis_yaw_deg(gazebo_pose, size_xyz, shape):
            if not gazebo_pose or gazebo_pose.get("yaw_deg", "") == "":
                return ""
            try:
                yaw = float(gazebo_pose["yaw_deg"])
            except Exception:
                return ""

            shape = str(shape or "").lower()
            if shape in ("rectangular_prism", "cuboid", "box"):
                if self.object_long_axis_local == "y":
                    yaw += 90.0
                elif self.object_long_axis_local == "x":
                    pass
                else:
                    # Fall back to the configured model dimensions, not the
                    # live OBB dimensions. OBB axes may be sorted/reoriented
                    # and therefore do not identify the SDF's local axes.
                    sx, sy, _ = _vec(self.object_size)
                    try:
                        if float(sy) > float(sx):
                            yaw += 90.0
                    except Exception:
                        yaw += 90.0
            return _normalize_deg(yaw)

        try:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            csv_path = os.path.join(self.experiment_log_dir, "mt3_trials.csv")
            jsonl_path = os.path.join(self.experiment_log_dir, "mt3_trials.jsonl")

            pose = (test_data or {}).get("pose", {})
            obj_base = (aligned or {}).get("object_pose_base", {})
            grasp = (aligned or {}).get("grasp_pose", {})
            obj_pos = _vec(obj_base.get("position"))
            grasp_pos = _vec(grasp.get("position"))
            bn = (aligned or {}).get("bottleneck_pose", grasp)
            bn_pos = _vec((bn or {}).get("position"))
            obj_size = _vec(obj_base.get("estimated_object_size", self.object_size))

            metrics = (icp_result or {}).get("metrics", {}) if icp_result else {}
            live_stats = (live_package or {}).get("stats", {}) if live_package else {}
            best_demo = best_demo or {}
            expected_demo_id = str(rospy.get_param("~expected_demo_id", ""))
            retrieval_meta = getattr(self.library, "last_retrieval_metadata", {}) or {}
            yaw_info = (aligned or {}).get("pointcloud_yaw_alignment", {})
            top_z_info = (aligned or {}).get("pointcloud_top_z_mapping", {})
            obb_info = (aligned or {}).get("pointcloud_obb_center_alignment", {})
            requested_x = _safe_read_param("~x", "")
            requested_y = _safe_read_param("~y", "")
            requested_z = _safe_read_param("~z", "")
            requested_yaw_deg = _safe_read_param("~yaw_deg", "")
            requested_position_xyz = [requested_x, requested_y, requested_z]
            final_gazebo_pose = self._get_gazebo_object_pose()
            gazebo_pose = self.pre_execution_gazebo_pose or final_gazebo_pose
            gazebo_gt_world_xyz = [
                (gazebo_pose or {}).get("x", ""),
                (gazebo_pose or {}).get("y", ""),
                (gazebo_pose or {}).get("z", ""),
            ]
            final_gazebo_xyz = [
                (final_gazebo_pose or {}).get("x", ""),
                (final_gazebo_pose or {}).get("y", ""),
                (final_gazebo_pose or {}).get("z", ""),
            ]
            # Gazebo model z is in the Gazebo world/model frame, while perception
            # estimates are in Sawyer base. Keep x/y for table-plane error only.
            gazebo_gt_base_xy_xyz = [
                (gazebo_pose or {}).get("x", ""),
                (gazebo_pose or {}).get("y", ""),
                "",
            ]
            gazebo_yaw_raw_deg = (gazebo_pose or {}).get("yaw_deg", "")
            gazebo_yaw_long_axis_deg = _gazebo_long_axis_yaw_deg(
                gazebo_pose, obj_size, self.object_shape)
            target_yaw_gt_deg = (
                gazebo_yaw_long_axis_deg
                if gazebo_yaw_long_axis_deg != ""
                else gazebo_yaw_raw_deg
            )
            yaw_error_method = ""
            if target_yaw_gt_deg != "":
                shape_name = str(self.object_shape or "").lower()
                yaw_error_method = (
                    "gazebo_long_axis_parallel_180"
                    if shape_name in ("rectangular_prism", "cuboid", "box")
                    else "gazebo_raw_parallel_180"
                )

            retrieval_correct = ""
            if expected_demo_id:
                retrieval_correct = str(best_demo.get("id", "") == expected_demo_id)

            position_xy_error = ""
            if gazebo_pose and obj_pos[0] != "" and obj_pos[1] != "":
                try:
                    dx = float(obj_pos[0]) - float(gazebo_pose["x"])
                    dy = float(obj_pos[1]) - float(gazebo_pose["y"])
                    position_xy_error = math.sqrt(dx * dx + dy * dy)
                except Exception:
                    position_xy_error = ""
            requested_position_xy_error = ""
            if requested_x != "" and requested_y != "" and obj_pos[0] != "" and obj_pos[1] != "":
                try:
                    dx = float(obj_pos[0]) - float(requested_x)
                    dy = float(obj_pos[1]) - float(requested_y)
                    requested_position_xy_error = math.sqrt(dx * dx + dy * dy)
                except Exception:
                    requested_position_xy_error = ""
            target_error_source = ""
            target_error_xy_m = position_xy_error
            if position_xy_error != "":
                target_error_source = "gazebo_model_states"
            elif requested_position_xy_error != "":
                target_error_xy_m = requested_position_xy_error
                target_error_source = "requested_condition"

            yaw_error_deg = ""
            if gazebo_pose and yaw_info.get("live_yaw_deg", "") != "" and target_yaw_gt_deg != "":
                yaw_error_deg = _axis_diff_deg(yaw_info.get("live_yaw_deg"), target_yaw_gt_deg)
            replay_used, replay_type, release_index = _replay_info()
            failure_stage_value = (
                "" if outcome in ("success", "dry_run")
                else str(self.last_execution_failure_stage or reason or "execution")
            )
            failure_reason_value = (
                "" if outcome in ("success", "dry_run")
                else str(self.last_execution_failure_reason or reason)
            )
            place_xyz = [
                _safe_read_param('/sawyer_auto_grasp/place_x'),
                _safe_read_param('/sawyer_auto_grasp/place_y'),
                _safe_read_param('/sawyer_auto_grasp/place_z'),
            ]
            has_place = self.task_type == "pick_place"
            postcheck = dict(getattr(self, "last_postcheck_info", {}) or {})
            final_target_error_xy_m = postcheck.get("final_target_error_xy_m", "")
            if final_target_error_xy_m == "" and has_place and final_gazebo_pose:
                try:
                    dx = float(final_gazebo_pose["x"]) - float(place_xyz[0])
                    dy = float(final_gazebo_pose["y"]) - float(place_xyz[1])
                    final_target_error_xy_m = math.sqrt(dx * dx + dy * dy)
                except Exception:
                    final_target_error_xy_m = ""
            elif final_target_error_xy_m == "" and final_gazebo_pose and obj_pos[0] != "" and obj_pos[1] != "":
                try:
                    dx = float(final_gazebo_pose["x"]) - float(obj_pos[0])
                    dy = float(final_gazebo_pose["y"]) - float(obj_pos[1])
                    final_target_error_xy_m = math.sqrt(dx * dx + dy * dy)
                except Exception:
                    final_target_error_xy_m = ""
            total_time_s = ""
            if self._timing.get("run_start"):
                total_time_s = time.time() - float(self._timing["run_start"])

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unix_time": "%.3f" % time.time(),
                "trial_id": self.trial_id,
                "query": self.language_query,
                "condition_id": str(self.condition_id),
                "repeat_id": str(self.repeat_id),
                "method_variant": str(self.method_variant),
                "object_label": str(self.object_label),
                "object_shape": str(self.object_shape),
                "target_shape": str(self.object_shape),
                "trial_note": str(self.trial_note),
                "dry_run": bool(self.dry_run),
                "outcome": outcome,
                "success": bool(outcome == "success"),
                "task_success": bool(outcome == "success"),
                "postcheck_success": postcheck.get(
                    "postcheck_success",
                    bool(outcome == "success") if outcome != "dry_run" else ""),
                "reason": reason,
                "failure_stage": failure_stage_value,
                "failure_reason": failure_reason_value,
                "failure_category": _failure_category(
                    failure_stage_value, failure_reason_value),
                "retrieved_demo_id": best_demo.get("id", ""),
                "retrieval_score": "" if score is None else float(score),
                "retrieval_mode": retrieval_meta.get("retrieval_mode", self.retrieval_mode),
                "retrieval_task_type_filter": retrieval_meta.get("task_type_filter", ""),
                "retrieval_language_score": retrieval_meta.get("language_score", ""),
                "retrieval_geometry_score": retrieval_meta.get("geometry_score", ""),
                "language_score": retrieval_meta.get("language_score", ""),
                "geometry_score": retrieval_meta.get("geometry_score", ""),
                "retrieval_candidates": json.dumps(
                    retrieval_meta.get("geometric_candidates", [])[:5],
                    ensure_ascii=False),
                "expected_demo_id": expected_demo_id,
                "retrieval_correct": retrieval_correct,
                "requested_x": requested_x,
                "requested_y": requested_y,
                "requested_z": requested_z,
                "requested_yaw_deg": requested_yaw_deg,
                "requested_object_shape": _safe_read_str_param(
                    "~object_shape", str(self.object_shape)),
                "requested_object_label": _safe_read_str_param(
                    "~object_label", str(self.object_label)),
                "requested_object_size": json.dumps(self.object_size, ensure_ascii=False),
                "requested_position_xyz": json.dumps(requested_position_xyz, ensure_ascii=False),
                "perception_method": pose.get("method", ""),
                "pose_source_frame": pose.get("source_frame", pose.get("frame", "")),
                "object_base_x": obj_pos[0],
                "object_base_y": obj_pos[1],
                "object_base_z": obj_pos[2],
                "target_est_xyz": _json_vec(obj_pos),
                "gazebo_model_name": (gazebo_pose or {}).get("name", ""),
                "gazebo_pose_found": bool(gazebo_pose),
                "gazebo_x": (gazebo_pose or {}).get("x", ""),
                "gazebo_y": (gazebo_pose or {}).get("y", ""),
                "gazebo_z": (gazebo_pose or {}).get("z", ""),
                "gazebo_yaw_deg": gazebo_yaw_raw_deg,
                "initial_object_xyz": json.dumps(gazebo_gt_world_xyz, ensure_ascii=False),
                "final_object_xyz": json.dumps(final_gazebo_xyz, ensure_ascii=False),
                "final_object_model_name": (final_gazebo_pose or {}).get("name", ""),
                "final_target_error_xy_m": final_target_error_xy_m,
                "final_lift_delta_m": postcheck.get("final_lift_delta_m", ""),
                "final_relation_error_xy_m": postcheck.get("final_relation_error_xy_m", ""),
                "insert_depth_m": postcheck.get("insert_depth_m", ""),
                "postcheck_reason": postcheck.get("postcheck_reason", ""),
                "target_gt_xyz": json.dumps(gazebo_gt_base_xy_xyz, ensure_ascii=False),
                "target_gt_world_xyz": json.dumps(gazebo_gt_world_xyz, ensure_ascii=False),
                "target_gt_frame": "gazebo_world_xy_only",
                "target_est_frame": "base",
                "target_error_z_m": "",
                "target_error_xyz_m": "",
                "target_yaw_gt_deg": target_yaw_gt_deg,
                "target_yaw_gt_raw_deg": gazebo_yaw_raw_deg,
                "target_yaw_gt_long_axis_deg": gazebo_yaw_long_axis_deg,
                "target_yaw_est_deg": yaw_info.get("live_yaw_deg", ""),
                "position_xy_error_m": position_xy_error,
                "requested_position_xy_error_m": requested_position_xy_error,
                "target_error_xy_m": target_error_xy_m,
                "target_error_source": target_error_source,
                "grasp_x": grasp_pos[0],
                "grasp_y": grasp_pos[1],
                "grasp_z": grasp_pos[2],
                "grasp_xyz": _json_vec(grasp_pos),
                "bottleneck_xyz": _json_vec(bn_pos),
                "estimated_size_x": obj_size[0],
                "estimated_size_y": obj_size[1],
                "estimated_size_z": obj_size[2],
                "target_size_xyz": _json_vec(obj_size),
                "demo_yaw_deg": yaw_info.get("demo_yaw_deg", ""),
                "live_yaw_deg": yaw_info.get("live_yaw_deg", ""),
                "delta_yaw_deg": yaw_info.get("delta_yaw_deg", ""),
                "yaw_error_deg": yaw_error_deg,
                "yaw_error_method": yaw_error_method,
                "demo_top_z": top_z_info.get("demo_top_z", ""),
                "live_top_z": top_z_info.get("live_top_z", ""),
                "demo_grasp_z": top_z_info.get("demo_grasp_z", ""),
                "demo_clearance_above_top": top_z_info.get("demo_clearance_above_top", ""),
                "mapped_grasp_z": top_z_info.get("mapped_grasp_z", ""),
                "obb_old_x": obb_info.get("old_x", ""),
                "obb_old_y": obb_info.get("old_y", ""),
                "obb_center_x": obb_info.get("obb_center_x", ""),
                "obb_center_y": obb_info.get("obb_center_y", ""),
                "obb_correction_dx": obb_info.get("dx", ""),
                "obb_correction_dy": obb_info.get("dy", ""),
                "obb_extent_long": obb_info.get("extent_long", ""),
                "obb_extent_short": obb_info.get("extent_short", ""),
                "icp_median_error_m": metrics.get("median_error_m", ""),
                "icp_p90_error_m": metrics.get("p90_error_m", ""),
                "icp_mean_error_m": metrics.get("mean_error_m", ""),
                "icp_iterations": metrics.get("iterations", ""),
                "mask_pixels": live_stats.get("segmap_pixels", ""),
                "pointcloud_points": live_stats.get("pointcloud_points", ""),
                "live_scene_package": (live_package or {}).get("name", ""),
                "live_scene_package_dir": (live_package or {}).get("package_dir", ""),
                "scene_package_path": (live_package or {}).get("package_dir", ""),
                "icp_output_dir": (icp_result or {}).get("output_dir", "") if icp_result else "",
                # ── 放置任务字段 ─────────────────────────────────
                "task_type": str(self.task_type),
                "place_mode": str(rospy.get_param('/sawyer_auto_grasp/place_mode', '')),
                "place_x": place_xyz[0],
                "place_y": place_xyz[1],
                "place_z": place_xyz[2],
                "place_or_insert_xyz": json.dumps(place_xyz if has_place else ["", "", ""], ensure_ascii=False),
                "place_direction": str(rospy.get_param('/sawyer_auto_grasp/place_direction', '')),
                "place_resolution_method": str(rospy.get_param('/sawyer_auto_grasp/place_resolution_method', '')),
                "place_resolution_confidence": _safe_read_param('/sawyer_auto_grasp/place_resolution_confidence'),
                "place_offset_xy": str(rospy.get_param('/sawyer_auto_grasp/place_offset_xy', '')),
                "replay_used": bool(replay_used),
                "replay_type": replay_type,
                "used_recovery_logic": self._timing.get("used_recovery_logic", ""),
                "replay_recovery_progress": self._timing.get("replay_recovery_progress", ""),
                "replay_recovery_stage": self._timing.get("replay_recovery_stage", ""),
                "execution_variant": self._timing.get("execution_variant", ""),
                "before_close_mouth_center_xy": json.dumps(
                    self._timing.get("before_close_mouth_center_xy", ["", ""]),
                    ensure_ascii=False),
                "before_close_mouth_error_xy": json.dumps(
                    self._timing.get("before_close_mouth_error_xy", ["", ""]),
                    ensure_ascii=False),
                "before_close_mouth_x": self._timing.get("before_close_mouth_x", ""),
                "before_close_mouth_y": self._timing.get("before_close_mouth_y", ""),
                "before_close_mouth_error_x_m": self._timing.get(
                    "before_close_mouth_error_x_m", ""),
                "before_close_mouth_error_y_m": self._timing.get(
                    "before_close_mouth_error_y_m", ""),
                "before_close_mouth_error_xy_m": self._timing.get(
                    "before_close_mouth_error_xy_m", ""),
                "release_index": release_index,
                "planning_success": self.last_planning_success,
                "execution_success": (
                    self.last_executor_success
                    if self.last_executor_success != ""
                    else bool(outcome == "success")),
                "manual_success_label": "",
                "total_time_s": total_time_s,
                "perception_time_s": self._timing.get("perception_time_s", ""),
                "retrieval_time_s": self._timing.get("retrieval_time_s", ""),
                "alignment_time_s": self._timing.get("alignment_time_s", ""),
                "planning_time_s": self._timing.get("planning_time_s", ""),
                "robot_execution_time_s": self._timing.get("robot_execution_time_s", ""),
                "execution_wall_time_s": self._timing.get("execution_time_s", ""),
                "planning_call_count": self._timing.get("planning_call_count", ""),
                "robot_execution_call_count": self._timing.get("robot_execution_call_count", ""),
                "timing_source": self._timing.get("timing_source", ""),
                "execution_time_s": self._timing.get("execution_time_s", ""),
                "rollout_path": self.last_rollout_trajectory_path or "",
            }

            self._append_experiment_csv(csv_path, row)

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            rospy.loginfo("  Experiment trial logged: %s", csv_path)
            return csv_path
        except Exception as e:
            rospy.logwarn("  Failed to log experiment trial: %s", e)
            return None

    def _append_experiment_csv(self, csv_path, row):
        """Append a row while preserving older rows when the schema grows."""
        fieldnames = list(row.keys())
        existing_rows = []
        existing_fields = []

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_fields = list(reader.fieldnames or [])
                    for old in reader:
                        clean = {k: v for k, v in old.items() if k is not None}
                        existing_rows.append(clean)
            except Exception as exc:
                backup = csv_path + ".schema_backup_%s" % time.strftime("%Y%m%d_%H%M%S")
                shutil.copy2(csv_path, backup)
                rospy.logwarn("  Could not read existing CSV; backed up to %s: %s", backup, exc)
                existing_rows = []
                existing_fields = []

        merged_fields = []
        for name in existing_fields + fieldnames:
            if name and name not in merged_fields:
                merged_fields.append(name)

        rewrite = merged_fields != existing_fields
        mode = "w" if rewrite else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields)
            if rewrite or not existing_fields:
                writer.writeheader()
                for old in existing_rows:
                    writer.writerow({k: old.get(k, "") for k in merged_fields})
            writer.writerow({k: row.get(k, "") for k in merged_fields})

    def _get_gazebo_object_pose(self):
        """Read the current Gazebo model pose for experiment error metrics."""
        try:
            from gazebo_msgs.msg import ModelStates
            explicit = str(rospy.get_param("~gazebo_model_name", "")).strip()
            msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=0.5)

            names = list(msg.name)
            chosen = None
            if explicit and explicit in names:
                chosen = explicit
            else:
                label = str(self.object_label or "").lower()
                shape = str(self.object_shape or "").lower()

                def _score(name):
                    low = name.lower()
                    if any(skip in low for skip in ["sawyer", "table", "workbench", "ground"]):
                        return -100
                    shape_tokens = {
                        "rectangular_prism": ["rectangular", "prism", "cuboid"],
                        "cuboid": ["rectangular", "prism", "cuboid"],
                        "cube": ["cube", "grasp_object"],
                        "cylinder": ["cylinder", "insert_cylinder"],
                        "sphere": ["sphere", "ellipsoid", "ball"],
                        "ellipsoid": ["sphere", "ellipsoid", "ball"],
                    }
                    required = shape_tokens.get(shape, [])
                    if required and not any(token in low for token in required):
                        # A colour match alone must never make a model of the
                        # wrong shape eligible as Gazebo ground truth.
                        return -50
                    score = 0
                    for token in [label, shape, "green"]:
                        token = token.strip()
                        if token and token != "unknown" and token in low:
                            score += 3
                    if shape in ["rectangular_prism", "cuboid"] and any(k in low for k in ["rectangular", "prism", "cuboid"]):
                        score += 8
                    elif shape == "cube" and ("cube" in low or "grasp_object" in low):
                        score += 8
                    elif shape == "cylinder" and ("cylinder" in low or "insert_cylinder" in low):
                        score += 8
                    elif shape in ["sphere", "ellipsoid"] and any(k in low for k in ["sphere", "ellipsoid", "ball"]):
                        score += 8
                    elif "green" in low:
                        score += 1
                    return score

                ranked = sorted(names, key=_score, reverse=True)
                if ranked and _score(ranked[0]) > 0:
                    chosen = ranked[0]

            if not chosen:
                return None

            idx = names.index(chosen)
            pose = msg.pose[idx]
            q = pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return {
                "name": chosen,
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "yaw_rad": float(yaw),
                "yaw_deg": float(math.degrees(yaw)),
            }
        except Exception:
            return None

    def _validate_post_grasp_success(self, aligned):
        """Use Gazebo final object pose to reject false-positive grasp success."""
        if self.dry_run or self.task_type != "grasp":
            return True

        group = str(getattr(self, "experiment_group", "") or "").lower()
        if group and group not in ("top_grasp", "rotated_top_grasp"):
            return True

        gazebo_pose = self._get_gazebo_object_pose()
        if not gazebo_pose:
            rospy.logwarn(
                "POST-GRASP CHECK: Gazebo pose unavailable; keeping executor result.")
            self.last_postcheck_info = {
                "postcheck_success": "",
                "postcheck_reason": "gazebo_pose_unavailable",
            }
            return True

        min_lift_z = float(rospy.get_param(
            "~post_grasp_min_gazebo_z",
            rospy.get_param("/sawyer_auto_grasp/post_grasp_min_gazebo_z", 0.43)))
        z_margin = float(rospy.get_param(
            "~post_grasp_z_accept_margin_m",
            rospy.get_param("/sawyer_auto_grasp/post_grasp_z_accept_margin_m", 0.010)))
        max_xy_shift = float(rospy.get_param(
            "~post_grasp_max_xy_shift_m",
            rospy.get_param("/sawyer_auto_grasp/post_grasp_max_xy_shift_m", 0.12)))
        min_lift_delta = float(rospy.get_param(
            "~post_grasp_min_lift_delta_m",
            rospy.get_param("/sawyer_auto_grasp/post_grasp_min_lift_delta_m", 0.030)))

        obj = (aligned or {}).get("object_pose_base", {})
        obj_pos = obj.get("position", None)
        xy_shift = ""
        if obj_pos and len(obj_pos) >= 2:
            try:
                dx = float(gazebo_pose["x"]) - float(obj_pos[0])
                dy = float(gazebo_pose["y"]) - float(obj_pos[1])
                xy_shift = math.sqrt(dx * dx + dy * dy)
            except Exception:
                xy_shift = ""

        final_z = float(gazebo_pose["z"])
        lift_delta = ""
        if self.pre_execution_gazebo_pose:
            try:
                lift_delta = final_z - float(self.pre_execution_gazebo_pose["z"])
            except Exception:
                lift_delta = ""
        z_ok = final_z >= min_lift_z
        z_margin_ok = final_z >= (min_lift_z - z_margin)
        lift_ok = (lift_delta != "" and float(lift_delta) >= min_lift_delta)
        xy_ok = (xy_shift == "" or float(xy_shift) <= max_xy_shift)
        rospy.loginfo(
            "POST-GRASP CHECK: model=%s z=%.3f min_z=%.3f z_margin=%.3f "
            "lift_delta=%s min_lift=%.3f xy_shift=%s max_xy=%.3f "
            "z_ok=%s z_margin_ok=%s lift_ok=%s xy_ok=%s",
            gazebo_pose.get("name", ""),
            final_z,
            min_lift_z,
            z_margin,
            ("%.3f" % lift_delta if lift_delta != "" else "n/a"),
            min_lift_delta,
            ("%.3f" % xy_shift if xy_shift != "" else "n/a"),
            max_xy_shift,
            z_ok,
            z_margin_ok,
            lift_ok,
            xy_ok)

        if xy_ok and (z_ok or z_margin_ok or lift_ok):
            self.last_postcheck_info = {
                "postcheck_success": True,
                "postcheck_reason": (
                    "" if z_ok else
                    "accepted_with_z_margin" if z_margin_ok else
                    "accepted_with_lift_delta"),
                "final_target_error_xy_m": xy_shift,
                "final_lift_delta_m": lift_delta,
            }
            return True

        self.last_execution_failure_stage = "grasp_verification"
        if not (z_ok or z_margin_ok or lift_ok):
            self.last_execution_failure_reason = "抓取后目标物体未被有效抬起或发生倾倒"
        else:
            self.last_execution_failure_reason = "抓取后目标物体发生明显位移"
        self.last_postcheck_info = {
            "postcheck_success": False,
            "postcheck_reason": self.last_execution_failure_reason,
            "final_target_error_xy_m": xy_shift,
            "final_lift_delta_m": lift_delta,
        }
        rospy.logerr(
            "POST-GRASP CHECK: rejecting executor success: %s",
            self.last_execution_failure_reason)
        return False

    def _validate_post_place_success(self):
        """Use Gazebo final object pose to verify no-anchor placement."""
        if self.dry_run or self.task_type != "pick_place":
            return True

        gazebo_pose = self._get_gazebo_object_pose()
        if not gazebo_pose:
            rospy.logwarn(
                "POST-PLACE CHECK: Gazebo pose unavailable; keeping executor result.")
            self.last_postcheck_info = {
                "postcheck_success": "",
                "postcheck_reason": "gazebo_pose_unavailable",
            }
            return True

        try:
            place_x = float(rospy.get_param('/sawyer_auto_grasp/place_x'))
            place_y = float(rospy.get_param('/sawyer_auto_grasp/place_y'))
        except Exception:
            rospy.logwarn(
                "POST-PLACE CHECK: place target unavailable; keeping executor result.")
            self.last_postcheck_info = {
                "postcheck_success": "",
                "postcheck_reason": "place_target_unavailable",
            }
            return True

        max_xy_error = float(rospy.get_param(
            "~post_place_max_xy_error_m",
            rospy.get_param("/sawyer_auto_grasp/post_place_max_xy_error_m", 0.060)))
        dx = float(gazebo_pose["x"]) - place_x
        dy = float(gazebo_pose["y"]) - place_y
        err = math.sqrt(dx * dx + dy * dy)
        ok = err <= max_xy_error
        self.last_postcheck_info = {
            "postcheck_success": bool(ok),
            "postcheck_reason": "" if ok else "目标物体最终放置位置偏差过大",
            "final_target_error_xy_m": float(err),
        }
        rospy.loginfo(
            "POST-PLACE CHECK: model=%s final=[%.3f, %.3f] "
            "target=[%.3f, %.3f] err=%.3fm max=%.3fm ok=%s",
            gazebo_pose.get("name", ""), float(gazebo_pose["x"]),
            float(gazebo_pose["y"]), place_x, place_y, err, max_xy_error, ok)
        if ok:
            return True

        self.last_execution_failure_stage = "placement_verification"
        self.last_execution_failure_reason = "目标物体最终放置位置偏差过大"
        rospy.logerr(
            "POST-PLACE CHECK: rejecting executor success: %s",
            self.last_execution_failure_reason)
        return False

    def _validate_perception_for_execution(self, test_data, aligned):
        """Reject unreliable perception before any robot execution."""
        pose = test_data.get("pose", {})
        method = pose.get("method", "unknown")
        obj = aligned.get("object_pose_base", {})
        obj_pos = obj.get("position", None)

        if self.use_perception and self.require_depth_pose and "depth" not in method:
            rospy.logerr(
                f"PERCEPTION SAFETY GATE: rejected pose method={method}. "
                "Expected depth-corrected pose (PnP+depth)."
            )
            return False

        if obj_pos is None or len(obj_pos) < 3:
            rospy.logerr("PERCEPTION SAFETY GATE: missing object_pose_base position.")
            return False

        obj_z = obj_pos[2]
        z_min = float(self.object_z_min)
        z_max = float(self.object_z_max)

        estimated_height = obj.get("estimated_object_height", None)
        if estimated_height is None:
            size = obj.get("estimated_object_size", None)
            if size and len(size) == 3:
                try:
                    estimated_height = max(float(v) for v in size)
                except Exception:
                    estimated_height = None
        try:
            estimated_height = None if estimated_height is None else float(estimated_height)
        except Exception:
            estimated_height = None

        if not self._is_top_down_grasp(aligned) and estimated_height is not None:
            # Tall side-grasp objects are often represented by the visible
            # point-cloud center rather than the tabletop/bottom contact z.
            # Keep the lower bound conservative, but raise the allowed upper
            # bound according to object height so valid tall cylinders are not
            # rejected by the old cube-only gate.
            z_max = max(z_max, self.object_z_min + estimated_height + 0.02)

        if not (z_min <= obj_z <= z_max):
            rospy.logerr(
                f"PERCEPTION SAFETY GATE: object_z={obj_z:.3f} outside "
                f"[{z_min:.3f}, {z_max:.3f}]."
            )
            return False

        rospy.loginfo(
            f"PERCEPTION SAFETY GATE: passed "
            f"(method={method}, object_z={obj_z:.3f}, "
            f"range=[{z_min:.3f}, {z_max:.3f}])"
        )
        return True

    # ================================================================
    # Step 1: Load test image
    # ================================================================
    def step1_load_test_image(self):
        rospy.loginfo("[Step 1/7] Loading test image from camera...")

        if not self.use_perception or self.perception is None:
            rospy.loginfo("  Perception disabled — using default test data")
            return self._default_test_data()

        # Wait for camera image (intrinsics are hardcoded, no need to wait)
        wait_count = 0
        while self.perception.head_image is None and wait_count < 50:
            rospy.sleep(0.2)
            wait_count += 1

        if self.perception.head_image is None:
            rospy.logwarn("  No camera image — using default data")
            return self._default_test_data()

        if getattr(self.perception, "use_pointcloud_pose", False):
            wait_count = 0
            while self.perception.head_points is None and wait_count < 50:
                rospy.sleep(0.2)
                wait_count += 1
            if self.perception.head_points is None:
                rospy.logwarn("  No point cloud yet - pointcloud pose may fall back")

        # Run perception to get detection + pose
        pose = self.perception.get_object_pose()
        bgr = self.perception.latest_bgr
        mask = self.perception.latest_clean_mask

        if pose is None:
            rospy.logwarn("  Object not detected — using default data")
            return self._default_test_data()

        # Build test data dict
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        segmap = mask.astype(bool) if mask is not None else None

        # Create synthetic depth from detection
        depth = self._make_depth_from_pose(pose, bgr.shape if bgr is not None else (480, 640))

        # Get camera intrinsics from perception node
        if self.perception.head_camera_info is not None:
            K = self.perception.head_camera_info.K
            intrinsics = np.array([[K[0], K[1], K[2]],
                                   [K[3], K[4], K[5]],
                                   [K[6], K[7], K[8]]], dtype=np.float64)
        else:
            # Default intrinsics (approximate for Sawyer Gazebo camera)
            intrinsics = np.array([[550.0, 0.0, 320.0],
                                   [0.0, 550.0, 240.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)

        pos = pose["position"]
        rospy.loginfo(f"  Detected: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                      f" method={pose['method']} conf={pose['confidence']:.2f}")

        return {
            "rgb": rgb, "depth": depth, "segmap": segmap,
            "intrinsics": intrinsics, "pose": pose
        }

    def _export_scene_package(self, scene_data, name, role, extra_metadata=None):
        if not self.export_scene_packages:
            return None
        try:
            package_dir = os.path.join(self.scene_package_dir, name)
            if os.path.isdir(package_dir):
                shutil.rmtree(package_dir)
            package = save_scene_package(
                scene_data,
                self.scene_package_dir,
                name=name,
                role=role,
                extra_metadata=extra_metadata)
            rospy.loginfo(
                "  Scene package saved: %s "
                "(points=%d, mask_px=%d)",
                package["package_dir"],
                package["stats"]["pointcloud_points"],
                package["stats"]["segmap_pixels"])
            return package
        except Exception as e:
            rospy.logwarn("  Failed to save scene package '%s': %s", name, e)
            return None

    def _archive_live_scene_package(self, live_package):
        if not self.archive_trial_scene or not live_package:
            return live_package
        try:
            src = live_package.get("package_dir")
            if not src or not os.path.isdir(src):
                return live_package
            archive_name = "live_trial_%s" % self.trial_id
            dst = os.path.join(self.scene_package_dir, archive_name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            archived = dict(live_package)
            archived["name"] = archive_name
            archived["package_dir"] = dst

            metadata_path = os.path.join(dst, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["role"] = "archived_live_trial"
                meta["trial_id"] = self.trial_id
                meta["object_shape"] = self.object_shape
                meta["object_label"] = self.object_label
                meta["trial_note"] = self.trial_note
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            rospy.loginfo("  Archived live trial scene package: %s", dst)
            return archived
        except Exception as e:
            rospy.logwarn("  Failed to archive live scene package: %s", e)
            return live_package

    def _package_has_pointcloud(self, package):
        if not package:
            return False
        package_dir = package.get("package_dir")
        if not package_dir:
            return False
        pointcloud_path = os.path.join(package_dir, "pointcloud.npy")
        if not os.path.exists(pointcloud_path):
            return False
        try:
            points = np.load(pointcloud_path)
            return points.ndim == 2 and points.shape[1] == 3 and len(points) >= 20
        except Exception:
            return False

    def _real_demo_package_for_icp(self):
        package_dir = os.path.join(self.scene_package_dir, self.real_demo_package_name)
        pointcloud_path = os.path.join(package_dir, "pointcloud.npy")
        if os.path.exists(pointcloud_path):
            rospy.loginfo("  ICP using real demo package: %s", package_dir)
            return {"package_dir": package_dir, "name": self.real_demo_package_name}
        rospy.logwarn("  Real demo package not found for ICP: %s", package_dir)
        return None

    def _select_demo_package_for_icp(self, demo_package, best_demo):
        """Prefer the retrieved demo point cloud; fall back to legacy real package."""
        demo_id = (best_demo or {}).get("id", "")
        if self._package_has_pointcloud(demo_package):
            rospy.loginfo("  ICP using retrieved demo package: %s",
                          demo_package["package_dir"])
            return demo_package

        explicit_name = rospy.get_param("~icp_demo_package_name", "")
        if explicit_name:
            package = {
                "package_dir": os.path.join(self.scene_package_dir, explicit_name),
                "name": explicit_name,
            }
            if self._package_has_pointcloud(package):
                rospy.loginfo("  ICP using explicit demo package: %s",
                              package["package_dir"])
                return package
            rospy.logwarn("  Explicit ICP demo package has no usable pointcloud: %s",
                          package["package_dir"])

        if demo_id:
            package = {
                "package_dir": os.path.join(self.scene_package_dir, "demo_" + demo_id),
                "name": "demo_" + demo_id,
            }
            if self._package_has_pointcloud(package):
                rospy.loginfo("  ICP using stored demo package: %s",
                              package["package_dir"])
                return package

        return self._real_demo_package_for_icp() or demo_package

    def _stored_demo_scene_package(self, demo_entry):
        demo_id = (demo_entry or {}).get("id", "")
        if not demo_id:
            return None
        package = {
            "package_dir": os.path.join(self.scene_package_dir, "demo_" + demo_id),
            "name": "demo_" + demo_id,
        }
        if self._package_has_pointcloud(package):
            rospy.loginfo("  Reusing stored demo scene package: %s",
                          package["package_dir"])
            return package
        return None

    def _record_success_demo(self, aligned, best_demo, score, live_package=None, icp_result=None):
        """Save a successful rollout as a reusable demo with its scene package."""
        if not self.auto_record_success or self.dry_run:
            return None
        try:
            recorded_dir = self.auto_record_dir
            os.makedirs(recorded_dir, exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            demo_id = f"auto_success_{best_demo.get('id', 'demo')}_{stamp}"
            obj = aligned.get("object_pose_base", {})
            obj_pos = obj.get("position", [0.6, 0.0, -0.58])
            obj_size = obj.get("estimated_object_size", self.object_size)
            if not obj_size or len(obj_size) != 3:
                obj_size = self.object_size

            grasp = aligned.get("grasp_pose", {})
            grasp_pos = grasp.get("position", obj_pos)
            grasp_ori = grasp.get("orientation", [-1.0, 0.0, 0.0, 0.0])

            saved_scene_package = None
            if live_package and live_package.get("package_dir"):
                src_scene = live_package["package_dir"]
                scene_name = f"demo_{demo_id}"
                dst_scene = os.path.join(self.scene_package_dir, scene_name)
                if os.path.isdir(src_scene):
                    if os.path.isdir(dst_scene):
                        shutil.rmtree(dst_scene)
                    shutil.copytree(src_scene, dst_scene)
                    saved_scene_package = scene_name

                    metadata_path = os.path.join(dst_scene, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            meta["role"] = "auto_success_demo_scene"
                            meta["linked_demo_id"] = demo_id
                            meta["source_live_package"] = live_package.get("name", "live_latest")
                            with open(metadata_path, "w", encoding="utf-8") as f:
                                json.dump(meta, f, indent=2, ensure_ascii=False)
                        except Exception as meta_exc:
                            rospy.logwarn("  Failed to update scene metadata: %s", meta_exc)

            icp_metrics = None
            if icp_result:
                icp_metrics = icp_result.get("metrics", None)
            rollout_trajectory = None
            if self.last_rollout_trajectory_path and os.path.exists(self.last_rollout_trajectory_path):
                try:
                    with open(self.last_rollout_trajectory_path, "r", encoding="utf-8") as f:
                        rollout_trajectory = json.load(f)
                except Exception as traj_exc:
                    rospy.logwarn("  Failed to load rollout trajectory: %s", traj_exc)
            trajectory = rollout_trajectory or {
                "format": "pipeline_success_pose_only",
                "frame": "base",
                "num_waypoints": 0,
                "velocities": [],
            }
            success = trajectory.get("success", True)
            estimated_height = obj.get("estimated_object_height", None)
            if estimated_height is None and obj_size and len(obj_size) == 3:
                estimated_height = max(float(v) for v in obj_size)
            object_category = (
                self.object_shape if self.object_shape != "unknown"
                else best_demo.get("object_category", "object")
            )
            grasp_pose_base = {
                "position": [float(v) for v in grasp_pos],
                "orientation_xyzw": [float(v) for v in grasp_ori],
            }
            recorded = {
                "id": demo_id,
                "format": "mt3_auto_success_v3",
                "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "mt3_pipeline_success",
                "query": self.language_query,
                "task": self.language_query,
                "object_label": str(self.object_label),
                "object_shape": str(self.object_shape),
                "object_category": str(object_category),
                "trial_note": str(self.trial_note),
                "retrieved_demo": best_demo.get("id", ""),
                "retrieved_demo_id": best_demo.get("id", ""),
                "source_demo_id": best_demo.get("id", ""),
                "retrieval_score": float(score),
                "success": bool(success),
                "estimated_object_size": [float(v) for v in obj_size],
                "estimated_object_height": (
                    None if estimated_height is None else float(estimated_height)
                ),
                "object_position_base": [float(v) for v in obj_pos],
                "grasp_pose_base": grasp_pose_base,
                "icp_mean_error_m": (
                    None if not icp_metrics else icp_metrics.get("mean_error_m")
                ),
                "icp_median_error_m": (
                    None if not icp_metrics else icp_metrics.get("median_error_m")
                ),
                "icp_p90_error_m": (
                    None if not icp_metrics else icp_metrics.get("p90_error_m")
                ),
                "language_description": self.language_query,
                "language_tags": [
                    self.language_query,
                    "grasp",
                    "pick up",
                    "green object",
                    "top-down grasp",
                    "successful rollout",
                    str(self.object_label),
                    str(self.object_shape),
                ],
                "object_info": {
                    "position_base": [float(v) for v in obj_pos],
                    "size_m": [float(v) for v in obj_size],
                    "height_m": (
                        None if estimated_height is None else float(estimated_height)
                    ),
                    "category": object_category,
                    "label": self.object_label,
                    "color": "green",
                },
                "scene_package": saved_scene_package,
                "scene_package_dir": (
                    os.path.join("demo_library", "scene_packages", saved_scene_package)
                    if saved_scene_package else None
                ),
                "icp_metrics": icp_metrics,
                "bottleneck_pose_base_frame": {
                    "position_m": {
                        "x": float(grasp_pos[0]),
                        "y": float(grasp_pos[1]),
                        "z": float(grasp_pos[2]),
                    },
                    "orientation_xyzw": {
                        "x": float(grasp_ori[0]),
                        "y": float(grasp_ori[1]),
                        "z": float(grasp_ori[2]),
                        "w": float(grasp_ori[3]),
                    },
                    "timestamp": rospy.get_time(),
                },
                "trajectory": trajectory,
                "rollout_trajectory_path": self.last_rollout_trajectory_path,
                "approach_direction": aligned.get("approach_direction", [0.0, 0.0, -1.0]),
                "retract_direction": aligned.get("retract_direction", [0.0, 0.0, 1.0]),
                "gripper_opening_m": aligned.get("gripper_opening", 0.07),
                "notes": "Auto-recorded after successful MT3 pipeline execution. "
                         "Stores successful bottleneck/grasp pose and linked RGB-D/mask/pointcloud scene package. "
                         "If available, trajectory contains sampled end-effector poses and approximate linear velocities.",
            }

            path = os.path.join(recorded_dir, demo_id + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(recorded, f, indent=2, ensure_ascii=False)
            rospy.loginfo("  Auto-recorded successful demo artifact: %s", path)
            rospy.loginfo("  Note: auto-recorded artifacts are not used for retrieval unless saved under demo_library/recorded")
            if saved_scene_package:
                rospy.loginfo("  Auto-recorded scene package: %s", saved_scene_package)
            return path
        except Exception as e:
            rospy.logwarn("  Failed to auto-record successful demo: %s", e)
            return None

    def _run_icp_registration(self, demo_package, live_package):
        if not self.run_icp or not demo_package or not live_package:
            return None
        try:
            icp_dir = os.path.join(self.scene_package_dir, "icp_%s" % self.trial_id)
            result = save_icp_outputs(
                demo_package["package_dir"],
                live_package["package_dir"],
                icp_dir)
            result["output_dir"] = icp_dir
            metrics = result.get("metrics", {})
            rospy.loginfo(
                "  ICP registration saved: %s "
                "(median=%.4fm, p90=%.4fm, iters=%d)",
                icp_dir,
                metrics.get("median_error_m", -1.0),
                metrics.get("p90_error_m", -1.0),
                metrics.get("iterations", 0))
            return result
        except Exception as e:
            rospy.logwarn("  ICP registration failed: %s", e)
            return None

    def _icp_object_pose(self, icp_result, fallback_pose):
        """Create a camera-frame object pose from the ICP demo->live transform."""
        try:
            if not icp_result:
                return None
            demo_points = np.load(os.path.join(
                icp_result["demo_package"], "pointcloud.npy")).astype(np.float64)
            demo_points = demo_points[np.all(np.isfinite(demo_points), axis=1)]
            if len(demo_points) < 5:
                return None
            demo_center = np.median(demo_points, axis=0)
            T = np.asarray(icp_result["transform_demo_to_live"], dtype=np.float64)
            live_center = (T[:3, :3] @ demo_center) + T[:3, 3]
            live_points = np.load(os.path.join(
                icp_result["live_package"], "pointcloud.npy")).astype(np.float64)
            live_size = self._estimate_object_size_from_pointcloud(live_points)
            old = np.asarray(fallback_pose.get("position", live_center.tolist()), dtype=np.float64)
            dist = float(np.linalg.norm(live_center - old))
            pose = dict(fallback_pose)
            pose["position"] = live_center.tolist()
            pose["method"] = "ICP+depth_pointcloud"
            pose["icp_vs_mask_center_m"] = dist
            if live_size is not None:
                pose["estimated_object_size"] = live_size
            rospy.loginfo(
                "  ICP object pose camera: [%.4f, %.4f, %.4f] "
                "(diff_from_mask_center=%.4fm)",
                live_center[0], live_center[1], live_center[2], dist)
            if live_size is not None:
                rospy.loginfo(
                    "  ICP/live pointcloud object size estimate: "
                    "[%.4f, %.4f, %.4f]",
                    live_size[0], live_size[1], live_size[2])
            return pose
        except Exception as e:
            rospy.logwarn("  Failed to build ICP object pose: %s", e)
            return None

    def _estimate_object_size_from_pointcloud(self, points):
        """Estimate visible object dimensions from robust point-cloud extents."""
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

    def _robust_base_z_top(self, package_dir, percentile=90.0):
        """Estimate visible top z in base frame from a scene package point cloud."""
        try:
            meta_path = os.path.join(package_dir, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            obj_pos = meta.get("object_position_base")
            obj_size = meta.get("object_size")
            if obj_pos and obj_size and len(obj_pos) >= 3 and len(obj_size) >= 3:
                return float(obj_pos[2]) + float(obj_size[2])
        except Exception:
            pass
        try:
            points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
        except Exception:
            return None
        source_frame = self._package_source_frame(package_dir)
        base_points = self._transform_points_to_base(points, source_frame)
        if base_points is None or len(base_points) < 10:
            return None
        z = base_points[:, 2]
        z = z[np.isfinite(z)]
        if len(z) < 10:
            return None
        return float(np.percentile(z, percentile))

    def _apply_icp_top_surface_grasp_z(self, aligned, demo_data, icp_result):
        """
        Map grasp height using demo/live point-cloud top surfaces.

        This is closer to the MT3 idea than center+height rules:
        demo grasp clearance above demo point-cloud top is preserved on the
        current live point cloud.
        """
        if not self._is_top_down_grasp(aligned, demo_data):
            rospy.loginfo(
                "  pointcloud top-surface z mapping skipped: non-top grasp demo")
            return False
        grasp = aligned.get("grasp_pose", {})
        grasp_pos = grasp.get("position")
        if not grasp_pos or len(grasp_pos) < 3:
            return False

        demo_entry = demo_data.get("demo_entry", {}) if demo_data else {}
        if demo_entry.get("task_type") == "pick_place":
            obj = aligned.get("object_pose_base", {}) or {}
            obj_pos = obj.get("position")
            if not obj_pos or len(obj_pos) < 3:
                return False
            size = obj.get("estimated_object_size", None)
            if not size or len(size) < 3:
                size = self.object_size
            try:
                object_height = float(size[2])
            except Exception:
                object_height = 0.045
            clearance = float(rospy.get_param(
                "~pick_place_top_grasp_clearance", 0.005))
            old_z = float(grasp_pos[2])
            mapped_z = float(obj_pos[2]) + object_height + clearance
            grasp_pos[2] = mapped_z
            if aligned.get("bottleneck_pose"):
                bn = aligned["bottleneck_pose"]
                bn_pos = bn.get("position")
                if bn_pos and len(bn_pos) >= 3:
                    bn_pos[2] += mapped_z - old_z
            aligned["pick_place_contact_z_mapping"] = {
                "object_z": float(obj_pos[2]),
                "object_height": object_height,
                "clearance": clearance,
                "old_grasp_z": old_z,
                "mapped_grasp_z": mapped_z,
                "source": "live_object_bottom_plus_height",
            }
            rospy.loginfo(
                "  pick-place contact z mapping: object_z=%.4f height=%.4f "
                "clearance=%.4f grasp_z %.4f -> %.4f",
                float(obj_pos[2]), object_height, clearance, old_z, mapped_z)
            return True

        if not icp_result:
            return False
        demo_package = icp_result.get("demo_package")
        live_package = icp_result.get("live_package")
        if not demo_package or not live_package:
            return False

        demo_top_z = self._robust_base_z_top(demo_package)
        live_top_z = self._robust_base_z_top(live_package)
        if demo_top_z is None or live_top_z is None:
            rospy.logwarn("  pointcloud top-surface z mapping skipped: top z unavailable")
            return False

        demo_grasp_z = None
        try:
            if "demo_entry" in demo_data:
                gp = demo_data["demo_entry"].get("grasp_pose_base_frame", {}).get("position_m", {})
                demo_grasp_z = float(gp["z"])
            elif "position" in demo_data:
                demo_grasp_z = float(demo_data["position"][2])
        except Exception:
            demo_grasp_z = None
        if demo_grasp_z is None:
            rospy.logwarn("  pointcloud top-surface z mapping skipped: demo grasp z unavailable")
            return False

        demo_clearance = max(0.005, demo_grasp_z - demo_top_z)
        old_z = float(grasp_pos[2])
        mapped_z = live_top_z + demo_clearance
        grasp_pos[2] = mapped_z
        if aligned.get("bottleneck_pose"):
            bn = aligned["bottleneck_pose"]
            bn_pos = bn.get("position")
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
            "  pointcloud top-surface z mapping: demo_top=%.4f live_top=%.4f "
            "demo_clearance=%.4f grasp_z %.4f -> %.4f",
            demo_top_z, live_top_z, demo_clearance, old_z, mapped_z)
        return True

    def _is_top_down_grasp(self, aligned=None, demo_data=None):
        approach = None
        if aligned:
            approach = aligned.get("approach_direction")
        if approach is None and demo_data and "demo_entry" in demo_data:
            approach = demo_data["demo_entry"].get("approach_direction")
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

    def _normalize_parallel_gripper_yaw_delta(self, yaw):
        """Normalize yaw delta where 180 degrees is equivalent for a parallel gripper."""
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        if yaw > math.pi / 2.0:
            yaw -= math.pi
        elif yaw < -math.pi / 2.0:
            yaw += math.pi
        return yaw

    def _yaw_quaternion(self, yaw):
        half = 0.5 * float(yaw)
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    def _transform_points_to_base(self, points, source_frame):
        """Transform camera-frame point cloud points into base frame using TF."""
        points = np.asarray(points, dtype=np.float64)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 10:
            return None
        if not source_frame:
            source_frame = "head_camera"
        if getattr(self.aligner, "_tf_buffer", None) is None:
            return None
        try:
            tf = self.aligner._tf_buffer.lookup_transform(
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
            for p in points:
                pos_base, _ = pose_compose(
                    tf_pos, tf_ori,
                    [float(p[0]), float(p[1]), float(p[2])],
                    identity)
                base_points.append(pos_base)
            return np.asarray(base_points, dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("  Pointcloud yaw TF transform failed (%s -> base): %s",
                          source_frame, exc)
            return None

    def _package_source_frame(self, package_dir):
        meta_path = os.path.join(package_dir, "metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pose = meta.get("pose") or {}
            return pose.get("source_frame") or pose.get("frame") or "head_camera"
        except Exception:
            return "head_camera"

    def _estimate_package_long_axis_yaw_base(self, package_dir):
        """Estimate rectangular object's long-axis yaw in base XY from package point cloud."""
        try:
            points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
        except Exception as exc:
            rospy.logwarn("  Pointcloud yaw: cannot read package %s: %s", package_dir, exc)
            return None

        source_frame = self._package_source_frame(package_dir)
        base_points = self._transform_points_to_base(points, source_frame)
        if base_points is None or len(base_points) < 10:
            return None

        xy = base_points[:, :2]
        center = np.median(xy, axis=0)
        dist = np.linalg.norm(xy - center, axis=1)
        if len(dist) >= 20:
            keep = dist <= np.percentile(dist, 92)
            xy = xy[keep]
        if len(xy) < 10:
            return None

        centered = xy - np.mean(xy, axis=0)
        cov = centered.T @ centered / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        yaw = math.atan2(float(axis[1]), float(axis[0]))
        return yaw

    def _apply_rectangular_prism_yaw_alignment(self, aligned, icp_result):
        """Rotate top-down grasp yaw according to demo/live rectangular prism point clouds."""
        if not self.use_pointcloud_yaw:
            return
        shape = str(self.object_shape or "").lower()
        enabled_shapes = [str(s).lower() for s in self.pointcloud_yaw_shapes]
        if shape not in enabled_shapes:
            return
        grasp = aligned.get("grasp_pose", {})
        if not grasp.get("orientation"):
            return
        if not icp_result:
            rospy.logwarn("  pointcloud yaw alignment skipped: no ICP result")
            return

        demo_package = icp_result.get("demo_package")
        live_package = icp_result.get("live_package")
        if not demo_package or not live_package:
            return

        demo_yaw = self._estimate_package_long_axis_yaw_base(demo_package)
        live_yaw = self._estimate_package_long_axis_yaw_base(live_package)
        if demo_yaw is None or live_yaw is None:
            rospy.logwarn("  pointcloud yaw alignment skipped: yaw estimate unavailable")
            return

        delta = self._normalize_parallel_gripper_yaw_delta(live_yaw - demo_yaw)
        if abs(delta) < math.radians(2.0):
            rospy.loginfo(
                "  pointcloud yaw alignment: demo=%.1fdeg live=%.1fdeg delta=%.1fdeg (no change)",
                math.degrees(demo_yaw), math.degrees(live_yaw), math.degrees(delta))
            return

        old_ori = [float(v) for v in grasp["orientation"]]
        new_ori = quat_multiply(self._yaw_quaternion(delta), old_ori)
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
            "  pointcloud yaw alignment: demo=%.1fdeg live=%.1fdeg delta=%.1fdeg",
            math.degrees(demo_yaw), math.degrees(live_yaw), math.degrees(delta))
        rospy.loginfo(
            "  grasp yaw-adjusted orientation: [%.3f, %.3f, %.3f, %.3f]",
            new_ori[0], new_ori[1], new_ori[2], new_ori[3])

    def _estimate_package_obb_center_base(self, package_dir):
        """Estimate XY center from a PCA-oriented bounding box in base frame."""
        try:
            points = np.load(os.path.join(package_dir, "pointcloud.npy")).astype(np.float64)
        except Exception as exc:
            rospy.logwarn("  OBB center: cannot read package %s: %s", package_dir, exc)
            return None

        source_frame = self._package_source_frame(package_dir)
        base_points = self._transform_points_to_base(points, source_frame)
        if base_points is None or len(base_points) < 20:
            return None

        xy = base_points[:, :2]
        center = np.median(xy, axis=0)
        dist = np.linalg.norm(xy - center, axis=1)
        if len(dist) >= 30:
            keep = dist <= np.percentile(dist, 92)
            xy = xy[keep]
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

    def _apply_rectangular_prism_obb_center_alignment(self, aligned, icp_result):
        """Use oriented point-cloud bounds for long-object top-grasp XY placement."""
        shape = str(self.object_shape or "").lower()
        enabled_shapes = [str(s).lower() for s in self.pointcloud_yaw_shapes]
        if shape not in enabled_shapes or not icp_result:
            return False

        live_package = icp_result.get("live_package")
        if not live_package:
            return False

        center = self._estimate_package_obb_center_base(live_package)
        if center is None:
            rospy.logwarn("  pointcloud OBB center alignment skipped: center unavailable")
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
                "  pointcloud OBB center alignment skipped: correction too large "
                "(dx=%.3f dy=%.3f)", dx, dy)
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
            "  pointcloud OBB center alignment: xy [%.4f, %.4f] -> [%.4f, %.4f] "
            "(dx=%.3f dy=%.3f extents=%.3fx%.3f)",
            old_x, old_y, center["x"], center["y"], dx, dy,
            center.get("extent_long", 0.0), center.get("extent_short", 0.0))
        return True

    def _default_test_data(self):
        """Return default test data matching the hardcoded grasp position."""
        return {
            "rgb": None, "depth": None, "segmap": None,
            "intrinsics": np.array([[550., 0., 320.], [0., 550., 240.], [0., 0., 1.]]),
            "pose": {"position": [0.6, 0.0, -0.58], "orientation": [0., 0., 0., 1.],
                     "confidence": 1.0, "method": "default"}
        }

    def _make_depth_from_pose(self, pose, img_shape):
        """Create synthetic depth map using detected cube position."""
        depth = np.zeros(img_shape[:2], dtype=np.uint16)
        if self.perception and self.perception.latest_clean_mask is not None:
            pos = pose["position"]
            depth_m = pos[2]  # Z in camera optical frame = depth
            depth[self.perception.latest_clean_mask] = max(1, int(abs(depth_m) * 1000))
        return depth

    # ================================================================
    # Step 2: Initialize live scene state
    # ================================================================
    def step2_init_scene_state(self, test_data):
        rospy.loginfo("[Step 2/7] Initializing live scene state...")
        rospy.loginfo(f"  RGB: {test_data['rgb'].shape if test_data['rgb'] is not None else 'None'}")
        rospy.loginfo(f"  Depth: {test_data['depth'].shape if test_data['depth'] is not None else 'None'}")
        rospy.loginfo(f"  Segmap: {test_data['segmap'].shape if test_data['segmap'] is not None else 'None'}")
        return test_data  # pass through

    # ================================================================
    # Step 3: Retrieve demonstration
    # ================================================================
    def step3_retrieve_demo(self, test_data):
        rospy.loginfo("[Step 3/7] Retrieving demonstration via hierarchical retrieval...")

        detected_features = {
            "shape": "box",
            "dimensions_m": [0.045, 0.045, 0.045],
            "aspect_ratio": [1.0, 1.0, 1.0],
            "color_rgb": [0.0, 1.0, 0.0],
        }
        if self.perception and self.perception.latest_detection:
            feats = self.perception.get_detected_features()
            if feats:
                detected_features.update(feats)
        if self.object_shape and self.object_shape != "unknown":
            detected_features["shape"] = self.object_shape
        if self.object_label and self.object_label != "unknown":
            detected_features["object_label"] = self.object_label

        # Language retrieval
        semantic_results = self.library.semantic_language_query(self.language_query, top_k=3)
        rospy.loginfo("  Semantic language retrieval:")
        if semantic_results:
            for demo, score, meta in semantic_results:
                method = meta.get("method", "semantic")
                reason = meta.get("reason", "")
                canonical = meta.get("canonical_task", "")
                suffix = f" [{method}]"
                if canonical:
                    suffix += f" task='{canonical}'"
                if reason:
                    suffix += f" reason='{reason}'"
                rospy.loginfo(f"    {demo['id']}: {score:.3f}{suffix}")
        else:
            lang_results = self.library.jaccard_language_query(self.language_query, top_k=3)
            rospy.loginfo("  Semantic retriever unavailable; using legacy Jaccard:")
            for demo, score in lang_results:
                rospy.loginfo(f"    {demo['id']}: {score:.3f}")

        # Geometric retrieval
        geo_results = self.library.geometric_query(detected_features, top_k=3)
        rospy.loginfo("  Geometric retrieval:")
        for demo, score in geo_results:
            rospy.loginfo(f"    {demo['id']}: {score:.3f}")

        # MT3-style retrieval: language candidate filtering, then geometry ranking.
        best_demo, combined_score = self.library.full_query(
            self.language_query, detected_features,
            lang_weight=self.lang_weight, geo_weight=self.geo_weight,
            task_type=self.task_type,
            retrieval_mode=self.retrieval_mode)
        meta = getattr(self.library, "last_retrieval_metadata", {}) or {}
        rospy.loginfo("  Retrieval mode: %s task_filter=%s",
                      meta.get("retrieval_mode", self.retrieval_mode),
                      meta.get("task_type_filter", self.task_type))
        for item in meta.get("geometric_candidates", [])[:5]:
            rospy.loginfo("    candidate %s: lang=%.3f geo=%.3f",
                          item.get("id", ""),
                          float(item.get("language_score", 0.0)),
                          float(item.get("geometry_score", 0.0)))
        rospy.loginfo(f"  Best match: {best_demo['id']} (score={combined_score:.3f})")

        return best_demo, combined_score

    # ================================================================
    # Step 4: Load retrieved demonstration
    # ================================================================
    def step4_load_demo(self, demo_entry):
        rospy.loginfo("[Step 4/7] Loading retrieved demonstration...")
        rospy.loginfo(f"  Demo: {demo_entry['id']}")
        rospy.loginfo(f"  Tags: {demo_entry.get('language_tags', [])[:3]}...")

        # Try to load official-format demo from mt3/assets/demonstrations/
        demo_data = self._try_load_official_demo(demo_entry)

        if demo_data is None:
            # Build demo data from our cube_demos.json entry
            demo_data = self._build_demo_from_library(demo_entry)

        return demo_data

    def _try_load_official_demo(self, demo_entry):
        """Try loading demo in official MT3 format."""
        demo_name = demo_entry.get("id", "")
        official_path = os.path.join(OFFICIAL_DEMO_DIR, demo_name)
        if not os.path.isdir(official_path):
            # Try common variations
            for name in ["pick_up_green_cube", "cube_green_45mm"]:
                p = os.path.join(OFFICIAL_DEMO_DIR, name)
                if os.path.isdir(p):
                    official_path = p
                    break
            else:
                return None

        try:
            rgb = cv2.cvtColor(
                cv2.imread(os.path.join(official_path, "head_camera_ws_rgb.png")),
                cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(official_path, "head_camera_ws_depth_to_rgb.png"),
                               cv2.IMREAD_UNCHANGED)
            segmap = np.load(os.path.join(official_path, "head_camera_ws_segmap.npy"))
            intrinsics = np.load(os.path.join(official_path, "head_camera_rgb_intrinsic_matrix.npy"))
            bottleneck = np.load(os.path.join(official_path, "bottleneck_pose.npy"))
            rospy.loginfo(f"  Loaded official demo from: {official_path}")
            return {
                "rgb": rgb, "depth": depth, "segmap": segmap,
                "intrinsics": intrinsics, "bottleneck_pose": bottleneck,
                "pose": {
                    "position": [float(bottleneck[0, 3]), float(bottleneck[1, 3]), float(bottleneck[2, 3])],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "method": "official_bottleneck_pose",
                    "confidence": 1.0,
                },
                "name": os.path.basename(official_path),
                "source": "official_format"
            }
        except Exception as e:
            rospy.logwarn(f"  Failed to load official demo: {e}")
            return None

    def _build_demo_from_library(self, demo_entry):
        """Build demo data from cube_demos.json entry."""
        gp = demo_entry.get("grasp_pose_base_frame", {}).get("position_m", {})
        demo_pos = [gp.get("x", 0.6), gp.get("y", 0.0), gp.get("z", -0.58)]
        # Generate synthetic demo RGB (green cube rendered via matplotlib)
        synthetic_rgb = self._render_synthetic_demo_rgb(demo_entry)
        synthetic_segmap = self._render_synthetic_demo_segmap()
        demo_depth = self._render_synthetic_demo_depth(demo_pos)
        rospy.loginfo(f"  Built demo from library (no official-format data found)")
        return {
            "rgb": synthetic_rgb, "depth": demo_depth, "segmap": synthetic_segmap,
            "intrinsics": np.array([[550., 0., 320.], [0., 550., 240.], [0., 0., 1.]]),
            "position": demo_pos,
            "pose": {
                "position": demo_pos,
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "method": "library_demo_pose",
                "confidence": 1.0,
            },
            "name": demo_entry["id"],
            "source": "library_json",
            "demo_entry": demo_entry,
        }

    def _render_synthetic_demo_rgb(self, demo_entry):
        """Render a synthetic top-down view of the target cube for retrieval viz."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        dpi = 50
        w, h = 640, 480
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        # Neutral tabletop background
        ax.set_facecolor((0.75, 0.75, 0.75))
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.axis('off')

        # Draw green cube (top-down = square)
        cx, cy, size = w//2, h//2, 80
        rect = Rectangle((cx - size//2, cy - size//2), size, size,
                         linewidth=2, edgecolor=(0.1, 0.5, 0.1),
                         facecolor=(0.2, 0.8, 0.2))
        ax.add_patch(rect)
        ax.text(cx, cy + size//2 + 20, demo_entry.get("id", "cube"),
                ha='center', fontsize=8, color='black')

        fig.canvas.draw()
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        return rgb

    def _render_synthetic_demo_segmap(self):
        """Generate binary segmap for synthetic demo (green square region)."""
        mask = np.zeros((480, 640), dtype=bool)
        cx, cy, half = 320, 240, 45
        mask[cy-half:cy+half, cx-half:cx+half] = True
        return mask

    def _render_synthetic_demo_depth(self, demo_pos):
        """Generate synthetic depth for demo at given position."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        cx, cy, half = 320, 240, 45
        depth_mm = int(abs(demo_pos[2]) * 1000) if len(demo_pos) > 2 else 800
        depth[cy-half:cy+half, cx-half:cx+half] = depth_mm
        return depth

    # ================================================================
    # Step 5: Estimate relative pose
    # ================================================================
    def step5_estimate_pose(self, test_data, demo_data, icp_result=None):
        rospy.loginfo("[Step 5/7] Estimating relative pose (PnP + Alignment)...")

        test_pose = test_data.get("pose")
        if test_pose is None:
            rospy.logwarn("  No test pose — using default")
            test_pose = {"position": [0.6, 0.0, -0.58],
                         "orientation": [0.0, 0.0, 0.0, 1.0]}
        if self.use_icp_object_pose:
            icp_pose = self._icp_object_pose(icp_result, test_pose)
            if icp_pose is not None:
                test_pose = icp_pose
                rospy.loginfo("  Using ICP object pose for grasp alignment")
            else:
                rospy.logwarn("  ICP object pose unavailable; using mask pointcloud pose")

        # Check if demo has bottleneck pose (official format)
        if "bottleneck_pose" in demo_data:
            bn_pose = demo_data["bottleneck_pose"]
            # Build a demo dict compatible with our aligner
            demo_for_align = {
                "grasp_pose_base_frame": {
                    "position_m": {"x": float(bn_pose[0, 3]),
                                   "y": float(bn_pose[1, 3]),
                                   "z": float(bn_pose[2, 3])},
                    "orientation_xyzw": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
                }
            }
        elif "demo_entry" in demo_data:
            demo_for_align = demo_data["demo_entry"]
        else:
            # Build from library entry
            demo_for_align = {
                "grasp_pose_base_frame": {
                    "position_m": {"x": demo_data.get("position", [0.6, 0.0, -0.58])[0],
                                   "y": demo_data.get("position", [0.6, 0.0, -0.58])[1],
                                   "z": demo_data.get("position", [0.6, 0.0, -0.58])[2]},
                    "orientation_xyzw": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
                }
            }

        # Use our alignment module to compute aligned grasp
        aligned = self.aligner.align(demo_for_align, test_pose)
        aligned["retrieved_demo_name"] = demo_data.get("name", "")
        aligned["retrieved_demo_source"] = demo_data.get("source", "")
        if "demo_entry" in demo_data:
            aligned["retrieved_demo_entry"] = demo_data["demo_entry"]

        # For MT3-style interaction replay, the alignment target is the
        # demonstration bottleneck pose, while the grasp pose is the contact
        # target.  Our earlier execution used only the contact target.  Keep
        # both so the Sawyer script can first reach bottleneck, then replay.
        try:
            if "demo_entry" in demo_data:
                entry = demo_data["demo_entry"]
                obj_frame = entry.get("object_pose_base_frame", {}).get("position_m", {})
                bn_frame = entry.get("bottleneck_pose_base_frame", {})
                bn_pos_m = bn_frame.get("position_m", {})
                bn_ori_m = bn_frame.get("orientation_xyzw", {})
                obj_current = aligned.get("object_pose_base", {}).get("position", None)
                if obj_current and bn_pos_m and obj_frame:
                    demo_obj_pos = [obj_frame["x"], obj_frame["y"], obj_frame["z"]]
                    demo_bn_pos = [bn_pos_m["x"], bn_pos_m["y"], bn_pos_m["z"]]
                    bn_delta = [
                        demo_bn_pos[0] - demo_obj_pos[0],
                        demo_bn_pos[1] - demo_obj_pos[1],
                        demo_bn_pos[2] - demo_obj_pos[2],
                    ]
                    aligned["bottleneck_pose"] = {
                        "position": [
                            obj_current[0] + bn_delta[0],
                            obj_current[1] + bn_delta[1],
                            obj_current[2] + bn_delta[2],
                        ],
                        "orientation": [
                            bn_ori_m.get("x", aligned["grasp_pose"]["orientation"][0]),
                            bn_ori_m.get("y", aligned["grasp_pose"]["orientation"][1]),
                            bn_ori_m.get("z", aligned["grasp_pose"]["orientation"][2]),
                            bn_ori_m.get("w", aligned["grasp_pose"]["orientation"][3]),
                        ],
                        "relative_to_object": bn_delta,
                    }
        except Exception as exc:
            rospy.logwarn("  Failed to build aligned bottleneck pose: %s", exc)

        self._apply_rectangular_prism_yaw_alignment(aligned, icp_result)
        self._apply_rectangular_prism_obb_center_alignment(aligned, icp_result)

        pos = aligned["grasp_pose"]["position"]
        ori = aligned["grasp_pose"]["orientation"]
        rospy.loginfo(f"  TF source: {aligned.get('tf_source', 'unknown')}")
        rospy.loginfo(f"  Object in base: [{aligned['object_pose_base']['position'][0]:.3f}, "
                      f"{aligned['object_pose_base']['position'][1]:.3f}, "
                      f"{aligned['object_pose_base']['position'][2]:.3f}]")
        if not self._apply_icp_top_surface_grasp_z(aligned, demo_data, icp_result):
            self._apply_height_aware_top_grasp(aligned)
        self._apply_side_grasp_midheight(aligned)
        self._apply_side_grasp_bottleneck_near_object(aligned)
        pos = aligned["grasp_pose"]["position"]
        rospy.loginfo(f"  Aligned grasp:  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        return aligned

    def _apply_height_aware_top_grasp(self, aligned):
        """Raise top-down grasp for taller objects using estimated point-cloud height."""
        if not self.use_height_aware_top_grasp:
            return
        if not self._is_top_down_grasp(aligned):
            rospy.loginfo("  height-aware top grasp skipped: non-top grasp demo")
            return
        obj = aligned.get("object_pose_base", {})
        grasp = aligned.get("grasp_pose", {})
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
        clearance = self.top_grasp_clearance
        desired_offset = max(current_offset, height * 0.5 + clearance)
        if desired_offset > current_offset + 1e-4:
            old_z = grasp_pos[2]
            grasp_pos[2] = float(obj_pos[2]) + desired_offset
            rospy.loginfo(
                "  height-aware top grasp z: %.4f -> %.4f "
                "(height=%.4f, clearance=%.4f, offset=%.4f)",
                old_z, grasp_pos[2], height, clearance, desired_offset)

    def _apply_side_grasp_midheight(self, aligned):
        """For side grasps, clamp the contact height around the visible object mid-height."""
        if self._is_top_down_grasp(aligned):
            return

        obj = aligned.get("object_pose_base", {})
        grasp = aligned.get("grasp_pose", {})
        obj_pos = obj.get("position")
        grasp_pos = grasp.get("position")
        if not obj_pos or not grasp_pos or len(obj_pos) < 3 or len(grasp_pos) < 3:
            return

        height = obj.get("estimated_object_height", None)
        if height is None:
            size = obj.get("estimated_object_size", None)
            if size and len(size) == 3:
                height = max(float(v) for v in size)
        try:
            height = float(height)
        except Exception:
            return
        if height <= 0.02 or height > 0.25:
            return

        old_z = float(grasp_pos[2])
        obj_z = float(obj_pos[2])
        if obj_z > self.object_z_max and height > 0.08:
            # For tall side-grasp objects, the Sawyer right_hand frame is
            # above the actual two-finger contact center when the gripper is
            # horizontal.  Use the visible point-cloud center plus a fraction
            # of the object height as the flange target; the fingertips then
            # land closer to the cylinder mid-height instead of chasing an
            # overly low right_hand z target.
            flange_ratio = float(rospy.get_param(
                "~side_grasp_flange_height_ratio", 0.25))
            flange_ratio = max(0.0, min(0.45, flange_ratio))
            desired_z = obj_z + flange_ratio * height
            z_source = "visible_pointcloud_center_plus_flange_ratio"
        else:
            desired_z = obj_z + 0.5 * height
            z_source = "bottom_plus_half_height"
        if abs(old_z - desired_z) <= 0.005:
            return

        grasp_pos[2] = desired_z
        aligned["side_grasp_midheight_correction"] = {
            "object_z": obj_z,
            "estimated_height": height,
            "old_grasp_z": old_z,
            "corrected_grasp_z": desired_z,
            "source": z_source,
            "flange_height_ratio": (
                flange_ratio
                if z_source == "visible_pointcloud_center_plus_flange_ratio"
                else None
            ),
        }
        rospy.loginfo(
            "  side-grasp mid-height z: %.4f -> %.4f "
            "(object_z=%.4f, height=%.4f, source=%s)",
            old_z, desired_z, obj_z, height, z_source)

    def _apply_side_grasp_bottleneck_near_object(self, aligned):
        """For side grasps, align y/z first and let replay approach only along x."""
        if self._is_top_down_grasp(aligned):
            return

        obj = aligned.get("object_pose_base", {})
        grasp = aligned.get("grasp_pose", {})
        obj_pos = obj.get("position")
        grasp_pos = grasp.get("position")
        if not obj_pos or not grasp_pos or len(obj_pos) < 3 or len(grasp_pos) < 3:
            return

        size = obj.get("estimated_object_size", None)
        radius = 0.025
        if size and len(size) >= 2:
            try:
                radius = max(0.018, min(0.06, 0.5 * max(float(size[0]), float(size[1]))))
            except Exception:
                radius = 0.025

        current_x_width = 0.045
        if size and len(size) >= 1:
            try:
                current_x_width = max(0.005, abs(float(size[0])))
            except Exception:
                current_x_width = 0.045

        def _pos_x(frame_key):
            frame = aligned.get("retrieved_demo_entry", {}).get(frame_key, {})
            pos = frame.get("position_m", {})
            return float(pos["x"])

        def _demo_size_x():
            entry = aligned.get("retrieved_demo_entry", {}) or {}
            candidates = []
            for key in ("object_size_m", "object_size", "size_m"):
                candidates.append(entry.get(key))
            for parent_key in ("object_info", "object", "metadata"):
                parent = entry.get(parent_key, {}) or {}
                for key in ("size_m", "object_size_m", "object_size"):
                    candidates.append(parent.get(key))
            for value in candidates:
                try:
                    if value and len(value) >= 1:
                        return max(0.005, abs(float(value[0])))
                except Exception:
                    pass
            return max(0.005, float(self.object_size[0]))

        raw_final_dx = float(grasp_pos[0]) - float(obj_pos[0])
        mapped_by_demo_scale = False
        try:
            demo_obj_x = _pos_x("object_pose_base_frame")
            demo_grasp_x = _pos_x("grasp_pose_base_frame")
            demo_x_width = _demo_size_x()
            demo_ratio = (demo_grasp_x - demo_obj_x) / demo_x_width
            max_ratio = abs(float(rospy.get_param("~side_grasp_max_width_ratio", 0.55)))
            if max_ratio <= 0.0:
                max_ratio = 0.55
            safe_ratio = max(-max_ratio, min(max_ratio, demo_ratio))
            old_x = float(grasp_pos[0])
            grasp_pos[0] = float(obj_pos[0]) + safe_ratio * current_x_width
            mapped_by_demo_scale = True
            rospy.loginfo(
                "  side-grasp demo-scale x mapping: %.3f -> %.3f "
                "(demo_dx=%.3f, demo_width=%.3f, ratio=%.3f, "
                "safe_ratio=%.3f, current_width=%.3f)",
                old_x, grasp_pos[0],
                demo_grasp_x - demo_obj_x, demo_x_width,
                demo_ratio, safe_ratio, current_x_width)
        except Exception as exc:
            rospy.logwarn(
                "  side-grasp demo-scale x mapping unavailable (%s); "
                "using current object width fallback.",
                exc)

        if not mapped_by_demo_scale:
            max_final_x_offset = max(0.016, min(0.024, current_x_width * 0.55))
            if abs(raw_final_dx) > max_final_x_offset:
                sign = 1.0 if raw_final_dx >= 0.0 else -1.0
                old_x = float(grasp_pos[0])
                grasp_pos[0] = float(obj_pos[0]) + sign * max_final_x_offset
                rospy.loginfo(
                    "  side-grasp final x limited by object width fallback: "
                    "%.3f -> %.3f (object_x=%.3f, raw_dx=%.3f, max_dx=%.3f)",
                    old_x, grasp_pos[0], float(obj_pos[0]),
                    raw_final_dx, max_final_x_offset)

        # For side grasps, the demo should decide the approach direction.
        # Earlier versions forced every side-grasp bottleneck into an x-only
        # approach, which breaks demos recorded from a different side.  Keep
        # the final grasp centered laterally, then rebuild the bottleneck from
        # the demo bottleneck->grasp offset in XY.
        old_y = float(grasp_pos[1])
        grasp_pos[1] = float(obj_pos[1])
        lateral_offset = float(rospy.get_param(
            "~side_grasp_lateral_center_offset", 0.0))
        if abs(lateral_offset) > 1e-6:
            old_centered_y = float(grasp_pos[1])
            grasp_pos[1] += lateral_offset
            rospy.loginfo(
                "  side-grasp lateral gripper-center offset applied: "
                "%.3f -> %.3f (offset=%.3f)",
                old_centered_y, grasp_pos[1], lateral_offset)
        rospy.loginfo(
            "  side-grasp lateral y centered before approach: %.3f -> %.3f "
            "(object_y=%.3f)",
            old_y, grasp_pos[1], float(obj_pos[1]))

        old_bn = aligned.get("bottleneck_pose", {})
        old_pos = old_bn.get("position", None)
        demo_delta_xy = None
        try:
            demo_bn_x = _pos_x("bottleneck_pose_base_frame")
            demo_bn_y = float(
                aligned.get("retrieved_demo_entry", {})
                .get("bottleneck_pose_base_frame", {})
                .get("position_m", {})["y"])
            demo_grasp_x = _pos_x("grasp_pose_base_frame")
            demo_grasp_y = float(
                aligned.get("retrieved_demo_entry", {})
                .get("grasp_pose_base_frame", {})
                .get("position_m", {})["y"])
            demo_delta_xy = [demo_bn_x - demo_grasp_x,
                             demo_bn_y - demo_grasp_y]
            retreat_source = "demo_bottleneck_to_grasp_xy"
        except Exception:
            demo_delta_xy = None
            retreat_source = "mapped_bottleneck_fallback"

        if demo_delta_xy is None and old_pos and len(old_pos) >= 2:
            demo_delta_xy = [
                float(old_pos[0]) - float(grasp_pos[0]),
                float(old_pos[1]) - float(grasp_pos[1]),
            ]

        if demo_delta_xy is None:
            demo_delta_xy = [-(radius + 0.06), 0.0]
            retreat_source = "geometry_fallback"

        retreat = math.sqrt(demo_delta_xy[0] ** 2 + demo_delta_xy[1] ** 2)
        # Keep the demo-derived bottleneck height.  The side-grasp bottleneck
        # is the start of the interaction trajectory, not the final grasp
        # contact height.  Forcing it to grasp_z makes the robot try to enter
        # the side grasp from an unnaturally low pose and breaks replay even
        # when the live object is at the same pose as the demo.
        if old_pos and len(old_pos) >= 3:
            bottleneck_z = float(old_pos[2])
        else:
            bottleneck_z = float(grasp_pos[2])
        new_pos = [
            float(grasp_pos[0]) + float(demo_delta_xy[0]),
            float(grasp_pos[1]) + float(demo_delta_xy[1]),
            bottleneck_z,
        ]
        old_ori = old_bn.get("orientation", None)
        if old_ori and len(old_ori) == 4:
            bottleneck_ori = [float(v) for v in old_ori]
        else:
            bottleneck_ori = list(grasp.get("orientation", [0.0, 0.0, 0.0, 1.0]))
        aligned["bottleneck_pose"] = {
            "position": new_pos,
            "orientation": bottleneck_ori,
            "relative_to_object": [
                new_pos[0] - float(obj_pos[0]),
                new_pos[1] - float(obj_pos[1]),
                new_pos[2] - float(obj_pos[2]),
            ],
            "source": "side_grasp_live_object_retreat",
        }
        if old_pos and len(old_pos) >= 3:
            rospy.loginfo(
                "  side-grasp bottleneck remapped from demo approach vector: "
                "[%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f] "
                "(x_retreat=%.3f source=%s)",
                old_pos[0], old_pos[1], old_pos[2],
                new_pos[0], new_pos[1], new_pos[2], retreat, retreat_source)
        else:
            rospy.loginfo(
                "  side-grasp bottleneck set from demo approach vector: [%.3f, %.3f, %.3f] "
                "(x_retreat=%.3f source=%s)",
                new_pos[0], new_pos[1], new_pos[2], retreat, retreat_source)

    # ================================================================
    # Step 6: Transform bottleneck pose
    # ================================================================
    def step6_transform_bottleneck(self, aligned):
        rospy.loginfo("[Step 6/7] Transforming bottleneck pose to live scene...")
        grasp = aligned["grasp_pose"]
        bottleneck = aligned.get("bottleneck_pose", grasp)
        rospy.loginfo(f"  Target bottleneck pose (T_WE):")
        rospy.loginfo(f"    Position:    [{bottleneck['position'][0]:.3f}, "
                      f"{bottleneck['position'][1]:.3f}, {bottleneck['position'][2]:.3f}]")
        rospy.loginfo(f"    Orientation: [{bottleneck['orientation'][0]:.3f}, "
                      f"{bottleneck['orientation'][1]:.3f}, "
                      f"{bottleneck['orientation'][2]:.3f}, {bottleneck['orientation'][3]:.3f}]")
        return aligned

    # ================================================================
    # Step 7: Load & execute interaction
    # ================================================================
    def step7_execute(self, aligned):
        rospy.loginfo("[Step 7/7] Executing grasp...")
        self.last_execution_failure_stage = ""
        self.last_execution_failure_reason = ""
        self.last_planning_success = ""
        self.last_executor_success = ""

        grasp_pose = aligned["grasp_pose"]
        pos = grasp_pose["position"]
        ori = grasp_pose["orientation"]
        bottleneck_pose = aligned.get("bottleneck_pose", grasp_pose)
        bn_pos = bottleneck_pose.get("position", pos)
        bn_ori = bottleneck_pose.get("orientation", ori)

        # --- 传递 MT3 算出的完整抓取位姿（位置+朝向） ---
        rospy.set_param('/sawyer_auto_grasp/grasp_x', pos[0])
        rospy.set_param('/sawyer_auto_grasp/grasp_y', pos[1])
        rospy.set_param('/sawyer_auto_grasp/grasp_z', pos[2])
        rospy.set_param('/sawyer_auto_grasp/grasp_qx', ori[0])
        rospy.set_param('/sawyer_auto_grasp/grasp_qy', ori[1])
        rospy.set_param('/sawyer_auto_grasp/grasp_qz', ori[2])
        rospy.set_param('/sawyer_auto_grasp/grasp_qw', ori[3])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_x', bn_pos[0])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_y', bn_pos[1])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_z', bn_pos[2])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_qx', bn_ori[0])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_qy', bn_ori[1])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_qz', bn_ori[2])
        rospy.set_param('/sawyer_auto_grasp/bottleneck_qw', bn_ori[3])
        rospy.set_param(
            '/sawyer_auto_grasp/grasp_mode',
            'top' if self._is_top_down_grasp(aligned) else 'side')

        # 保留兼容旧参数名
        rospy.set_param('/sawyer_auto_grasp/object_x', pos[0])
        rospy.set_param('/sawyer_auto_grasp/object_y', pos[1])
        rospy.set_param('/sawyer_auto_grasp/object_z', pos[2])

        # 传递物体在 base 帧的位置（用于计算安全过渡点）
        obj_base = aligned.get("object_pose_base", {})
        obj_pos = obj_base.get("position", pos)
        rospy.set_param('/sawyer_auto_grasp/obj_base_x', obj_pos[0])
        rospy.set_param('/sawyer_auto_grasp/obj_base_y', obj_pos[1])
        rospy.set_param('/sawyer_auto_grasp/obj_base_z', obj_pos[2])
        rospy.set_param('/sawyer_auto_grasp/object_base_x', obj_pos[0])
        rospy.set_param('/sawyer_auto_grasp/object_base_y', obj_pos[1])
        rospy.set_param('/sawyer_auto_grasp/object_base_z', obj_pos[2])
        rospy.set_param('/sawyer_auto_grasp/object_shape', str(self.object_shape))
        rospy.set_param('/sawyer_auto_grasp/object_label', str(self.object_label))
        rospy.set_param('/sawyer_auto_grasp/experiment_group',
                        str(self.experiment_group))

        # 传递物体尺寸和夹爪开度
        estimated_size = obj_base.get("estimated_object_size", None)
        if estimated_size is not None and len(estimated_size) == 3:
            execution_object_size = [float(v) for v in estimated_size]
        else:
            execution_object_size = self.object_size
        rospy.set_param('/sawyer_auto_grasp/object_size', execution_object_size)
        rospy.set_param('/sawyer_auto_grasp/gripper_opening',
                        aligned.get("gripper_opening", 0.07))
        os.makedirs(self.rollout_trajectory_dir, exist_ok=True)
        self.last_rollout_trajectory_path = os.path.join(
            self.rollout_trajectory_dir,
            "rollout_%s.json" % self.trial_id)
        rospy.set_param(
            '/sawyer_auto_grasp/trajectory_record_path',
            self.last_rollout_trajectory_path)
        rospy.set_param('/sawyer_auto_grasp/trajectory_record_rate_hz', 10.0)
        rospy.set_param('/sawyer_auto_grasp/use_demo_replay', bool(self.use_demo_replay))
        rospy.set_param('/sawyer_auto_grasp/use_top_grasp_replay',
                        bool(self.use_top_grasp_replay))
        rospy.set_param('/sawyer_auto_grasp/use_side_staged_replay',
                        bool(self.use_side_staged_replay))
        rospy.set_param('/sawyer_auto_grasp/prefer_pose_replay',
                        bool(self.prefer_pose_replay))
        rospy.set_param('/sawyer_auto_grasp/use_segmented_replay',
                        bool(self.use_segmented_replay))
        rospy.set_param('/sawyer_auto_grasp/close_on_replay_blocked',
                        bool(self.close_on_replay_blocked))
        rospy.set_param('/sawyer_auto_grasp/replay_close_on_blocked_min_progress',
                        float(self.replay_close_on_blocked_min_progress))

        self.last_demo_replay_path = ""
        if self.use_demo_replay:
            demo_entry = aligned.get("retrieved_demo_entry", {})
            demo_traj = demo_entry.get("_recorded_trajectory")
            if demo_traj:
                self.last_demo_replay_path = os.path.join(
                    self.rollout_trajectory_dir,
                    "replay_input_%s.json" % self.trial_id)
                replay_payload = {
                    "format": "mt3_demo_replay_input_v1",
                    "source_demo": demo_entry.get("id", aligned.get("retrieved_demo_name", "")),
                    "trajectory": demo_traj,
                    "replay_yaw_delta_rad": float(
                        (aligned.get("pointcloud_yaw_alignment", {}) or {}).get(
                            "delta_yaw_rad", 0.0)),
                    "replay_yaw_delta_deg": float(
                        (aligned.get("pointcloud_yaw_alignment", {}) or {}).get(
                            "delta_yaw_deg", 0.0)),
                    "aligned_bottleneck_pose": {
                        "position": [float(v) for v in bn_pos],
                        "orientation": [float(v) for v in bn_ori],
                    },
                    "aligned_grasp_pose": {
                        "position": [float(v) for v in pos],
                        "orientation": [float(v) for v in ori],
                    },
                }
                with open(self.last_demo_replay_path, "w", encoding="utf-8") as f:
                    json.dump(replay_payload, f, indent=2, ensure_ascii=False)
                rospy.loginfo("  Demo replay input: %s", self.last_demo_replay_path)
            else:
                rospy.logwarn(
                    "  Demo replay requested, but retrieved demo has no recorded trajectory; "
                    "falling back to scripted grasp.")
                rospy.set_param('/sawyer_auto_grasp/use_demo_replay', False)
        rospy.set_param('/sawyer_auto_grasp/demo_replay_trajectory_path',
                        self.last_demo_replay_path)

        rospy.loginfo(f"  MT3 grasp pose: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                      f"ori=[{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
        rospy.loginfo(f"  MT3 bottleneck pose: pos=[{bn_pos[0]:.3f}, {bn_pos[1]:.3f}, {bn_pos[2]:.3f}] "
                      f"ori=[{bn_ori[0]:.3f}, {bn_ori[1]:.3f}, {bn_ori[2]:.3f}, {bn_ori[3]:.3f}]")
        rospy.loginfo(f"  Object in base:  [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        rospy.loginfo(f"  Rollout trajectory path: {self.last_rollout_trajectory_path}")

        # ── 放置参数 (pick_place 任务) ──────────────────────────
        if self.task_type == "pick_place":
            self._write_place_params(obj_pos, execution_object_size)

        if self.dry_run:
            rospy.loginfo("  DRY RUN — skipping execution.")
            script_hint = ("mt3_sawyer_place.py" if self.task_type == "pick_place"
                          else "mt3_sawyer_grasp.py")
            rospy.loginfo(f"  Run manually: rosrun sawyer_gazebo {script_hint}")
            return True

        try:
            import subprocess
            ros_ws_src = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "ros_ws", "src", "sawyer_gazebo", "scripts")

            if self.task_type == "pick_place":
                exec_script = os.path.join(ros_ws_src, "mt3_sawyer_place.py")
                script_label = "Place"
            else:
                exec_script = os.path.join(ros_ws_src, "mt3_sawyer_grasp.py")
                script_label = "Grasp"

            proc = subprocess.Popen(
                ["python", exec_script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            error_seen = False
            detected_failure = None
            error_patterns = [
                (
                    "replay transition planning failed",
                    "motion_planning",
                    "replay_transition_planning_failed",
                ),
                (
                    "no motion plan found",
                    "motion_planning",
                    "moveit_no_motion_plan",
                ),
                (
                    "aborted: control_failed",
                    "trajectory_execution",
                    "controller_control_failed",
                ),
                (
                    "path_tolerance_violated",
                    "trajectory_execution",
                    "path_tolerance_violated",
                ),
                (
                    "execution completed: aborted",
                    "trajectory_execution",
                    "trajectory_execution_aborted",
                ),
                (
                    "controller handle /robot/limb/right reports status aborted",
                    "trajectory_execution",
                    "right_arm_controller_aborted",
                ),
            ]
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    rospy.loginfo(f"  [{script_label}] {line}")
                    low = line.lower()
                    for pattern, stage, reason in error_patterns:
                        if pattern in low:
                            error_seen = True
                            if detected_failure is None:
                                detected_failure = (stage, reason)
            proc.wait()
            if error_seen:
                recovery_used = False
                try:
                    recovery_used = bool(rospy.get_param(
                        '/sawyer_auto_grasp/used_recovery_logic', False))
                except Exception:
                    recovery_used = False
                rollout_success = None
                if self.last_rollout_trajectory_path and os.path.exists(
                        self.last_rollout_trajectory_path):
                    try:
                        with open(self.last_rollout_trajectory_path, "r",
                                  encoding="utf-8") as f:
                            rollout = json.load(f)
                        rollout_success = rollout.get("success")
                    except Exception as rollout_exc:
                        rospy.logwarn("  Could not read rollout success flag: %s",
                                      rollout_exc)
                if recovery_used and rollout_success is True:
                    rospy.logwarn(
                        "  %s output contained a transient failure, but recovery "
                        "logic completed and rollout success=True; continuing to "
                        "post-execution validation.",
                        script_label)
                else:
                    stage, reason = detected_failure or (
                        "grasp_execution", "execution_output_reported_failure")
                    self.last_execution_failure_stage = stage
                    self.last_execution_failure_reason = reason
                    self.last_planning_success = (
                        False if stage == "motion_planning" else True)
                    rospy.logerr("  %s script output contained execution/planning failure.", script_label)
                    return False
            if proc.returncode != 0:
                self.last_execution_failure_stage = "execution_process"
                self.last_execution_failure_reason = (
                    "execution_script_exit_code_%d" % proc.returncode)
                return False
            if self.last_rollout_trajectory_path and os.path.exists(self.last_rollout_trajectory_path):
                try:
                    with open(self.last_rollout_trajectory_path, "r", encoding="utf-8") as f:
                        rollout = json.load(f)
                    if rollout.get("success") is False:
                        self.last_execution_failure_stage = "grasp_execution"
                        self.last_execution_failure_reason = (
                            "rollout_reported_success_false")
                        rospy.logerr("  %s script reported success=False in rollout trajectory.", script_label)
                        return False
                except Exception as rollout_exc:
                    rospy.logwarn("  Could not read rollout success flag: %s", rollout_exc)
            self.last_planning_success = True
            return True
        except FileNotFoundError:
            rospy.logwarn("  Execution script not found. Params set — run manually.")
            return True
        except Exception as e:
            self.last_execution_failure_stage = "execution_exception"
            self.last_execution_failure_reason = type(e).__name__
            rospy.logerr(f"  Execution error: {e}")
            return False

    # ── 放置参数写入 ────────────────────────────────────────────

    def _write_place_params(self, obj_pos, object_size):
        """根据语言指令解析放置方向, 计算放置坐标并写入 ROS params."""
        # Use LLM direction parsing when available, but keep deterministic
        # keyword parsing as the baseline. A single pick-place demo supplies
        # the grasp/release skill; the target direction is recomputed here.
        direction_info = parse_place_direction(self.language_query)
        retriever = getattr(self.library, "semantic_retriever", None)
        if retriever:
            try:
                direction_info = retriever.resolve_place_direction(self.language_query)
            except Exception as exc:
                rospy.logwarn(
                    "LLM direction resolution failed: %s; using keyword fallback",
                    exc)

        surface_z_offset = float(rospy.get_param(
            "~place_surface_z_offset",
            rospy.get_param("/sawyer_auto_grasp/place_surface_z_offset", 0.0)))
        target = compute_place_target(
            obj_pos, object_size, direction_info,
            surface_z_offset=surface_z_offset)
        place_x, place_y, place_z = target["place_xyz"]
        offset_xy = target["offset_xy"]
        mode = target["mode"]
        direction = target["direction"]
        place_label = target["label"]

        rospy.set_param('/sawyer_auto_grasp/place_mode', place_label)
        rospy.set_param('/sawyer_auto_grasp/place_x', place_x)
        rospy.set_param('/sawyer_auto_grasp/place_y', place_y)
        rospy.set_param('/sawyer_auto_grasp/place_z', place_z)
        rospy.set_param('/sawyer_auto_grasp/place_direction', place_label)
        rospy.set_param('/sawyer_auto_grasp/place_offset_xy',
                        [float(v) for v in offset_xy])
        rospy.set_param('/sawyer_auto_grasp/place_resolution_method',
                        target["resolution_method"])
        rospy.set_param('/sawyer_auto_grasp/place_resolution_confidence',
                        float(target["confidence"]))
        rospy.set_param('/sawyer_auto_grasp/place_surface_z_offset',
                        float(surface_z_offset))

        rospy.loginfo("  Place params (%s, confidence=%.2f):",
                      target["resolution_method"], target["confidence"])
        rospy.loginfo("    mode: %s  direction: %s  offset_xy: %s",
                      mode, direction or "custom", offset_xy)
        rospy.loginfo("    place_xyz: [%.3f, %.3f, %.3f]",
                      place_x, place_y, place_z)
        rospy.loginfo("    reason: %s", target["reason"])

    # ================================================================
    # Main run loop
    # ================================================================
    def run(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 Pipeline: Starting execution")
        rospy.loginfo("=" * 60)
        self._timing["run_start"] = time.time()

        # Step 1: Load test image
        t0 = time.time()
        test_data = self.step1_load_test_image()
        self._timing["perception_time_s"] = time.time() - t0
        live_package = self._export_scene_package(
            test_data,
            name="live_latest",
            role="live_scene",
            extra_metadata={"language_query": self.language_query})
        live_package = self._archive_live_scene_package(live_package)

        # Step 2: Initialize scene state
        scene_state = self.step2_init_scene_state(test_data)

        # Step 3: Retrieve demonstration
        t_retrieval = time.time()
        best_demo, score = self.step3_retrieve_demo(test_data)
        self._timing["retrieval_time_s"] = time.time() - t_retrieval

        if score < 0.3:
            rospy.logwarn(f"  Low retrieval score ({score:.2f}) — continuing anyway")

        # Step 4/5 timing: demo loading + ICP/geometric alignment.
        t_alignment = time.time()
        demo_data = self.step4_load_demo(best_demo)
        demo_package = self._stored_demo_scene_package(best_demo)
        if demo_package is None:
            demo_package = self._export_scene_package(
                demo_data,
                name=f"demo_{demo_data.get('name', best_demo.get('id', 'unknown'))}",
                role="retrieved_demo",
                extra_metadata={
                    "demo_id": best_demo.get("id", ""),
                    "demo_source": demo_data.get("source", ""),
                    "retrieval_score": float(score),
                })
        icp_demo_package = self._select_demo_package_for_icp(demo_package, best_demo)
        icp_result = self._run_icp_registration(icp_demo_package, live_package)

        # Step 5: Estimate relative pose
        aligned = self.step5_estimate_pose(test_data, demo_data, icp_result=icp_result)
        self._timing["alignment_time_s"] = time.time() - t_alignment

        # --- Generate official-style visualizations ---
        rospy.loginfo("[Visualization] Generating official MT3-style outputs...")
        generate_all_official(test_data, demo_data, aligned, frame_no=self.frame_no)
        rospy.loginfo("  Saved to ~/.mt3_debug/")

        if not self._validate_perception_for_execution(test_data, aligned):
            rospy.logerr("MT3 Pipeline: ABORTED by perception safety gate.")
            self.last_executor_success = False
            self._record_experiment_result(
                "failed",
                test_data=test_data,
                aligned=aligned,
                best_demo=best_demo,
                score=score,
                live_package=live_package,
                icp_result=icp_result,
                reason="perception_safety_gate")
            return False

        # Step 6: Transform bottleneck pose
        aligned = self.step6_transform_bottleneck(aligned)

        # Step 7: Execute interaction
        self._reset_executor_timing_params()
        self.pre_execution_gazebo_pose = self._get_gazebo_object_pose()
        self.last_postcheck_info = {}
        t_exec = time.time()
        success = self.step7_execute(aligned)
        self.last_executor_success = bool(success)
        self._timing["execution_time_s"] = time.time() - t_exec
        self._timing.update(self._read_executor_timing_params())
        if success:
            if self.task_type == "pick_place":
                if not self._validate_post_place_success():
                    success = False
            elif not self._validate_post_grasp_success(aligned):
                success = False
        self._record_experiment_result(
            "success" if success else "failed",
            test_data=test_data,
            aligned=aligned,
            best_demo=best_demo,
            score=score,
            live_package=live_package,
            icp_result=icp_result,
            reason=(
                self.last_execution_failure_reason or "grasp_execution"
                if not success else ""))
        if success:
            self._record_success_demo(
                aligned, best_demo, score,
                live_package=live_package,
                icp_result=icp_result)

        # --- Verification ---
        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("VERIFICATION: MT3 computed vs hardcoded defaults")
        rospy.loginfo("=" * 60)
        obj_x = rospy.get_param('/sawyer_auto_grasp/object_x', 0.60)
        obj_y = rospy.get_param('/sawyer_auto_grasp/object_y', 0.00)
        obj_z = rospy.get_param('/sawyer_auto_grasp/object_z', -0.58)
        hardcoded = [0.60, 0.00, -0.58]
        mt3_vals = [obj_x, obj_y, obj_z]
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(mt3_vals, hardcoded)))
        rospy.loginfo(f"  Hardcoded: [{hardcoded[0]:.3f}, {hardcoded[1]:.3f}, {hardcoded[2]:.3f}]")
        rospy.loginfo(f"  MT3:       [{mt3_vals[0]:.3f}, {mt3_vals[1]:.3f}, {mt3_vals[2]:.3f}]")
        rospy.loginfo(f"  Difference: {dist:.3f}m")
        if dist < 0.01:
            rospy.logwarn("  >>> Nearly identical to hardcoded (possible fallback)")
        else:
            rospy.loginfo("  >>> MT3 is computing position DYNAMICALLY")
        rospy.loginfo("=" * 60)

        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        if success:
            rospy.loginfo("MT3 Pipeline: COMPLETED SUCCESSFULLY")
        else:
            rospy.logerr("MT3 Pipeline: COMPLETED WITH ERRORS")
        rospy.loginfo("=" * 60)

        return success


if __name__ == "__main__":
    try:
        pipeline = MT3Pipeline()
        sys.exit(0 if pipeline.run() else 1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Pipeline interrupted")
        sys.exit(130)
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
