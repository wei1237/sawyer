#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-Sawyer wrapper for the current MT3 generalization pipeline.

The simulation mt3_generalize.py is left untouched.  This module reuses its
retrieval, alignment, bottleneck mapping, replay-building and experiment logic,
but replaces the camera/alignment path with ASC60C registered RGB-D + calibrated
TF and removes Gazebo-only execution assumptions.

Safety defaults:
  * strict real execution_environment check
  * perception is required; no hardcoded/default object-pose fallback
  * real RGB/depth/CameraInfo are saved into the live scene package
  * autonomous execution defaults to DRY RUN until allow_real_execution=true
  * real executor files are separate *_real.py entry points
  * Gazebo post-checks are disabled; executor-only/manual post-check is explicit
"""

import copy
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import rospy

import mt3_generalize as _sim
from mt3_alignment_real import TrajectoryAligner as RealTrajectoryAligner
from mt3_perception_real import PerceptionNode as RealPerceptionNode

# The parent constructor resolves these names from the mt3_generalize module.
# Patch only this process; the original source file is not modified.
_sim.PerceptionNode = RealPerceptionNode
_sim.TrajectoryAligner = RealTrajectoryAligner


def _global(name):
    return "/sawyer_auto_grasp/%s" % str(name).lstrip("~/")


def _param(name, default=None):
    private = "~%s" % str(name).lstrip("~/")
    if rospy.has_param(private):
        return rospy.get_param(private)
    return rospy.get_param(_global(name), default)


def _bool(name, default=False):
    value = _param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _optional_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "success", "pass"):
        return True
    if text in ("0", "false", "no", "off", "failed", "fail"):
        return False
    return None


class MT3RealPipeline(_sim.MT3Pipeline):
    """Real-only MT3 pipeline while retaining current mt3_generalize logic."""

    def __init__(self):
        super().__init__()

        env = str(_param("execution_environment", "")).strip().lower()
        if env != "real":
            raise RuntimeError(
                "mt3_pipeline_real.py requires "
                "/sawyer_auto_grasp/execution_environment=real")

        # Real perception must never silently fall back to hardcoded object data.
        self.use_perception = _bool("use_perception", True)
        if not self.use_perception:
            raise RuntimeError(
                "Real MT3 requires use_perception=true. "
                "Do not use simulation/default object coordinates on Sawyer.")
        self.require_depth_pose = _bool("require_depth_pose", True)

        # Kinesthetic demos are pose trajectories; make pose replay the real default.
        self.use_demo_replay = _bool("use_demo_replay", True)
        self.prefer_pose_replay = _bool("prefer_pose_replay", True)

        # Do not automatically contaminate the demo library before a real success
        # verifier is deliberately configured.
        self.auto_record_success = _bool("auto_record_success", False)

        # DRY RUN is the default for this real wrapper unless explicitly requested.
        if not rospy.has_param("~dry_run"):
            self.dry_run = _bool("dry_run", True)
        self.allow_real_execution = _bool("allow_real_execution", False)
        if not self.dry_run and not self.allow_real_execution:
            raise RuntimeError(
                "Real execution requested but allow_real_execution=false. "
                "Keep dry_run=true until the real executor/safety parameters are verified.")

        # Use configured real workspace z bounds instead of the old Gazebo cube range.
        workspace = _param("workspace", [0.10, -0.45, -0.20, 1.00, 0.45, 0.60])
        try:
            ws = [float(v) for v in workspace]
            if len(ws) == 6:
                self.object_z_min = float(_param("object_z_min", ws[2]))
                self.object_z_max = float(_param("object_z_max", ws[5]))
        except Exception:
            pass

        self.real_postcheck_mode = str(
            _param("real_postcheck_mode", "executor_only")).strip().lower()
        self.real_camera_frame = str(
            _param("camera_frame", "ascamera_hp60c_color_0"))

        # Real-only empirical Z alignment for camera-derived object top surfaces.
        # This does NOT modify the frozen eye-to-hand transform and does NOT
        # blindly shift arbitrary right_hand/grasp targets.
        self.enable_real_top_z_offset = _bool("enable_real_top_z_offset", True)
        self.real_top_z_offset_m = float(_param("real_top_z_offset_m", 0.044))

        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 REAL wrapper active")
        rospy.loginfo("  camera_frame: %s", self.real_camera_frame)
        rospy.loginfo("  dry_run: %s", self.dry_run)
        rospy.loginfo("  allow_real_execution: %s", self.allow_real_execution)
        rospy.loginfo("  use_demo_replay: %s", self.use_demo_replay)
        rospy.loginfo("  prefer_pose_replay: %s", self.prefer_pose_replay)
        rospy.loginfo("  real_postcheck_mode: %s", self.real_postcheck_mode)
        rospy.loginfo(
            "  real top-Z offset: enabled=%s offset=%+.1f mm",
            self.enable_real_top_z_offset,
            self.real_top_z_offset_m * 1000.0)
        rospy.loginfo("=" * 60)

    def step5_estimate_pose(self, test_data, demo_data, icp_result=None):
        """Run current mt3_generalize alignment with real-demo schema compatibility.

        record_demo_real.py stores the demonstrated object position under
        object_info.position_base.  The current simulation generalizer builds the
        explicit bottleneck-relative-to-object transform from
        object_pose_base_frame.position_m.  For real recorded demos only, expose
        the former through the latter in-memory before calling the shared parent
        implementation.  The recorded JSON and mt3_generalize.py are not modified.
        """
        patched_demo_data = demo_data
        entry = (demo_data or {}).get("demo_entry") if isinstance(demo_data, dict) else None
        if isinstance(entry, dict):
            obj_frame = (entry.get("object_pose_base_frame") or {}).get("position_m")
            raw = (entry.get("object_info") or {}).get("position_base")
            if (not obj_frame) and raw and len(raw) >= 3:
                patched_entry = copy.copy(entry)
                patched_entry["object_pose_base_frame"] = {
                    "position_m": {
                        "x": float(raw[0]),
                        "y": float(raw[1]),
                        "z": float(raw[2]),
                    },
                    "source": "real_demo_object_info_position_base_compat",
                }
                patched_demo_data = copy.copy(demo_data)
                patched_demo_data["demo_entry"] = patched_entry
                rospy.loginfo(
                    "  REAL demo schema bridge: object_info.position_base -> "
                    "object_pose_base_frame.position_m for bottleneck mapping")

        return super().step5_estimate_pose(
            test_data, patched_demo_data, icp_result=icp_result)

    def _is_real_camera_scene_package(self, package_dir):
        """Return True only for scene packages produced by the calibrated real camera."""
        meta_path = os.path.join(package_dir, "metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False

        pose = meta.get("pose") or {}
        source_frame = str(
            pose.get("source_frame")
            or pose.get("frame")
            or meta.get("camera_frame")
            or "")
        execution_environment = str(
            meta.get("execution_environment") or "").strip().lower()

        return (
            execution_environment == "real"
            or source_frame == self.real_camera_frame
        )

    def _robust_base_z_top(self, package_dir, percentile=90.0):
        """Return object top Z without ever applying the +44 mm correction twice.

        Formal recorded real-demo packages contain explicit robot-frame object
        bottom + size metadata.  That top is already in the corrected robot
        coordinate convention and must be used directly.  The empirical +44 mm
        correction is applied only when top Z is derived from camera-frame depth
        pointcloud geometry (the live-scene path).
        """
        meta = {}
        try:
            meta_path = os.path.join(package_dir, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        obj_pos = meta.get("object_position_base")
        obj_size = meta.get("object_size")
        if (obj_pos and obj_size and len(obj_pos) >= 3 and len(obj_size) >= 3):
            explicit_top = float(obj_pos[2]) + float(obj_size[2])
            rospy.loginfo(
                "  REAL demo top-Z from explicit base metadata: %.4f "
                "(no +44mm re-application) package=%s",
                explicit_top, os.path.basename(os.path.normpath(package_dir)))
            return explicit_top
        if meta.get("object_top_z_base") is not None:
            explicit_top = float(meta["object_top_z_base"])
            rospy.loginfo(
                "  REAL demo top-Z from explicit object_top_z_base: %.4f "
                "(no +44mm re-application) package=%s",
                explicit_top, os.path.basename(os.path.normpath(package_dir)))
            return explicit_top

        raw_top_z = super()._robust_base_z_top(
            package_dir, percentile=percentile)
        if raw_top_z is None:
            return None
        if not self.enable_real_top_z_offset:
            return raw_top_z
        if not self._is_real_camera_scene_package(package_dir):
            return raw_top_z

        corrected_top_z = float(raw_top_z) + float(self.real_top_z_offset_m)
        rospy.loginfo(
            "  REAL camera-derived top-Z correction: raw=%.4f offset=%+.4f corrected=%.4f "
            "package=%s",
            float(raw_top_z), float(self.real_top_z_offset_m),
            corrected_top_z, os.path.basename(os.path.normpath(package_dir)))
        return corrected_top_z

    def step1_load_test_image(self):
        """Strict ASC60C acquisition: no default/synthetic scene fallback."""
        rospy.loginfo("[Step 1/7 REAL] Waiting for ASC60C RGB-D + CameraInfo...")
        if self.perception is None:
            raise RuntimeError("Real PerceptionNode is unavailable")

        timeout_s = float(_param("perception_timeout_s", 8.0))
        if not self.perception.wait_for_registered_rgbd(timeout_s=timeout_s):
            raise RuntimeError("ASC60C RGB/depth/CameraInfo timeout")

        pose = self.perception.get_object_pose()
        if pose is None:
            raise RuntimeError(
                "Real object pose failed. Check LangSAM mask, registered depth and CameraInfo.")

        bgr = self.perception.latest_bgr
        if bgr is None and self.perception.head_image is not None:
            bgr = self.perception.bridge.imgmsg_to_cv2(
                self.perception.head_image, desired_encoding="bgr8")
            self.perception.latest_bgr = bgr
        if bgr is None:
            raise RuntimeError("ASC60C RGB image could not be converted")

        depth = self.perception._load_registered_depth()
        if depth is None:
            raise RuntimeError("ASC60C registered depth could not be converted")

        mask = self.perception.latest_clean_mask
        if mask is None:
            raise RuntimeError("No LangSAM target mask available after real perception")
        mask = np.asarray(mask).astype(bool)
        if mask.shape != np.asarray(depth).shape[:2]:
            raise RuntimeError(
                "Mask/depth resolution mismatch after real perception: %s vs %s" %
                (str(mask.shape), str(np.asarray(depth).shape[:2])))

        K = self.perception._camera_matrix()
        if K is None:
            raise RuntimeError("ASC60C CameraInfo intrinsics unavailable")

        source_frame = str(pose.get("source_frame") or "")
        if source_frame != self.real_camera_frame:
            raise RuntimeError(
                "Real pose source_frame=%s does not match calibrated camera_frame=%s" %
                (source_frame, self.real_camera_frame))

        rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
        pos = pose["position"]
        rospy.loginfo(
            "  REAL detected camera pose: [%.4f %.4f %.4f] method=%s frame=%s",
            pos[0], pos[1], pos[2], pose.get("method", ""), source_frame)
        return {
            "rgb": rgb,
            "depth": np.asarray(depth),
            "segmap": mask,
            "intrinsics": np.asarray(K, dtype=np.float64),
            "pose": pose,
        }

    def _package_source_frame(self, package_dir):
        """Never silently label a real point cloud as the old head_camera frame."""
        meta_path = os.path.join(package_dir, "metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pose = meta.get("pose") or {}
            return pose.get("source_frame") or pose.get("frame") or self.real_camera_frame
        except Exception:
            return self.real_camera_frame

    def _get_gazebo_object_pose(self):
        """Gazebo ground truth is intentionally unavailable in real mode."""
        return None

    def _real_postcheck(self, task_name):
        mode = self.real_postcheck_mode
        if mode in ("manual", "manual_param"):
            value = _param("manual_success_label", "")
            parsed = _optional_bool(value)
            if parsed is None:
                rospy.logwarn(
                    "REAL %s postcheck requested manual label, but manual_success_label "
                    "is unset; leaving postcheck unknown.", task_name)
                self.last_postcheck_info = {
                    "postcheck_success": "",
                    "postcheck_reason": "manual_success_label_unset",
                }
                return True
            self.last_postcheck_info = {
                "postcheck_success": bool(parsed),
                "postcheck_reason": "manual_success_label",
            }
            if not parsed:
                self.last_execution_failure_stage = "%s_verification" % task_name
                self.last_execution_failure_reason = "manual_success_label_false"
            return bool(parsed)

        # Until a real visual/post-contact verifier is implemented, do not pretend
        # that Gazebo validation exists. Keep executor result but label postcheck unknown.
        self.last_postcheck_info = {
            "postcheck_success": "",
            "postcheck_reason": "real_executor_only_no_gazebo_gt",
        }
        rospy.logwarn(
            "REAL %s postcheck: no Gazebo GT. Keeping executor result; "
            "postcheck is recorded as unknown.", task_name)
        return True

    def _validate_post_grasp_success(self, aligned):
        if self.dry_run or self.task_type != "grasp":
            return True
        return self._real_postcheck("grasp")

    def _validate_post_place_success(self):
        if self.dry_run or self.task_type != "pick_place":
            return True
        return self._real_postcheck("placement")

    def _real_executor_path(self):
        if self.task_type == "pick_place":
            configured = _param("place_executor_path", "")
            default_name = "mt3_sawyer_place_real.py"
        else:
            configured = _param("grasp_executor_path", "")
            default_name = "mt3_sawyer_grasp_real.py"
        if configured:
            return os.path.expanduser(str(configured))
        return os.path.expanduser(
            "~/ros_ws/src/sawyer_gazebo/scripts/%s" % default_name)

    def step7_execute(self, aligned):
        """Reuse shared param/replay preparation, then launch only a *_real executor."""
        requested_dry_run = bool(self.dry_run)

        # Parent dry-run path writes every grasp/bottleneck/replay ROS parameter but
        # does not launch mt3_sawyer_grasp.py / mt3_sawyer_place.py.
        self.dry_run = True
        try:
            prepared = super().step7_execute(aligned)
        finally:
            self.dry_run = requested_dry_run
        if not prepared:
            return False
        if requested_dry_run:
            rospy.loginfo("REAL DRY RUN: parameters/replay input prepared; no robot motion.")
            return True
        if not self.allow_real_execution:
            self.last_execution_failure_stage = "real_execution_gate"
            self.last_execution_failure_reason = "allow_real_execution_false"
            return False

        script = self._real_executor_path()
        if not os.path.isfile(script):
            self.last_execution_failure_stage = "real_executor_missing"
            self.last_execution_failure_reason = "missing_%s" % os.path.basename(script)
            rospy.logerr(
                "REAL executor not found: %s. Upload/check the real executor before motion.",
                script)
            return False

        rospy.logwarn("REAL ROBOT EXECUTION: launching %s", script)
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        failure_patterns = (
            "no motion plan found",
            "control_failed",
            "path_tolerance_violated",
            "execution completed: aborted",
            "reports status aborted",
        )
        output_failure = False
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                rospy.loginfo("  [REAL EXEC] %s", line)
                low = line.lower()
                if any(p in low for p in failure_patterns):
                    output_failure = True
        proc.wait()
        if proc.returncode != 0 or output_failure:
            self.last_execution_failure_stage = "real_executor"
            self.last_execution_failure_reason = (
                "executor_exit_%d" % proc.returncode
                if proc.returncode != 0 else "executor_output_reported_failure")
            self.last_planning_success = False if "plan" in self.last_execution_failure_reason else True
            return False

        if self.last_rollout_trajectory_path and os.path.exists(
                self.last_rollout_trajectory_path):
            try:
                with open(self.last_rollout_trajectory_path, "r", encoding="utf-8") as f:
                    rollout = json.load(f)
                if rollout.get("success") is False:
                    self.last_execution_failure_stage = "real_executor"
                    self.last_execution_failure_reason = "rollout_reported_success_false"
                    return False
            except Exception as exc:
                rospy.logwarn("REAL rollout success flag could not be read: %s", exc)

        self.last_planning_success = True
        return True

    def run(self):
        """Parent 7-step flow without Gazebo pre/post state or hardcoded -0.58 check."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 REAL Pipeline: Starting execution")
        rospy.loginfo("=" * 60)
        self._timing["run_start"] = rospy.get_time()

        import time
        t0 = time.time()
        test_data = self.step1_load_test_image()
        self._timing["perception_time_s"] = time.time() - t0
        live_package = self._export_scene_package(
            test_data,
            name="live_latest_real",
            role="live_scene_real",
            extra_metadata={
                "language_query": self.language_query,
                "execution_environment": "real",
                "camera_frame": self.real_camera_frame,
            })
        live_package = self._archive_live_scene_package(live_package)

        self.step2_init_scene_state(test_data)

        t_retrieval = time.time()
        best_demo, score = self.step3_retrieve_demo(test_data)
        self._timing["retrieval_time_s"] = time.time() - t_retrieval
        if score < 0.3:
            rospy.logwarn("  Low retrieval score (%.2f) — continuing to safety gates", score)

        t_alignment = time.time()
        demo_data = self.step4_load_demo(best_demo)
        demo_package = self._stored_demo_scene_package(best_demo)
        if demo_package is None:
            demo_package = self._export_scene_package(
                demo_data,
                name="demo_%s" % demo_data.get("name", best_demo.get("id", "unknown")),
                role="retrieved_demo",
                extra_metadata={
                    "demo_id": best_demo.get("id", ""),
                    "demo_source": demo_data.get("source", ""),
                    "retrieval_score": float(score),
                })
        icp_demo_package = self._select_demo_package_for_icp(demo_package, best_demo)
        icp_result = self._run_icp_registration(icp_demo_package, live_package)
        aligned = self.step5_estimate_pose(test_data, demo_data, icp_result=icp_result)
        self._timing["alignment_time_s"] = time.time() - t_alignment

        rospy.loginfo("[Visualization] Generating MT3 diagnostic outputs...")
        _sim.generate_all_official(test_data, demo_data, aligned, frame_no=self.frame_no)

        if not self._validate_perception_for_execution(test_data, aligned):
            self.last_executor_success = False
            self._record_experiment_result(
                "failed", test_data=test_data, aligned=aligned,
                best_demo=best_demo, score=score, live_package=live_package,
                icp_result=icp_result, reason="perception_safety_gate")
            return False

        aligned = self.step6_transform_bottleneck(aligned)
        self._reset_executor_timing_params()
        self.pre_execution_gazebo_pose = None
        self.last_postcheck_info = {}
        t_exec = time.time()
        success = self.step7_execute(aligned)
        self.last_executor_success = bool(success)
        self._timing["execution_time_s"] = time.time() - t_exec
        self._timing.update(self._read_executor_timing_params())

        if success:
            if self.task_type == "pick_place":
                success = self._validate_post_place_success()
            else:
                success = self._validate_post_grasp_success(aligned)

        self._record_experiment_result(
            "success" if success else "failed",
            test_data=test_data, aligned=aligned, best_demo=best_demo, score=score,
            live_package=live_package, icp_result=icp_result,
            reason=(self.last_execution_failure_reason if not success else ""))

        if success and self.auto_record_success:
            self._record_success_demo(
                aligned, best_demo, score,
                live_package=live_package, icp_result=icp_result)

        rospy.loginfo("=" * 60)
        if success:
            rospy.loginfo("MT3 REAL Pipeline: COMPLETED")
        else:
            rospy.logerr("MT3 REAL Pipeline: FAILED")
        rospy.loginfo("=" * 60)
        return bool(success)


if __name__ == "__main__":
    try:
        pipeline = MT3RealPipeline()
        sys.exit(0 if pipeline.run() else 1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Real MT3 pipeline interrupted")
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("mt3_pipeline_real failed: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)
