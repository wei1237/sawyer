#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Sawyer MT3 anchor-relative pick-and-place executor.

This is the real-robot continuation of the already-verified
``mt3_sawyer_real_grasp.py`` path.  It does *not* use a fixed scripted place
baseline.  The runtime chain is:

    target mask + anchor mask
        -> DualMaskAnchorPerception
        -> live target / live anchor geometry
        -> verified real top-grasp mapping and replay
        -> compute_anchor_place_target(demo relation, live anchor)
        -> mapped place bottleneck
        -> demo-relative place trajectory replay
        -> recorded gripper-open event
        -> post-release replay

The anchor relation in ``mt3_anchor_place_generalization.py`` is a translation
in Sawyer base/table axes.  Therefore place trajectory positions are translated
by the mapped place-bottleneck delta while recorded end-effector orientations
are preserved.

Default mode is dry-run.  Real Sawyer motion requires ``--execute``.
"""

from __future__ import print_function

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import rospy
from geometry_msgs.msg import Pose


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATCH_DIR = os.path.join(CODE_DIR, "real_perception_patch")
if REAL_PATCH_DIR not in sys.path:
    sys.path.insert(0, REAL_PATCH_DIR)

from mt3_anchor_place_generalization import compute_anchor_place_target
from mt3_anchor_perception_real import DualMaskAnchorPerception
from mt3_sawyer_real_grasp import (
    DEFAULT_LOG_DIR,
    RealTopGraspReplay,
    _pose_from_xyz_quat,
    _xyz_from_pose_msg,
)


DEFAULT_ANCHOR_DEMO_PATHS = [
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/real/recorded/"
        "cube_place_on_blue_platform_real.json"),
    os.path.expanduser(
        "~/code/learning_thousand_tasks/demo_library/real/recorded/"
        "cube_place_in_blue_socket_real.json"),
]


def _json_list(value):
    if value is None:
        return json.dumps([])
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                out.append(item)
        return json.dumps(out)
    return json.dumps(value)


def _norm_xyz(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(arr) < 3:
        raise RuntimeError("Expected xyz, got %s" % value)
    return arr[:3]


def _safe_size(value, fallback):
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(arr) >= 3 and np.all(np.isfinite(arr[:3])):
            arr = np.abs(arr[:3])
            if np.all(arr > 0.002) and np.all(arr < 0.40):
                return arr
    except Exception:
        pass
    return np.asarray(fallback, dtype=np.float64)


class RealAnchorPlaceReplay(RealTopGraspReplay):
    """Verified real top grasp + anchor-relative real place replay."""

    def __init__(self, demo_path="", trial_id="", target_mask_path="",
                 anchor_mask_path="", replay_velocity_scale=None,
                 object_offset_xyz=None, replay_offset_xyz=None,
                 vision_y_linear_calibration_enabled=False,
                 vision_y_piecewise_compensation_enabled=False,
                 move_to_start_pose=None, manual_success_label=""):
        if not demo_path:
            demo_path = self._find_default_anchor_demo()

        super(RealAnchorPlaceReplay, self).__init__(
            demo_path=demo_path,
            trial_id=(trial_id or time.strftime("real_place_%Y%m%d_%H%M%S")),
            replay_velocity_scale=replay_velocity_scale,
            object_offset_xyz=object_offset_xyz,
            replay_offset_xyz=replay_offset_xyz,
            vision_y_linear_calibration_enabled=(
                vision_y_linear_calibration_enabled),
            vision_y_piecewise_compensation_enabled=(
                vision_y_piecewise_compensation_enabled),
            move_to_start_pose=move_to_start_pose)

        self._validate_anchor_place_demo()

        self.target_mask_path = os.path.expanduser(
            target_mask_path or str(rospy.get_param(
                "~target_mask_path",
                "/mnt/hgfs2/ascamera_data/current_mask.npy")))
        self.anchor_mask_path = os.path.expanduser(
            anchor_mask_path or str(rospy.get_param(
                "~anchor_mask_path",
                "/mnt/hgfs2/ascamera_data/current_anchor_mask.npy")))

        anchor_size = (
            (self.demo.get("anchor_info") or {}).get("size_m") or
            rospy.get_param("~anchor_size_m", None))
        self.anchor_perception = DualMaskAnchorPerception(
            target_mask_path=self.target_mask_path,
            anchor_mask_path=self.anchor_mask_path,
            target_size=self.demo_size.tolist(),
            anchor_size=anchor_size)

        self.manual_success_label = str(manual_success_label or "")
        self.place_log_dir = os.path.expanduser(str(rospy.get_param(
            "~real_anchor_place_log_dir", DEFAULT_LOG_DIR)))

        self.target_top_percentile = float(rospy.get_param(
            "~target_top_percentile", 90.0))
        self.real_top_z_offset_m = float(rospy.get_param(
            "~real_top_z_offset_m", 0.044))
        self.place_transport_clearance_m = float(rospy.get_param(
            "~place_transport_clearance_m", 0.100))
        self.place_cartesian_eef_step = float(rospy.get_param(
            "~place_cartesian_eef_step", 0.004))
        self.place_cartesian_min_fraction = float(rospy.get_param(
            "~place_cartesian_min_fraction", 0.80))
        self.place_replay_stride = max(1, int(rospy.get_param(
            "~place_replay_stride", 2)))
        self.place_max_relative_xy_m = float(rospy.get_param(
            "~place_max_relative_xy_m", 0.20))
        self.place_max_relative_z_m = float(rospy.get_param(
            "~place_max_relative_z_m", 0.30))
        self.place_transport_velocity_scale = float(rospy.get_param(
            "~place_transport_velocity_scale", 0.08))
        self.place_transport_acceleration_scale = float(rospy.get_param(
            "~place_transport_acceleration_scale", 0.08))
        self.place_replay_velocity_scale = float(rospy.get_param(
            "~place_replay_velocity_scale", 0.06))
        self.place_replay_acceleration_scale = float(rospy.get_param(
            "~place_replay_acceleration_scale", 0.06))

        anchor_info = self.demo.get("anchor_info") or {}
        self.anchor_profile = {
            "name": anchor_info.get("name", "anchor"),
            "category": anchor_info.get("category", "unknown"),
            "size_m": anchor_info.get("size_m"),
            # The recorded demo's anchor->place relation already contains any
            # surface_z_offset that was applied while recording.  Keeping this
            # zero prevents compute_anchor_place_target() from adding it twice.
            "surface_z_offset": 0.0,
        }

        self.last_scene = None
        self.last_place_mapping = None
        self.place_debug = {}

        rospy.loginfo("Loaded real anchor-place demo: %s", self.demo_path)
        rospy.loginfo("Target mask: %s", self.target_mask_path)
        rospy.loginfo("Anchor mask: %s", self.anchor_mask_path)

    @staticmethod
    def _find_default_anchor_demo():
        for path in DEFAULT_ANCHOR_DEMO_PATHS:
            if os.path.isfile(path):
                return path
        raise RuntimeError(
            "No default real anchor-place demo found. Pass --demo_path. "
            "Tried: %s" % DEFAULT_ANCHOR_DEMO_PATHS)

    def _validate_anchor_place_demo(self):
        required = [
            "object_info",
            "anchor_info",
            "place_info",
            "grasp_trajectory",
            "place_trajectory",
            "place_bottleneck_pose_base_frame",
            "place_release_pose_base_frame",
        ]
        missing = [key for key in required if not self.demo.get(key)]
        if missing:
            raise RuntimeError(
                "Real anchor-place demo is missing fields: %s" %
                ", ".join(missing))

        # These fields are required by the already-verified parent grasp class.
        compat = [
            "bottleneck_pose_base_frame",
            "grasp_pose_base_frame",
            "trajectory",
            "top_grasp_mouth_center_calibration",
        ]
        compat_missing = [key for key in compat if not self.demo.get(key)]
        if compat_missing:
            raise RuntimeError(
                "Anchor-place demo predates the real-grasp compatibility "
                "schema. Re-record it with the updated "
                "record_anchor_place_demo_real.py. Missing: %s" %
                ", ".join(compat_missing))

        place_traj = self.demo.get("place_trajectory") or {}
        poses = place_traj.get("poses") or []
        if len(poses) < 2:
            raise RuntimeError("place_trajectory must contain at least 2 poses")

    def _points_base_for_detection(self, detection):
        try:
            return self.anchor_perception._points_to_base(
                detection.get("pose_source") or {})
        except Exception as exc:
            rospy.logwarn("Could not transform masked points to base: %s", exc)
            return None

    def _geometry_from_detection(self, detection, demo_size,
                                 target=False):
        pos = _norm_xyz(detection.get("position_base"))
        points_base = self._points_base_for_detection(detection)

        size = None
        raw_top_z = None
        if points_base is not None and len(points_base) >= 10:
            pts = np.asarray(points_base, dtype=np.float64)
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if len(pts) >= 10:
                low = np.percentile(pts, 5, axis=0)
                high = np.percentile(pts, 95, axis=0)
                size = high - low
                raw_top_z = float(np.percentile(
                    pts[:, 2], self.target_top_percentile))

        size = _safe_size(
            size if size is not None else detection.get("estimated_size"),
            demo_size)

        out = {
            "position": pos,
            "size": size,
            "method": detection.get("method", "dual_mask_registered_rgbd"),
            "geometry_center_correction": (
                (detection.get("pose_base") or {}).get(
                    "geometry_center_correction") or {}),
        }
        if target:
            if raw_top_z is not None:
                top_z = raw_top_z + self.real_top_z_offset_m
                top_source = "base_mask_points_p%.1f_plus_offset" % (
                    self.target_top_percentile)
            else:
                top_z = float(pos[2]) + 0.5 * abs(float(size[2]))
                top_source = "detected_center_plus_half_size_fallback"
            out.update({
                "top_z": float(top_z),
                "top_z_raw": raw_top_z,
                "top_z_offset_m": float(self.real_top_z_offset_m),
                "top_z_source": top_source,
            })
        return out

    def update_dual_perception_once(self):
        timeout_s = float(rospy.get_param(
            "~anchor_perception_timeout_s", 8.0))
        scene = self.anchor_perception.detect_scene(timeout_s=timeout_s)
        if scene is None:
            raise RuntimeError("Dual target/anchor real perception failed")

        target_det = scene["target"]
        anchor_det = scene["anchor"]
        target_geom = self._geometry_from_detection(
            target_det, self.demo_size, target=True)
        anchor_demo_size = (
            (self.demo.get("anchor_info") or {}).get("size_m") or
            [0.10, 0.10, 0.02])
        anchor_geom = self._geometry_from_detection(
            anchor_det, anchor_demo_size, target=False)

        # Publish target under the same contract already consumed by
        # RealTopGraspReplay.get_current_object_geometry(). This preserves its
        # existing offset/calibration behavior instead of duplicating it here.
        target_pos = target_geom["position"]
        rospy.set_param("/mt3/current_object_x", float(target_pos[0]))
        rospy.set_param("/mt3/current_object_y", float(target_pos[1]))
        rospy.set_param("/mt3/current_object_z", float(target_pos[2]))
        rospy.set_param(
            "/mt3/current_object_size_m",
            [float(v) for v in target_geom["size"]])
        rospy.set_param(
            "/mt3/current_object_top_z_base", float(target_geom["top_z"]))
        rospy.set_param("/mt3/current_object_z_semantics", "center_base")
        rospy.set_param(
            "/mt3/current_object_source_frame", "base_dual_mask_anchor_perception")

        anchor_pos = anchor_geom["position"]
        rospy.set_param("/mt3/current_anchor_x", float(anchor_pos[0]))
        rospy.set_param("/mt3/current_anchor_y", float(anchor_pos[1]))
        rospy.set_param("/mt3/current_anchor_z", float(anchor_pos[2]))
        rospy.set_param(
            "/mt3/current_anchor_size_m",
            [float(v) for v in anchor_geom["size"]])
        rospy.set_param(
            "/mt3/current_anchor_method", str(anchor_geom.get("method", "")))

        self.last_scene = {
            "raw_scene": scene,
            "target": target_geom,
            "anchor": anchor_geom,
        }
        rospy.loginfo(
            "REAL DUAL PERCEPTION target=[%.4f %.4f %.4f] top_z=%.4f "
            "anchor=[%.4f %.4f %.4f]",
            target_pos[0], target_pos[1], target_pos[2],
            target_geom["top_z"],
            anchor_pos[0], anchor_pos[1], anchor_pos[2])
        return self.last_scene

    def get_current_anchor_geometry(self):
        keys = [
            "/mt3/current_anchor_x",
            "/mt3/current_anchor_y",
            "/mt3/current_anchor_z",
        ]
        if not all(rospy.has_param(key) for key in keys):
            raise RuntimeError(
                "Missing live anchor params. Run with --update_perception so "
                "mt3_anchor_perception_real.py detects both masks.")
        pos = np.asarray([rospy.get_param(key) for key in keys],
                         dtype=np.float64)
        size = rospy.get_param(
            "/mt3/current_anchor_size_m",
            (self.demo.get("anchor_info") or {}).get(
                "size_m", [0.10, 0.10, 0.02]))
        return {
            "position": pos,
            "size": _safe_size(size, [0.10, 0.10, 0.02]),
            "method": rospy.get_param(
                "/mt3/current_anchor_method", "cached_ros_param"),
        }

    def _place_release_index(self):
        traj = self.demo.get("place_trajectory") or {}
        poses = traj.get("poses") or []
        explicit = traj.get("release_index")
        if explicit is not None:
            idx = max(0, min(len(poses) - 1, int(explicit)))
            return idx

        prev = None
        last_transition = None
        seen_closed = False
        fallback = None
        for idx, sample in enumerate(poses):
            state = self._gripper_binary(
                sample.get("gripper_next", sample.get("gripper_state")))
            if prev == 1 and state == 0:
                last_transition = idx
            if seen_closed and state == 0 and fallback is None:
                fallback = idx
            if state == 1:
                seen_closed = True
            if state is not None:
                prev = state
        out = last_transition if last_transition is not None else fallback
        if out is None:
            raise RuntimeError(
                "place_trajectory has no explicit release_index/open event")
        return int(out)

    def compute_place_mapping(self, current_target_geometry,
                              current_anchor_geometry):
        live_target = _norm_xyz(current_target_geometry["position"])
        live_anchor = _norm_xyz(current_anchor_geometry["position"])

        # Formal runtime uses the demonstrated relation.  An explicit ROS
        # override remains available for calibration/ablation, but is empty by
        # default and therefore cannot silently replace the MT3 relation.
        override = rospy.get_param("~anchor_place_override_offset_xyz", None)
        place_result = compute_anchor_place_target(
            live_anchor.tolist(),
            object_position_base=live_target.tolist(),
            object_size=[float(v) for v in current_target_geometry["size"]],
            demo_entry=self.demo,
            anchor_profile=self.anchor_profile,
            override_offset_xyz=override)

        mapped_place = _norm_xyz(place_result["place_xyz"])
        demo_place = _norm_xyz(
            (self.demo.get("place_info") or {}).get("place_xyz"))
        demo_bn_block = self.demo["place_bottleneck_pose_base_frame"]
        demo_bn = self._pose_block_position(demo_bn_block)
        demo_bn = _norm_xyz(demo_bn)
        demo_bn_q = self._pose_block_orientation(demo_bn_block)

        bottleneck_offset = demo_bn - demo_place
        mapped_bn = mapped_place + bottleneck_offset
        mapped_bn_pose = _pose_from_xyz_quat(mapped_bn, demo_bn_q)

        place_traj = self.demo.get("place_trajectory") or {}
        poses = place_traj.get("poses") or []
        release_idx = self._place_release_index()
        demo_release = self._pose_sample_position(poses[release_idx])
        mapped_release = mapped_bn + (demo_release - demo_bn)

        mapping = {
            "place_result": place_result,
            "demo_place_xyz": demo_place,
            "live_place_xyz": mapped_place,
            "demo_place_bottleneck_xyz": demo_bn,
            "place_bottleneck_offset_xyz": bottleneck_offset,
            "mapped_place_bottleneck_xyz": mapped_bn,
            "mapped_place_bottleneck_pose": mapped_bn_pose,
            "release_index": int(release_idx),
            "demo_release_tcp_xyz": demo_release,
            "mapped_release_tcp_xyz": mapped_release,
        }
        self.last_place_mapping = mapping

        rospy.loginfo("===== REAL MT3 ANCHOR PLACE MAPPING =====")
        rospy.loginfo("demo anchor=%s", (self.demo.get("anchor_info") or {}).get(
            "position_base"))
        rospy.loginfo("demo place=%s demo place bottleneck=%s",
                      demo_place, demo_bn)
        rospy.loginfo("live anchor=%s live place=%s",
                      live_anchor, mapped_place)
        rospy.loginfo("anchor->place offset=%s",
                      place_result.get("offset_xyz"))
        rospy.loginfo("place->bottleneck offset=%s", bottleneck_offset)
        rospy.loginfo("mapped place bottleneck=%s mapped release TCP=%s",
                      mapped_bn, mapped_release)
        return mapping

    def _mapped_place_waypoints(self, mapping):
        traj = self.demo.get("place_trajectory") or {}
        poses = traj.get("poses") or []
        demo_bn = _norm_xyz(mapping["demo_place_bottleneck_xyz"])
        mapped_bn = _norm_xyz(mapping["mapped_place_bottleneck_xyz"])
        release_idx = int(mapping["release_index"])

        selected_before = list(range(
            0, release_idx + 1, self.place_replay_stride))
        if not selected_before or selected_before[-1] != release_idx:
            selected_before.append(release_idx)
        selected_after = list(range(
            release_idx + 1, len(poses), self.place_replay_stride))
        if release_idx + 1 < len(poses) and (
                not selected_after or selected_after[-1] != len(poses) - 1):
            selected_after.append(len(poses) - 1)

        def mapped_pose(sample):
            demo_xyz = self._pose_sample_position(sample)
            rel = demo_xyz - demo_bn
            rel_xy = math.sqrt(float(rel[0]) ** 2 + float(rel[1]) ** 2)
            if rel_xy > self.place_max_relative_xy_m:
                raise RuntimeError(
                    "Place demo relative XY %.3fm exceeds safety gate %.3fm" %
                    (rel_xy, self.place_max_relative_xy_m))
            if abs(float(rel[2])) > self.place_max_relative_z_m:
                raise RuntimeError(
                    "Place demo relative Z %.3fm exceeds safety gate %.3fm" %
                    (abs(float(rel[2])), self.place_max_relative_z_m))
            xyz = mapped_bn + rel
            q = self._pose_block_orientation(sample)
            return _pose_from_xyz_quat(xyz, q)

        before = [mapped_pose(poses[i]) for i in selected_before]
        after = [mapped_pose(poses[i]) for i in selected_after]
        release_pose = mapped_pose(poses[release_idx])
        return before, release_pose, after, selected_before, selected_after

    def _set_place_motion_scale(self, velocity, acceleration):
        self.move_group.set_max_velocity_scaling_factor(max(
            0.01, min(1.0, float(velocity))))
        self.move_group.set_max_acceleration_scaling_factor(max(
            0.01, min(1.0, float(acceleration))))

    def _move_to_place_transport_pose(self, mapping):
        self._check_emergency_stop()
        mapped_bn_pose = mapping["mapped_place_bottleneck_pose"]
        approach = copy.deepcopy(mapped_bn_pose)
        current_z = float(self.move_group.get_current_pose().pose.position.z)
        approach.position.z = max(
            float(mapped_bn_pose.position.z) + self.place_transport_clearance_m,
            current_z)

        self._set_place_motion_scale(
            self.place_transport_velocity_scale,
            self.place_transport_acceleration_scale)
        target_xyz = np.asarray(_xyz_from_pose_msg(approach), dtype=np.float64)
        rospy.loginfo(
            "REAL PLACE TRANSPORT: moving carried object above mapped place "
            "bottleneck target=%s", target_xyz)
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(approach)
        t_exec = time.time()
        ok = self.move_group.go(wait=True)
        self.timing["robot_execution_time_s"] += time.time() - t_exec
        self.timing["robot_execution_call_count"] += 1
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self._check_emergency_stop()
        if not ok:
            raise RuntimeError(
                "MoveIt failed to transport object above mapped place bottleneck")
        rospy.sleep(0.3)
        actual = np.asarray(
            _xyz_from_pose_msg(self.move_group.get_current_pose().pose),
            dtype=np.float64)
        self.place_debug["place_transport_target_xyz"] = target_xyz.tolist()
        self.place_debug["place_transport_actual_xyz"] = actual.tolist()
        self.place_debug["place_transport_error_m"] = (
            actual - target_xyz).tolist()
        return approach

    def _execute_place_cartesian_segment(self, waypoints, label):
        if not waypoints:
            return 1.0
        self._check_emergency_stop()
        self._set_place_motion_scale(
            self.place_replay_velocity_scale,
            self.place_replay_acceleration_scale)
        self.move_group.set_start_state_to_current_state()
        t_plan = time.time()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, self.place_cartesian_eef_step, True)
        self.timing["planning_time_s"] += time.time() - t_plan
        self.timing["planning_call_count"] += 1
        rospy.loginfo(
            "%s cartesian fraction %.1f%% waypoints=%d",
            label, fraction * 100.0, len(waypoints))
        if fraction < self.place_cartesian_min_fraction:
            raise RuntimeError(
                "%s cartesian fraction %.1f%% < %.1f%%" %
                (label, fraction * 100.0,
                 self.place_cartesian_min_fraction * 100.0))
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

    def execute_place_replay(self, mapping, dry_run=True):
        before, release_pose, after, before_indices, after_indices = \
            self._mapped_place_waypoints(mapping)

        if dry_run:
            rospy.logwarn(
                "DRY RUN: mapped anchor-place replay computed; no Sawyer motion.")
            return {
                "success": True,
                "dry_run": True,
                "release_executed": False,
                "place_before_fraction": "",
                "place_after_fraction": "",
                "mapped_place_waypoints_before": len(before),
                "mapped_place_waypoints_after": len(after),
                "place_source_indices_before": before_indices,
                "place_source_indices_after": after_indices,
            }

        self._init_robot_interfaces()
        self._check_emergency_stop()
        self._move_to_place_transport_pose(mapping)

        # Enter the demonstrated terminal bottleneck, then replay the complete
        # local place interaction through the recorded release event.
        mapped_bn = mapping["mapped_place_bottleneck_pose"]
        bottleneck_fraction = self._execute_place_cartesian_segment(
            [mapped_bn], "real place: enter mapped bottleneck")

        # before includes the bottleneck pose as its first sample. Replaying it
        # again is harmless, but skip the duplicate when possible.
        before_exec = before
        if before_exec:
            first = np.asarray(_xyz_from_pose_msg(before_exec[0]))
            bn = np.asarray(_xyz_from_pose_msg(mapped_bn))
            if np.linalg.norm(first - bn) < 1.0e-6:
                before_exec = before_exec[1:]
        before_fraction = self._execute_place_cartesian_segment(
            before_exec, "real place: demonstrated approach to release")

        actual_release_before_open = self.move_group.get_current_pose().pose
        actual_release_xyz = np.asarray(
            _xyz_from_pose_msg(actual_release_before_open), dtype=np.float64)
        planned_release_xyz = np.asarray(
            _xyz_from_pose_msg(release_pose), dtype=np.float64)
        release_error = actual_release_xyz - planned_release_xyz
        rospy.logwarn(
            "REAL PLACE RELEASE TCP: planned=%s actual=%s "
            "delta=[%.1f %.1f %.1f]mm norm=%.1fmm",
            planned_release_xyz, actual_release_xyz,
            release_error[0] * 1000.0,
            release_error[1] * 1000.0,
            release_error[2] * 1000.0,
            np.linalg.norm(release_error) * 1000.0)

        self._check_emergency_stop()
        rospy.loginfo("REAL PLACE: opening gripper at recorded release event.")
        self.gripper.open()
        rospy.sleep(float(rospy.get_param("~place_release_wait_s", 1.0)))
        self._check_emergency_stop()

        after_fraction = self._execute_place_cartesian_segment(
            after, "real place: demonstrated post-release retreat")
        final_xyz = np.asarray(
            _xyz_from_pose_msg(self.move_group.get_current_pose().pose),
            dtype=np.float64)

        return {
            "success": True,
            "dry_run": False,
            "release_executed": True,
            "place_bottleneck_fraction": float(bottleneck_fraction),
            "place_before_fraction": float(before_fraction),
            "place_after_fraction": float(after_fraction),
            "planned_release_tcp_xyz": planned_release_xyz.tolist(),
            "actual_release_tcp_xyz": actual_release_xyz.tolist(),
            "release_tcp_error_m": release_error.tolist(),
            "release_tcp_error_norm_m": float(np.linalg.norm(release_error)),
            "final_retreat_tcp_xyz": final_xyz.tolist(),
            "mapped_place_waypoints_before": len(before),
            "mapped_place_waypoints_after": len(after),
            "place_source_indices_before": before_indices,
            "place_source_indices_after": after_indices,
        }

    def _write_place_log(self, target_geometry, anchor_geometry,
                         grasp_replay, grasp_result, mapping, place_result,
                         failure_reason=""):
        os.makedirs(self.place_log_dir, exist_ok=True)
        path = os.path.join(
            self.place_log_dir, "mt3_real_anchor_place_trials.csv")

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": "%.3f" % time.time(),
            "trial_id": self.trial_id,
            "demo_path": self.demo_path,
            "success": bool(place_result.get("success", False)),
            "dry_run": bool(place_result.get("dry_run", True)),
            "manual_success_label": self.manual_success_label,
            "failure_reason": failure_reason or place_result.get(
                "failure_reason", ""),
            "emergency_stop_requested": bool(self.emergency_stop_requested),

            "target_xyz": _json_list(target_geometry.get("position")),
            "target_size_m": _json_list(target_geometry.get("size")),
            "target_top_z_base": target_geometry.get("top_z", ""),
            "target_z_semantics": target_geometry.get("z_semantics", ""),
            "target_mask_path": self.target_mask_path,
            "anchor_xyz": _json_list(anchor_geometry.get("position")),
            "anchor_size_m": _json_list(anchor_geometry.get("size")),
            "anchor_method": anchor_geometry.get("method", ""),
            "anchor_mask_path": self.anchor_mask_path,

            "demo_place_xyz": _json_list(mapping.get("demo_place_xyz")),
            "anchor_place_offset_xyz": _json_list(
                (mapping.get("place_result") or {}).get("offset_xyz")),
            "mapped_place_xyz": _json_list(mapping.get("live_place_xyz")),
            "demo_place_bottleneck_xyz": _json_list(
                mapping.get("demo_place_bottleneck_xyz")),
            "place_bottleneck_offset_xyz": _json_list(
                mapping.get("place_bottleneck_offset_xyz")),
            "mapped_place_bottleneck_xyz": _json_list(
                mapping.get("mapped_place_bottleneck_xyz")),
            "mapped_release_tcp_xyz": _json_list(
                mapping.get("mapped_release_tcp_xyz")),

            "mapped_grasp_bottleneck_xyz": _json_list(
                grasp_replay.get("mapped_bottleneck_tcp")),
            "planned_grasp_close_tcp_xyz": _json_list(
                grasp_replay.get("planned_close_tcp")),
            "actual_grasp_close_tcp_xyz": _json_list(
                grasp_result.get("actual_close_tcp")),
            "grasp_hand_lift_m": grasp_result.get("hand_lift_m", ""),
            "grasp_before_close_fraction": grasp_result.get(
                "before_close_cartesian_fraction", ""),
            "grasp_lift_fraction": grasp_result.get(
                "after_close_cartesian_fraction", ""),

            "release_executed": bool(place_result.get(
                "release_executed", False)),
            "planned_release_tcp_xyz": _json_list(
                place_result.get("planned_release_tcp_xyz")),
            "actual_release_tcp_xyz": _json_list(
                place_result.get("actual_release_tcp_xyz")),
            "release_tcp_error_norm_m": place_result.get(
                "release_tcp_error_norm_m", ""),
            "place_bottleneck_fraction": place_result.get(
                "place_bottleneck_fraction", ""),
            "place_before_fraction": place_result.get(
                "place_before_fraction", ""),
            "place_after_fraction": place_result.get(
                "place_after_fraction", ""),
            "final_retreat_tcp_xyz": _json_list(
                place_result.get("final_retreat_tcp_xyz")),

            "perception_time_s": self.timing.get("perception_time_s", 0.0),
            "alignment_time_s": self.timing.get("alignment_time_s", 0.0),
            "planning_time_s": self.timing.get("planning_time_s", 0.0),
            "robot_execution_time_s": self.timing.get(
                "robot_execution_time_s", 0.0),
            "planning_call_count": self.timing.get("planning_call_count", 0),
            "robot_execution_call_count": self.timing.get(
                "robot_execution_call_count", 0),
            "execution_wall_time_s": self.timing.get(
                "execution_wall_time_s", 0.0),
            "total_time_s": self.timing.get("total_time_s", 0.0),
        }

        exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        rospy.loginfo("Real anchor-place trial logged: %s", path)
        return path

    def run_place(self, dry_run=True, update_perception=False):
        if not dry_run:
            self._move_to_real_start_pose()

        run_start = time.time()
        self.timing["run_start"] = run_start
        grasp_replay = {}
        grasp_result = {}
        mapping = {}
        place_result = {
            "success": False,
            "dry_run": bool(dry_run),
            "release_executed": False,
        }
        target_geometry = {}
        anchor_geometry = {}

        try:
            if update_perception:
                t0 = time.time()
                self.update_dual_perception_once()
                self.timing["perception_time_s"] = time.time() - t0
            else:
                self.timing["perception_time_s"] = 0.0

            # The target params are intentionally read through the already-
            # verified RealTopGraspReplay geometry contract.
            target_geometry = self.get_current_object_geometry()
            anchor_geometry = self.get_current_anchor_geometry()

            t_align = time.time()
            grasp_replay = self.make_real_replay_waypoints(target_geometry)
            grasp_replay = self.apply_replay_waypoint_offset(grasp_replay)
            mapping = self.compute_place_mapping(
                target_geometry, anchor_geometry)
            self.timing["alignment_time_s"] = time.time() - t_align

            if dry_run:
                grasp_result = self.execute_replay_waypoints(
                    grasp_replay, dry_run=True, pregrasp_only=False)
                place_result = self.execute_place_replay(
                    mapping, dry_run=True)
                place_result["success"] = True
            else:
                t_exec = time.time()
                grasp_result = self.execute_replay_waypoints(
                    grasp_replay, dry_run=False, pregrasp_only=False)
                if not bool(grasp_result.get("success", False)):
                    raise RuntimeError(
                        "Verified real grasp stage did not satisfy lift threshold")

                place_result = self.execute_place_replay(
                    mapping, dry_run=False)
                self.timing["execution_wall_time_s"] = time.time() - t_exec

            self.timing["total_time_s"] = time.time() - run_start
            self._write_place_log(
                target_geometry, anchor_geometry,
                grasp_replay, grasp_result, mapping, place_result)
            return bool(place_result.get("success", False))

        except Exception as exc:
            self.timing["total_time_s"] = time.time() - run_start
            if "execution_wall_time_s" not in self.timing:
                self.timing["execution_wall_time_s"] = 0.0
            place_result.update({
                "success": False,
                "failure_reason": str(exc),
            })
            try:
                self._write_place_log(
                    target_geometry, anchor_geometry,
                    grasp_replay, grasp_result, mapping, place_result,
                    failure_reason=str(exc))
            except Exception as log_exc:
                rospy.logerr("Failed to write real anchor-place failure log: %s",
                             log_exc)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Force dry-run even if --execute is present.")
    parser.add_argument("--execute", action="store_true",
                        help="Allow real Sawyer motion.")
    parser.add_argument("--update_perception", action="store_true",
                        help="Run dual target+anchor ASC60C perception once.")
    parser.add_argument("--demo_path", default="",
                        help="Real anchor-place demo JSON path.")
    parser.add_argument("--trial_id", default="",
                        help="Experiment trial id.")
    parser.add_argument("--target_mask_path", default="",
                        help="Target LangSAM mask .npy path.")
    parser.add_argument("--anchor_mask_path", default="",
                        help="Anchor LangSAM mask .npy path.")
    parser.add_argument("--manual_success_label", default="",
                        help="Optional manual outcome label written to CSV.")
    parser.add_argument("--replay_velocity_scale", type=float, default=None,
                        help="Extra multiplier for verified grasp replay speed.")
    parser.add_argument("--object_offset_x", type=float, default=0.0)
    parser.add_argument("--object_offset_y", type=float, default=0.0)
    parser.add_argument("--object_offset_z", type=float, default=0.0)
    parser.add_argument("--replay_offset_x", type=float, default=0.0)
    parser.add_argument("--replay_offset_y", type=float, default=0.0)
    parser.add_argument("--replay_offset_z", type=float, default=0.0)
    parser.add_argument("--enable_vision_y_linear_calibration",
                        action="store_true")
    parser.add_argument("--enable_vision_y_piecewise_compensation",
                        action="store_true")
    parser.add_argument("--move_to_start_pose", action="store_true")
    parser.add_argument("--disable_move_to_start_pose", action="store_true")
    args = parser.parse_args()

    dry_run = bool(args.dry_run or not args.execute)
    move_to_start_pose = None
    if args.move_to_start_pose:
        move_to_start_pose = True
    if args.disable_move_to_start_pose:
        move_to_start_pose = False

    robot = RealAnchorPlaceReplay(
        demo_path=args.demo_path,
        trial_id=args.trial_id,
        target_mask_path=args.target_mask_path,
        anchor_mask_path=args.anchor_mask_path,
        replay_velocity_scale=args.replay_velocity_scale,
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
        vision_y_linear_calibration_enabled=bool(
            args.enable_vision_y_linear_calibration),
        vision_y_piecewise_compensation_enabled=bool(
            args.enable_vision_y_piecewise_compensation),
        move_to_start_pose=move_to_start_pose,
        manual_success_label=args.manual_success_label)

    ok = robot.run_place(
        dry_run=dry_run,
        update_perception=bool(args.update_perception))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("mt3_sawyer_real_place failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)
