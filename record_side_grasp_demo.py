#!/usr/bin/env python3
"""
Record an MT3-style side-grasp demo for tall objects.

The demo records a bottleneck pose beside the object, a horizontal approach to
the object middle height, gripper close, and a vertical lift. It saves the same
JSON trajectory format as record_demo.py, plus an RGB-D/mask/pointcloud scene
package for ICP retrieval/alignment.
"""

import copy
import json
import math
import os
import threading

import geometry_msgs.msg
import moveit_commander
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler, quaternion_multiply

try:
    from intera_interface import Gripper
except Exception:
    Gripper = None

try:
    from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
except Exception:
    SolvePositionIK = None
    SolvePositionIKRequest = None

try:
    from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
except Exception:
    GetPositionIK = None
    GetPositionIKRequest = None

from mt3_scene_package import save_scene_package
from record_demo import DemoRecorder, OUTPUT_DIR


class SideGraspDemoRecorder(DemoRecorder):
    def __init__(
            self, object_x, object_y, object_z, object_size, demo_name,
            approach_axis="-y", gripper_roll=math.pi / 2.0,
            gripper_pitch=0.0, gripper_yaw_deg=None, mask_path="",
            pregrasp_distance=0.12, side_surface_margin=0.005,
            flange_approach_offset=0.040, side_flange_z_offset=0.040,
            lift_height=0.04, auto_try_axes=False,
            use_ik_motion=True, motion_tip_name="right_gripper_tip",
            tip_grasp_offset=0.0, ik_service_mode="moveit",
            right_hand_to_tip_z=0.13562,
            reach_then_roll=True, post_bottleneck_wrist_roll_deg=90.0,
            staged_side_approach=True, side_entry_lateral_offset=0.055,
            side_final_lateral_offset=None,
            side_y_final_lateral_sign=1.0,
            stage_c_commit_remaining=0.045,
            stage_c_cartesian=True,
            side_grasp_height_fraction=0.65,
            side_use_mouth_center=True,
            side_mouth_clearance=0.006,
            side_mouth_align_tolerance=0.020,
            side_mouth_z_tolerance=0.035):
        super().__init__(object_x, object_y, object_z, object_size, demo_name)
        self.approach_axis = str(approach_axis)
        self.gripper_roll = float(gripper_roll)
        self.gripper_pitch = float(gripper_pitch)
        self.gripper_yaw = self._default_yaw_for_axis() if gripper_yaw_deg is None else math.radians(float(gripper_yaw_deg))
        self.gripper_yaw_deg = math.degrees(self.gripper_yaw)
        self.mask_path = str(mask_path or "")
        self.pregrasp_distance = float(pregrasp_distance)
        self.side_surface_margin = float(side_surface_margin)
        self.flange_approach_offset = float(flange_approach_offset)
        self.side_flange_z_offset = float(side_flange_z_offset)
        self.lift_height = float(lift_height)
        self.auto_try_axes = bool(auto_try_axes)
        self.use_ik_motion = bool(use_ik_motion)
        self.motion_tip_name = str(motion_tip_name or "right_gripper_tip")
        self.tip_grasp_offset = float(tip_grasp_offset)
        self.ik_service_mode = str(ik_service_mode or "moveit").strip().lower()
        self.right_hand_to_tip_z = float(right_hand_to_tip_z)
        self.reach_then_roll = bool(reach_then_roll)
        if self._is_y_axis(self.approach_axis) and self.reach_then_roll:
            # For true +y/-y side grasps, the gripper tip axis itself must
            # point along the approach direction. Rolling the wrist after
            # reaching only changes finger opening; it leaves the hand facing
            # the old x direction and makes a finger sweep into the object.
            rospy.loginfo(
                "Y-axis side grasp: using direct y-facing wrist search "
                "instead of reach-then-roll.")
            self.reach_then_roll = False
        self.post_bottleneck_wrist_roll = math.radians(
            float(post_bottleneck_wrist_roll_deg))
        self.staged_side_approach = bool(staged_side_approach)
        self.side_entry_lateral_offset = float(side_entry_lateral_offset)
        self.side_y_final_lateral_sign = 1.0 if float(side_y_final_lateral_sign) >= 0.0 else -1.0
        if self._is_y_axis(self.approach_axis) and abs(self.side_entry_lateral_offset - 0.055) < 1e-9:
            # For +y/-y side grasps the final approach must really happen
            # along y. The old x offset made one finger sweep into tall
            # cylinders before the mouth was centered.
            self.side_entry_lateral_offset = 0.0
        self.stage_c_commit_remaining = float(stage_c_commit_remaining)
        self.stage_c_cartesian = bool(stage_c_cartesian)
        self.side_grasp_height_fraction = max(
            0.25, min(0.85, float(side_grasp_height_fraction)))
        self.side_use_mouth_center = bool(side_use_mouth_center)
        self.side_mouth_clearance = float(side_mouth_clearance)
        self.side_mouth_align_tolerance = float(side_mouth_align_tolerance)
        self.side_mouth_z_tolerance = float(side_mouth_z_tolerance)
        if self.side_use_mouth_center and self.motion_tip_name == "right_gripper_tip":
            rospy.loginfo(
                "Mouth-center side grasp: using right_hand as the controlled "
                "MoveIt link, then aligning the actual finger-mouth center.")
            self.motion_tip_name = "right_hand"
            self.use_ik_motion = False
        if side_final_lateral_offset is None:
            self.side_final_lateral_offset = self._auto_side_final_lateral_offset()
            self.side_final_lateral_offset_source = "auto_object_width"
        else:
            self.side_final_lateral_offset = float(side_final_lateral_offset)
            self.side_final_lateral_offset_source = "manual_param"
        self.gripper = None
        try:
            self.move_group.set_planning_time(10.0)
            self.move_group.set_num_planning_attempts(5)
        except Exception:
            pass

    def _axis_vector(self):
        return self._axis_vector_for(self.approach_axis)

    def _axis_vector_for(self, approach_axis):
        axis = str(approach_axis).strip().lower()
        if axis in ["+x", "x+"]:
            return np.array([1.0, 0.0, 0.0])
        if axis in ["-x", "x-"]:
            return np.array([-1.0, 0.0, 0.0])
        if axis in ["+y", "y+"]:
            return np.array([0.0, 1.0, 0.0])
        if axis in ["-y", "y-"]:
            return np.array([0.0, -1.0, 0.0])
        raise ValueError("approach_axis must be one of +x, -x, +y, -y")

    def _is_y_axis(self, approach_axis=None):
        axis = str(approach_axis or self.approach_axis).strip().lower()
        return axis in ["+y", "y+", "-y", "y-"]

    def _default_yaw_for_axis(self):
        return self._default_yaw_for_axis_value(self.approach_axis)

    def _default_yaw_for_axis_value(self, approach_axis):
        axis = str(approach_axis).strip().lower()
        if axis in ["+x", "x+"]:
            return 0.0
        if axis in ["-x", "x-"]:
            return math.pi
        if axis in ["+y", "y+"]:
            return math.pi / 2.0
        if axis in ["-y", "y-"]:
            return -math.pi / 2.0
        return -math.pi / 2.0

    def _candidate_axes(self):
        preferred = self.approach_axis.strip().lower()
        axes = [preferred]
        if self.auto_try_axes:
            for axis in ["+y", "-y", "+x", "-x"]:
                if axis not in axes:
                    axes.append(axis)
        return axes

    def _candidate_z_offsets(self):
        if getattr(self, "side_use_mouth_center", False):
            return [0.0]
        if self.motion_tip_name == "right_gripper_tip":
            # When commanding the gripper tip directly, the target height is
            # the object's visual center. The flange offset is only needed when
            # MoveIt is asked to place the right_hand link instead of the tip.
            return [0.0, -0.010, 0.010, -0.020, 0.020]
        offsets = [self.side_flange_z_offset]
        for offset in [0.060, 0.080, 0.100, 0.030]:
            if all(abs(offset - v) > 1e-6 for v in offsets):
                offsets.append(offset)
        return offsets

    def _auto_side_final_lateral_offset(self, approach_axis=None):
        sx, sy, _ = [float(v) for v in self.object_size]
        axis = str(approach_axis or self.approach_axis).strip().lower()
        if axis in ["+y", "y+", "-y", "y-"]:
            lateral_width = sx
        else:
            lateral_width = sy
        # Move from one-fingertip contact toward the mouth center using object
        # width, but keep the shift conservative because deeper side-grasp tip
        # targets quickly become unreachable near the table.
        return max(0.010, min(0.024, lateral_width * 0.40))

    def _candidate_bottleneck_z_offsets(self):
        if getattr(self, "side_use_mouth_center", False):
            return [0.040]
        if self.motion_tip_name == "right_gripper_tip":
            # The side grasp itself should be around the object's center, but
            # Sawyer often cannot solve IK for a low sideways bottleneck. Reach
            # a slightly higher pregrasp pose first, then descend/approach.
            return [0.040, 0.030, 0.050, 0.020, 0.060, 0.010, 0.0]
        return self._candidate_z_offsets()

    def _side_orientation(self):
        q = quaternion_from_euler(
            self.gripper_roll, self.gripper_pitch, self.gripper_yaw)
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    def _rotation_from_rpy(self, roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0.0],
                       [sy, cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz.dot(ry).dot(rx)

    def _side_orientation_candidates(self, approach_axis=None):
        """Candidate wrist orientations for side grasp; MoveIt will pick reachable one.

        For a side grasp, the gripper tip axis should point from the pregrasp
        side toward the object, while the finger opening axis should stay
        horizontal. This avoids the vertical up/down finger pose that cannot
        clamp a tall cylinder laterally.
        """
        yaw = self.gripper_yaw
        if approach_axis is not None and rospy.get_param("~gripper_yaw_deg", None) is None:
            yaw = self._default_yaw_for_axis_value(approach_axis)
        axis = approach_axis or self.approach_axis
        desired_tip_axis = -self._axis_vector_for(axis)
        desired_tip_axis = desired_tip_axis / max(1e-9, np.linalg.norm(desired_tip_axis))

        values = [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]
        yaw_values = [yaw, yaw + math.pi, yaw - math.pi,
                      yaw + math.pi / 2.0, yaw - math.pi / 2.0]
        rpy_scored = []
        for roll in values:
            for pitch in values:
                for yaw_val in yaw_values:
                    rot = self._rotation_from_rpy(roll, pitch, yaw_val)
                    open_axis = rot.dot(np.array([0.0, 1.0, 0.0]))
                    tip_axis = rot.dot(np.array([0.0, 0.0, 1.0]))
                    tip_score = float(np.dot(tip_axis, desired_tip_axis))
                    open_horizontal = 1.0 - abs(float(open_axis[2]))
                    open_perp_to_approach = 1.0 - abs(float(np.dot(open_axis, desired_tip_axis)))
                    score = 4.0 * tip_score + 2.0 * open_horizontal + open_perp_to_approach
                    if tip_score > 0.70 and abs(float(open_axis[2])) < 0.35:
                        rpy_scored.append((score, roll, pitch, yaw_val, open_axis, tip_axis))

        fallback_rpy = [
            (self.gripper_roll, self.gripper_pitch, yaw),
            (-math.pi / 2.0, 0.0, yaw),
            (math.pi / 2.0, 0.0, yaw),
            (math.pi, math.pi / 2.0, yaw),
            (math.pi, -math.pi / 2.0, yaw),
            (0.0, math.pi / 2.0, yaw),
            (0.0, -math.pi / 2.0, yaw),
            (math.pi / 2.0, math.pi / 2.0, yaw),
            (-math.pi / 2.0, -math.pi / 2.0, yaw),
        ]
        rpy_scored.sort(key=lambda item: item[0], reverse=True)
        candidates = []
        seen = set()

        def add_quat_candidate(quat, rpy, source):
            key = tuple(round(float(v), 4) for v in quat)
            if key in seen:
                return
            seen.add(key)
            open_axis = np.array(self._rotate_by_quat(quat, [0.0, 1.0, 0.0]))
            tip_axis = np.array(self._rotate_by_quat(quat, [0.0, 0.0, 1.0]))
            candidates.append({
                "rpy": [float(v) for v in rpy],
                "quat": [float(v) for v in quat],
                "open_axis": [float(v) for v in open_axis],
                "tip_axis": [float(v) for v in tip_axis],
                "source": source,
            })

        if self.reach_then_roll:
            # First reach with the old side-grasp wrist poses that are known to
            # be reachable, then roll the wrist in-place before approaching.
            for roll, pitch, yaw_val in fallback_rpy:
                q = quaternion_from_euler(roll, pitch, yaw_val)
                add_quat_candidate(
                    q, [roll, pitch, yaw_val],
                    "reach_then_roll_bottleneck")
            return candidates

        for _, roll, pitch, yaw_val, _, _ in rpy_scored:
            q = quaternion_from_euler(roll, pitch, yaw_val)
            add_quat_candidate(q, [roll, pitch, yaw_val], "horizontal_search")

        if self._is_y_axis(axis):
            # For a requested +y/-y side grasp, do not fall back to x-facing
            # wrist poses. They are reachable, but they make the fingers sweep
            # into the object instead of letting the object enter the gripper
            # mouth from the side.
            return candidates

        # The old candidates include wrist poses known to be reachable in this
        # Sawyer Gazebo setup. Rotate each of those around the gripper's own tip
        # axis by 90 deg first, so the finger opening changes from vertical to
        # horizontal while keeping a similar arm reachability.
        local_z_rotations = [math.pi / 2.0, -math.pi / 2.0, math.pi]
        for roll, pitch, yaw_val in fallback_rpy:
            base_q = quaternion_from_euler(roll, pitch, yaw_val)
            for delta in local_z_rotations:
                delta_q = quaternion_from_euler(0.0, 0.0, delta)
                q = quaternion_multiply(base_q, delta_q)
                open_axis = np.array(self._rotate_by_quat(q, [0.0, 1.0, 0.0]))
                if abs(float(open_axis[2])) < 0.35:
                    add_quat_candidate(
                        q, [roll, pitch, yaw_val],
                        "reachable_pose_tip_axis_roll")

        # Keep originals as the very last fallback. They may be reachable but
        # can have vertical finger opening, so they should not be selected first.
        for roll, pitch, yaw_val in fallback_rpy:
            q = quaternion_from_euler(roll, pitch, yaw_val)
            add_quat_candidate(q, [roll, pitch, yaw_val], "legacy_fallback")
        return candidates

    def _half_extent_along_approach(self):
        return self._half_extent_along_approach_for_axis(self.approach_axis)

    def _half_extent_along_approach_for_axis(self, approach_axis):
        d = np.abs(self._axis_vector_for(approach_axis))
        sx, sy, _ = [float(v) for v in self.object_size]
        if d[0] > 0.5:
            return sx / 2.0
        return sy / 2.0

    def _make_pose(self, xyz, quat):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose

    def _lookup_link_pose(self, link_name):
        try:
            tf = self.tf_buffer.lookup_transform(
                "base", link_name, rospy.Time(0), rospy.Duration(0.6))
            pos = tf.transform.translation
            ori = tf.transform.rotation
            pose = geometry_msgs.msg.Pose()
            pose.position.x = pos.x
            pose.position.y = pos.y
            pose.position.z = pos.z
            pose.orientation.x = ori.x
            pose.orientation.y = ori.y
            pose.orientation.z = ori.z
            pose.orientation.w = ori.w
            return pose
        except Exception:
            return None

    def _current_motion_tip_pose(self):
        pose = self._lookup_link_pose(self.motion_tip_name)
        if pose is not None:
            return pose
        return self.move_group.get_current_pose().pose

    def _pose_xyz(self, pose):
        return np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=np.float64)

    def _copy_pose_with_xyz(self, pose, xyz):
        out = copy.deepcopy(pose)
        out.position.x = float(xyz[0])
        out.position.y = float(xyz[1])
        out.position.z = float(xyz[2])
        return out

    def _get_gripper_mouth_state(self):
        left = self._lookup_link_pose("right_gripper_l_finger_tip")
        right = self._lookup_link_pose("right_gripper_r_finger_tip")
        tip = self._current_motion_tip_pose()
        if left is None or right is None or tip is None:
            rospy.logwarn(
                "  Gripper mouth TF unavailable; falling back to %s target",
                self.motion_tip_name)
            return None
        left_xyz = self._pose_xyz(left)
        right_xyz = self._pose_xyz(right)
        mouth_xyz = 0.5 * (left_xyz + right_xyz)
        tip_xyz = self._pose_xyz(tip)
        opening = float(np.linalg.norm(left_xyz - right_xyz))
        return {
            "left": left_xyz,
            "right": right_xyz,
            "center": mouth_xyz,
            "tip": tip_xyz,
            "offset": mouth_xyz - tip_xyz,
            "opening": opening,
        }

    def _command_pose_for_mouth_center(self, reference_pose, mouth_center, mouth_offset):
        command_xyz = np.array(mouth_center, dtype=np.float64) - np.array(mouth_offset, dtype=np.float64)
        return self._copy_pose_with_xyz(reference_pose, command_xyz)

    def _opening_half_extent_for_axis(self):
        sx, sy, _ = [float(v) for v in self.object_size]
        if self._is_y_axis(self.approach_axis):
            return sx / 2.0
        return sy / 2.0

    def _log_mouth_alignment(self, label, desired_mouth_center, require_ok=False):
        state = self._get_gripper_mouth_state()
        if state is None:
            return not require_ok
        center = state["center"]
        desired = np.array(desired_mouth_center, dtype=np.float64)
        approach = self._axis_vector_for(self.approach_axis)
        lateral_delta = center - desired
        approach_error = abs(float(np.dot(lateral_delta, approach)))
        if self._is_y_axis(self.approach_axis):
            lateral_error = abs(float(lateral_delta[0]))
        else:
            lateral_error = abs(float(lateral_delta[1]))
        z_error = abs(float(lateral_delta[2]))
        half_opening = state["opening"] / 2.0
        object_half = self._opening_half_extent_for_axis()
        finger_clearance = half_opening - object_half - lateral_error
        rospy.loginfo(
            "  %s mouth check: center=[%.3f, %.3f, %.3f] desired=[%.3f, %.3f, %.3f] "
            "approach_err=%.1fcm lateral_err=%.1fcm z_err=%.1fcm opening=%.1fcm clearance=%.1fcm",
            label,
            center[0], center[1], center[2],
            desired[0], desired[1], desired[2],
            approach_error * 100.0,
            lateral_error * 100.0,
            z_error * 100.0,
            state["opening"] * 100.0,
            finger_clearance * 100.0)
        if not require_ok:
            return True
        if lateral_error > self.side_mouth_align_tolerance:
            rospy.logerr(
                "  %s blocked: mouth lateral error %.1fcm > %.1fcm",
                label, lateral_error * 100.0,
                self.side_mouth_align_tolerance * 100.0)
            return False
        if z_error > self.side_mouth_z_tolerance:
            rospy.logerr(
                "  %s blocked: mouth z error %.1fcm > %.1fcm",
                label, z_error * 100.0,
                self.side_mouth_z_tolerance * 100.0)
            return False
        if finger_clearance < -self.side_mouth_clearance:
            rospy.logerr(
                "  %s blocked: finger corridor too narrow by %.1fcm",
                label, -finger_clearance * 100.0)
            return False
        return True

    def _solve_moveit_ik(self, pose, label="pose"):
        if GetPositionIK is None or GetPositionIKRequest is None:
            rospy.logwarn("  MoveIt IK service type unavailable")
            return None
        requests = [(self.motion_tip_name, pose)]
        if self.motion_tip_name == "right_gripper_tip":
            hand_pose = copy.deepcopy(pose)
            offset = self._rotate_by_quat(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                [0.0, 0.0, self.right_hand_to_tip_z])
            hand_pose.position.x = float(pose.position.x - offset[0])
            hand_pose.position.y = float(pose.position.y - offset[1])
            hand_pose.position.z = float(pose.position.z - offset[2])
            requests.append(("right_hand", hand_pose))
        ns = "/robot/compute_ik"
        iksvc = rospy.ServiceProxy(ns, GetPositionIK)
        try:
            rospy.wait_for_service(ns, 1.5)
        except Exception as exc:
            rospy.logwarn("  MoveIt IK service failed for %s: %s", label, exc)
            return None
        for link_name, request_pose in requests:
            ikreq = GetPositionIKRequest()
            ikreq.ik_request.group_name = "right_arm"
            ikreq.ik_request.robot_state = self.robot.get_current_state()
            ikreq.ik_request.timeout = rospy.Duration(0.8)
            ikreq.ik_request.avoid_collisions = True
            ikreq.ik_request.ik_link_name = link_name
            stamped = geometry_msgs.msg.PoseStamped()
            stamped.header.stamp = rospy.Time.now()
            stamped.header.frame_id = "base"
            stamped.pose = request_pose
            ikreq.ik_request.pose_stamped = stamped
            try:
                resp = iksvc(ikreq)
            except Exception as exc:
                rospy.logwarn("  MoveIt IK request failed for %s: %s", label, exc)
                continue
            if resp.error_code.val == resp.error_code.SUCCESS:
                if link_name != self.motion_tip_name:
                    rospy.loginfo(
                        "  MoveIt IK solved %s using %s converted from %s",
                        label, link_name, self.motion_tip_name)
                return {
                    str(name): float(value)
                    for name, value in zip(resp.solution.joint_state.name,
                                           resp.solution.joint_state.position)
                    if str(name).startswith("right_j")
                }
            rospy.loginfo(
                "  MoveIt IK rejected %s for link=%s code=%s",
                label, link_name, resp.error_code.val)
        return None

    def _solve_intera_ik(self, pose, label="pose"):
        if SolvePositionIK is None or SolvePositionIKRequest is None:
            rospy.logwarn("  IK service type unavailable; falling back to MoveIt pose planning")
            return None
        ns = "/ExternalTools/right/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        stamped = geometry_msgs.msg.PoseStamped()
        stamped.header.stamp = rospy.Time.now()
        stamped.header.frame_id = "base"
        stamped.pose = pose
        ikreq.pose_stamp.append(stamped)
        ikreq.tip_names.append(self.motion_tip_name)
        try:
            rospy.wait_for_service(ns, 1.5)
            resp = iksvc(ikreq)
        except Exception as exc:
            rospy.logwarn("  IK service failed for %s: %s", label, exc)
            return None
        if not resp.result_type or resp.result_type[0] <= 0:
            rospy.loginfo(
                "  IK rejected %s for tip=%s", label, self.motion_tip_name)
            return None
        return dict(zip(resp.joints[0].name, resp.joints[0].position))

    def _solve_ik(self, pose, label="pose"):
        if self.ik_service_mode in ["moveit", "auto"]:
            joints = self._solve_moveit_ik(pose, label)
            if joints:
                return joints
            if self.ik_service_mode == "moveit":
                return None
        return self._solve_intera_ik(pose, label)

    def _init_gripper(self):
        if Gripper is None:
            rospy.logwarn("  intera_interface.Gripper not available; gripper commands skipped")
            return False
        if self.gripper is None:
            self.gripper = Gripper("right_gripper")
            if not self.gripper.is_calibrated():
                self.gripper.calibrate()
                rospy.sleep(1.0)
            self.gripper.set_cmd_velocity(0.1)
        return True

    def _open_gripper(self):
        if not self._init_gripper():
            return False
        self.gripper_command_state = 0
        self.gripper.open()
        rospy.sleep(0.8)
        return True

    def _close_gripper(self):
        if not self._init_gripper():
            return False
        self.gripper_command_state = 1
        self.gripper.close()
        rospy.sleep(1.3)
        return True

    def _move_pose_target(self, pose, label, velocity=0.12, acceleration=0.12):
        self.move_group.set_end_effector_link("right_hand")
        self.move_group.set_max_velocity_scaling_factor(velocity)
        self.move_group.set_max_acceleration_scaling_factor(acceleration)
        self.move_group.set_pose_target(pose)
        ok = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        rospy.sleep(0.5)
        if not ok:
            rospy.logerr("  %s failed.", label)
        return bool(ok)

    def _move_tip_pose_target(self, pose, label, velocity=0.12, acceleration=0.12):
        if self.use_ik_motion:
            joints = self._solve_ik(pose, label)
            if joints is None:
                if self.motion_tip_name == "right_hand":
                    rospy.logwarn(
                        "  %s IK failed for right_hand; falling back to MoveIt pose target.",
                        label)
                    return self._move_pose_target(
                        pose, label, velocity=velocity, acceleration=acceleration)
                rospy.logerr("  %s failed: no IK solution for %s", label, self.motion_tip_name)
                return False
            self.move_group.set_max_velocity_scaling_factor(velocity)
            self.move_group.set_max_acceleration_scaling_factor(acceleration)
            self.move_group.set_joint_value_target(joints)
            ok = self.move_group.go(wait=True)
            self.move_group.stop()
            rospy.sleep(0.35)
            if not ok:
                rospy.logerr("  %s failed while executing IK joint target.", label)
            return bool(ok)
        return self._move_pose_target(pose, label, velocity, acceleration)

    def _pose_for_move_group_link(self, pose):
        if self.motion_tip_name != "right_gripper_tip":
            return copy.deepcopy(pose)
        hand_pose = copy.deepcopy(pose)
        offset = self._rotate_by_quat(
            [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w],
            [0.0, 0.0, self.right_hand_to_tip_z])
        hand_pose.position.x = float(pose.position.x - offset[0])
        hand_pose.position.y = float(pose.position.y - offset[1])
        hand_pose.position.z = float(pose.position.z - offset[2])
        return hand_pose

    def _execute_cartesian_to_motion_tip(self, target_pose, label,
                                         eef_step=0.003, min_fraction=0.45,
                                         accept_error=None):
        """Move the final short side-grasp segment as a straight Cartesian line."""
        self.move_group.set_end_effector_link("right_hand")
        start_pose = self.move_group.get_current_pose().pose
        target_hand_pose = self._pose_for_move_group_link(target_pose)
        waypoints = [copy.deepcopy(start_pose), copy.deepcopy(target_hand_pose)]
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step,
            True,
        )
        rospy.loginfo(
            "  %s cartesian line fraction: %.1f%%",
            label, fraction * 100.0)
        if fraction < min_fraction or not plan.joint_trajectory.points:
            rospy.logerr(
                "  %s cartesian line failed: fraction %.1f%% < %.1f%%",
                label, fraction * 100.0, min_fraction * 100.0)
            return False

        ok = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        rospy.sleep(0.35)
        actual_tip_pose = self._current_motion_tip_pose()
        target = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        ], dtype=np.float64)
        actual = np.array([
            actual_tip_pose.position.x,
            actual_tip_pose.position.y,
            actual_tip_pose.position.z,
        ], dtype=np.float64)
        error = float(np.linalg.norm(target - actual))
        rospy.loginfo(
            "  %s cartesian result: ok=%s remaining=%.1fcm",
            label, ok, error * 100.0)
        if accept_error is not None and error > float(accept_error):
            rospy.logerr(
                "  %s cartesian stopped too far: %.1fcm > %.1fcm",
                label, error * 100.0, float(accept_error) * 100.0)
            return False
        return bool(ok)

    def _roll_wrist_joint(self, delta_rad):
        if abs(delta_rad) < 1e-6:
            return True
        try:
            names = list(self.move_group.get_active_joints())
            joints = list(self.move_group.get_current_joint_values())
        except Exception as exc:
            rospy.logerr("  wrist roll failed: cannot read joints: %s", exc)
            return False
        if "right_j6" not in names:
            rospy.logerr("  wrist roll failed: right_j6 not in active joints %s", names)
            return False

        idx = names.index("right_j6")
        target = list(joints)
        target[idx] = max(-5.20, min(5.20, target[idx] + float(delta_rad)))
        rospy.loginfo(
            "  Rolling wrist right_j6: %.1f deg -> %.1f deg",
            math.degrees(joints[idx]), math.degrees(target[idx]))

        self.move_group.set_max_velocity_scaling_factor(0.10)
        self.move_group.set_max_acceleration_scaling_factor(0.10)
        self.move_group.set_joint_value_target(dict(zip(names, target)))
        ok = self.move_group.go(wait=True)
        self.move_group.stop()
        rospy.sleep(0.5)
        if not ok:
            rospy.logerr("  wrist roll failed while executing right_j6 target.")
            return False
        return True

    def _reach_bottleneck_with_reachable_orientation(self, bottleneck_xyz, approach_axis):
        for idx, candidate in enumerate(self._side_orientation_candidates(approach_axis), start=1):
            pose = self._make_pose(bottleneck_xyz, candidate["quat"])
            rpy_deg = [math.degrees(v) for v in candidate["rpy"]]
            rospy.loginfo(
                "  Trying side bottleneck axis=%s orientation %02d source=%s: rpy_deg=[%.1f, %.1f, %.1f] open_axis=[%.2f, %.2f, %.2f] tip_axis=[%.2f, %.2f, %.2f]",
                approach_axis, idx, candidate.get("source", "?"),
                rpy_deg[0], rpy_deg[1], rpy_deg[2],
                candidate["open_axis"][0], candidate["open_axis"][1], candidate["open_axis"][2],
                candidate["tip_axis"][0], candidate["tip_axis"][1], candidate["tip_axis"][2])
            if self._move_tip_pose_target(
                    pose,
                    "side bottleneck %s orientation %02d" % (approach_axis, idx),
                    velocity=0.10,
                    acceleration=0.10):
                rospy.loginfo(
                    "  Selected side-grasp axis=%s orientation %02d source=%s: rpy_deg=[%.1f, %.1f, %.1f] open_axis=[%.2f, %.2f, %.2f] tip_axis=[%.2f, %.2f, %.2f]",
                    approach_axis, idx, candidate.get("source", "?"),
                    rpy_deg[0], rpy_deg[1], rpy_deg[2],
                    candidate["open_axis"][0], candidate["open_axis"][1], candidate["open_axis"][2],
                    candidate["tip_axis"][0], candidate["tip_axis"][1], candidate["tip_axis"][2])
                return candidate
        rospy.logerr("  No reachable side bottleneck orientation found.")
        return None

    def _find_reachable_side_plan(self, object_center):
        for axis in self._candidate_axes():
            d = self._axis_vector_for(axis)
            if self.motion_tip_name == "right_gripper_tip":
                side_offset = self.tip_grasp_offset
                rospy.loginfo(
                    "  Using gripper-tip side grasp target: tip offset along %s = %.3f",
                    axis, side_offset)
            else:
                side_offset = (
                    self._half_extent_along_approach_for_axis(axis)
                    + self.side_surface_margin
                    + self.flange_approach_offset)
            for grasp_z_offset in self._candidate_z_offsets():
                grasp_center = object_center + np.array([0.0, 0.0, grasp_z_offset])
                approach_center_xyz = grasp_center + d * side_offset
                grasp_xyz = np.array(approach_center_xyz, dtype=np.float64)
                if self.motion_tip_name == "right_gripper_tip":
                    if self.side_final_lateral_offset_source == "auto_object_width":
                        final_lateral_offset = self._auto_side_final_lateral_offset(axis)
                    else:
                        final_lateral_offset = self.side_final_lateral_offset
                    final_shift = (
                        -self._side_entry_lateral_vector(axis)
                        * final_lateral_offset)
                    grasp_xyz = grasp_xyz + final_shift
                    self.side_final_lateral_offset = final_lateral_offset
                    rospy.loginfo(
                        "  Applying final lateral grasp shift: [%.3f, %.3f, %.3f]",
                        final_shift[0], final_shift[1], final_shift[2])
                lift_xyz = grasp_xyz + np.array([0.0, 0.0, self.lift_height])
                if self.motion_tip_name == "right_gripper_tip":
                    rospy.loginfo(
                        "  Using gripper-tip side grasp height: object center z + %.3f",
                        grasp_z_offset)
                for bottleneck_z_offset in self._candidate_bottleneck_z_offsets():
                    bottleneck_center = object_center + np.array(
                        [0.0, 0.0, bottleneck_z_offset])
                    bottleneck_xyz = (
                        bottleneck_center
                        + d * (side_offset + self.pregrasp_distance))
                    rospy.loginfo(
                        "  Trying side axis=%s grasp_z_offset=%.3f bottleneck_z_offset=%.3f bottleneck=[%.3f, %.3f, %.3f] grasp=[%.3f, %.3f, %.3f]",
                        axis, grasp_z_offset, bottleneck_z_offset,
                        bottleneck_xyz[0], bottleneck_xyz[1], bottleneck_xyz[2],
                        grasp_xyz[0], grasp_xyz[1], grasp_xyz[2])
                    selected = self._reach_bottleneck_with_reachable_orientation(
                        bottleneck_xyz, axis)
                    if selected is not None:
                        self.approach_axis = axis
                        self.side_flange_z_offset = grasp_z_offset
                        self.gripper_roll, self.gripper_pitch, self.gripper_yaw = selected["rpy"]
                        self.gripper_yaw_deg = math.degrees(self.gripper_yaw)
                        return {
                            "axis": axis,
                            "z_offset": grasp_z_offset,
                            "bottleneck_z_offset": bottleneck_z_offset,
                            "quat": selected["quat"],
                            "bottleneck_xyz": bottleneck_xyz,
                            "approach_center_xyz": approach_center_xyz,
                            "grasp_xyz": grasp_xyz,
                            "lift_xyz": lift_xyz,
                        }
        return None

    def _execute_incremental_line_to(self, target_pose, label, step=0.015,
                                     max_error=0.025, max_steps=25,
                                     max_failed_steps=3,
                                     commit_remaining=None,
                                     axes=("x", "y", "z")):
        rospy.loginfo("  %s incremental line move", label)
        axes = set(axes)
        target = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        ], dtype=np.float64)

        failed_steps = 0
        for i in range(max_steps):
            current_pose = self.move_group.get_current_pose().pose
            current_pose = self._current_motion_tip_pose()
            current = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z,
            ], dtype=np.float64)
            delta = target - current
            if "x" not in axes:
                delta[0] = 0.0
            if "y" not in axes:
                delta[1] = 0.0
            if "z" not in axes:
                delta[2] = 0.0
            dist = float(np.linalg.norm(delta))
            if dist <= max_error:
                rospy.loginfo(
                    "  %s reached: error=%.1fcm", label, dist * 100.0)
                return True
            if commit_remaining is not None and dist <= commit_remaining:
                rospy.loginfo(
                    "  %s commit early: remaining=%.1fcm <= %.1fcm",
                    label, dist * 100.0, float(commit_remaining) * 100.0)
                return True
            direction = delta / max(1e-9, dist)
            next_xyz = current + direction * min(step, dist)
            next_pose = copy.deepcopy(target_pose)
            next_pose.position.x = float(next_xyz[0] if "x" in axes else current[0])
            next_pose.position.y = float(next_xyz[1] if "y" in axes else current[1])
            next_pose.position.z = float(next_xyz[2] if "z" in axes else current[2])
            ok = self._move_tip_pose_target(next_pose, "%s step %02d" % (label, i + 1),
                                            velocity=0.08, acceleration=0.08)
            rospy.loginfo(
                "  %s step %02d ok=%s remaining=%.1fcm",
                label, i + 1, ok, dist * 100.0)
            if not ok and dist > max_error:
                failed_steps += 1
                rospy.logwarn("  %s planner did not finish; retrying", label)
                if failed_steps >= max_failed_steps:
                    rospy.logerr(
                        "  %s aborted after %d failed planning steps to avoid wrist twisting",
                        label, failed_steps)
                    return False
            else:
                failed_steps = 0

        final_pose = self._current_motion_tip_pose()
        final = np.array([
            final_pose.position.x,
            final_pose.position.y,
            final_pose.position.z,
        ], dtype=np.float64)
        error_vec = target - final
        if "x" not in axes:
            error_vec[0] = 0.0
        if "y" not in axes:
            error_vec[1] = 0.0
        if "z" not in axes:
            error_vec[2] = 0.0
        error = float(np.linalg.norm(error_vec))
        rospy.logerr(
            "  %s stopped too far from target: %.1fcm > %.1fcm",
            label, error * 100.0, max_error * 100.0)
        return False

    def _side_entry_lateral_vector(self, approach_axis=None):
        axis = str(approach_axis or self.approach_axis).strip().lower()
        if axis in ["+y", "y+", "-y", "y-"]:
            return np.array([self.side_y_final_lateral_sign, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)

    def _replace_axis_projection(self, point, reference, axis_vec):
        axis_vec = axis_vec / max(1e-9, np.linalg.norm(axis_vec))
        delta = float(np.dot(reference - point, axis_vec))
        return point + axis_vec * delta

    def _execute_staged_side_approach(self, grasp_pose, approach_center_pose=None):
        if not self.staged_side_approach:
            return self._execute_incremental_line_to(grasp_pose, "side approach")

        approach_vec = self._axis_vector_for(self.approach_axis)
        mouth_mode = False
        desired_final_mouth = None
        if self.side_use_mouth_center and getattr(self, "planned_side_grasp_center", None) is not None:
            mouth_state = self._get_gripper_mouth_state()
            if mouth_state is not None:
                mouth_mode = True
                desired_final_mouth = np.array(
                    self.planned_side_grasp_center, dtype=np.float64)
                grasp_pose = self._command_pose_for_mouth_center(
                    grasp_pose, desired_final_mouth, mouth_state["offset"])
                precontact_mouth = (
                    desired_final_mouth
                    + approach_vec * (
                        self._half_extent_along_approach()
                        + self.side_surface_margin
                        + self.side_mouth_clearance
                        + self.pregrasp_distance))
                approach_center_pose = self._command_pose_for_mouth_center(
                    grasp_pose, precontact_mouth, mouth_state["offset"])
                rospy.loginfo(
                    "  Side mouth-center target rebuilt from finger TF: "
                    "mouth_offset=[%.3f, %.3f, %.3f] final_mouth=[%.3f, %.3f, %.3f] "
                    "safe_mouth=[%.3f, %.3f, %.3f]",
                    mouth_state["offset"][0], mouth_state["offset"][1], mouth_state["offset"][2],
                    desired_final_mouth[0], desired_final_mouth[1], desired_final_mouth[2],
                    precontact_mouth[0], precontact_mouth[1], precontact_mouth[2])
                self._log_mouth_alignment(
                    "side staged start", desired_final_mouth, require_ok=False)

        final_target = np.array([
            grasp_pose.position.x,
            grasp_pose.position.y,
            grasp_pose.position.z,
        ], dtype=np.float64)
        if approach_center_pose is None:
            approach_target = np.array(final_target, dtype=np.float64)
        else:
            approach_target = np.array([
                approach_center_pose.position.x,
                approach_center_pose.position.y,
                approach_center_pose.position.z,
            ], dtype=np.float64)
        current_pose = self._current_motion_tip_pose()
        current = np.array([
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        ], dtype=np.float64)

        if self._is_y_axis(self.approach_axis):
            if mouth_mode:
                entry_target = np.array(approach_target, dtype=np.float64)
            else:
                entry_target = final_target + approach_vec * self.pregrasp_distance
            entry_pre = np.array(entry_target, dtype=np.float64)
        else:
            lateral_vec = self._side_entry_lateral_vector()
            entry_target = approach_target + lateral_vec * self.side_entry_lateral_offset
            entry_pre = self._replace_axis_projection(
                entry_target, current, approach_vec)
        entry_safe_high = np.array(entry_pre, dtype=np.float64)
        entry_safe_high[2] = current[2]
        entry_pre[2] = approach_target[2]

        rospy.loginfo(
            "  staged side approach: entry_lateral_offset=%.3f final_lateral_offset=%.3f (%s) stage_c_commit_remaining=%.3f stage_c_cartesian=%s",
            self.side_entry_lateral_offset,
            self.side_final_lateral_offset,
            self.side_final_lateral_offset_source,
            self.stage_c_commit_remaining,
            self.stage_c_cartesian)
        rospy.loginfo(
            "    approach center target: [%.3f, %.3f, %.3f]",
            approach_target[0], approach_target[1], approach_target[2])
        rospy.loginfo(
            "    final grasp target:     [%.3f, %.3f, %.3f]",
            final_target[0], final_target[1], final_target[2])
        rospy.loginfo(
            "    stage A1 safe lateral high: [%.3f, %.3f, %.3f]",
            entry_safe_high[0], entry_safe_high[1], entry_safe_high[2])
        rospy.loginfo(
            "    stage A2 safe lateral low:  [%.3f, %.3f, %.3f]",
            entry_pre[0], entry_pre[1], entry_pre[2])
        rospy.loginfo(
            "    stage B pre-approach align: [%.3f, %.3f, %.3f]",
            entry_target[0], entry_target[1], entry_target[2])
        rospy.loginfo(
            "    stage C approach-axis close: [%.3f, %.3f, %.3f]",
            final_target[0], final_target[1], final_target[2])

        stage_a1 = copy.deepcopy(grasp_pose)
        stage_a1.position.x = float(entry_safe_high[0])
        stage_a1.position.y = float(entry_safe_high[1])
        stage_a1.position.z = float(entry_safe_high[2])
        if not self._execute_incremental_line_to(
                stage_a1, "side approach stage A1",
                step=0.008, max_error=0.018, max_steps=24,
                max_failed_steps=3):
            return False

        stage_a2 = copy.deepcopy(grasp_pose)
        stage_a2.position.x = float(entry_pre[0])
        stage_a2.position.y = float(entry_pre[1])
        stage_a2.position.z = float(entry_pre[2])
        if not self._execute_incremental_line_to(
                stage_a2, "side approach stage A2",
                step=0.008, max_error=0.012, max_steps=24,
                max_failed_steps=3):
            return False

        stage_b = copy.deepcopy(grasp_pose)
        stage_b.position.x = float(entry_target[0])
        stage_b.position.y = float(entry_target[1])
        stage_b.position.z = float(entry_target[2])
        if not self._execute_incremental_line_to(
                stage_b, "side approach stage B",
                step=0.008, max_error=0.012, max_steps=28,
                max_failed_steps=3):
            return False
        if mouth_mode and not self._log_mouth_alignment(
                "side before final approach", desired_final_mouth,
                require_ok=False):
            return False

        if self.stage_c_cartesian:
            accept_error = self.stage_c_commit_remaining
            if self._is_y_axis(self.approach_axis):
                accept_error = min(accept_error, 0.012)
            ok = self._execute_cartesian_to_motion_tip(
                grasp_pose, "side approach stage C",
                eef_step=0.003, min_fraction=0.45,
                accept_error=accept_error)
        else:
            ok = self._execute_incremental_line_to(
                grasp_pose, "side approach stage C",
                step=0.006, max_error=0.018, max_steps=12,
                max_failed_steps=2,
                commit_remaining=self.stage_c_commit_remaining)

        if ok and mouth_mode:
            return self._log_mouth_alignment(
                "side before gripper close", desired_final_mouth,
                require_ok=True)
        return ok

    def execute_and_record(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 Side-Grasp Demo Recording: %s", self.demo_name)
        rospy.loginfo("Object at: %s size=%s", self.object_pos, self.object_size)
        rospy.loginfo("Approach axis: %s yaw=%.1f deg", self.approach_axis, self.gripper_yaw_deg)
        rospy.loginfo(
            "Motion reference link: %s (side final lateral offset %.3f, %s)",
            self.motion_tip_name,
            self.side_final_lateral_offset,
            self.side_final_lateral_offset_source)
        if self._is_y_axis(self.approach_axis):
            rospy.loginfo(
                "Y-side mouth-center compensation: x sign=%+.0f",
                self.side_y_final_lateral_sign)
        rospy.loginfo(
            "Side grasp height fraction: %.2f",
            self.side_grasp_height_fraction)
        rospy.loginfo(
            "Side mouth-center control: %s clearance=%.1fcm lateral_tol=%.1fcm z_tol=%.1fcm",
            self.side_use_mouth_center,
            self.side_mouth_clearance * 100.0,
            self.side_mouth_align_tolerance * 100.0,
            self.side_mouth_z_tolerance * 100.0)
        rospy.loginfo("=" * 60)

        rospy.loginfo("[1/5] Moving to safe starting pose...")
        safe_joints = {
            "right_j0": 0.0, "right_j1": -0.8, "right_j2": 0.0,
            "right_j3": 1.8, "right_j4": 0.0, "right_j5": 0.0, "right_j6": 0.0
        }
        self.move_group.set_joint_value_target(safe_joints)
        self.move_group.go(wait=True)
        rospy.sleep(1.0)

        bx, by, bz = [float(v) for v in self.object_pos]
        sx, sy, sz = [float(v) for v in self.object_size]
        object_center = np.array([bx, by, bz + sz / 2.0], dtype=np.float64)
        side_grasp_center = np.array(
            [bx, by, bz + sz * self.side_grasp_height_fraction],
            dtype=np.float64)
        self.planned_object_center = object_center
        self.planned_side_grasp_center = side_grasp_center

        rospy.loginfo(
            "  Object visual center: [%.3f, %.3f, %.3f]",
            object_center[0], object_center[1], object_center[2])
        rospy.loginfo(
            "  Side grasp reference: [%.3f, %.3f, %.3f] = object_z %.3f + height %.3f * %.2f",
            side_grasp_center[0], side_grasp_center[1], side_grasp_center[2],
            bz, sz, self.side_grasp_height_fraction)

        rospy.loginfo("[2/5] Moving to side bottleneck pose...")
        plan = self._find_reachable_side_plan(side_grasp_center)
        if plan is None:
            rospy.logerr("  No reachable side approach axis/height/orientation found.")
            return False
        q = plan["quat"]
        bottleneck_xyz = plan["bottleneck_xyz"]
        approach_center_xyz = plan.get("approach_center_xyz", plan["grasp_xyz"])
        grasp_xyz = plan["grasp_xyz"]
        lift_xyz = plan["lift_xyz"]
        rospy.loginfo(
            "  Selected side approach: axis=%s z_offset=%.3f",
            self.approach_axis, self.side_flange_z_offset)
        if "bottleneck_z_offset" in plan:
            rospy.loginfo(
                "  Selected side bottleneck z_offset: %.3f",
                plan["bottleneck_z_offset"])
        rospy.loginfo(
            "  Selected side bottleneck xyz: [%.3f, %.3f, %.3f]",
            bottleneck_xyz[0], bottleneck_xyz[1], bottleneck_xyz[2])
        rospy.loginfo(
            "  Selected side grasp xyz:      [%.3f, %.3f, %.3f]",
            grasp_xyz[0], grasp_xyz[1], grasp_xyz[2])

        if self.reach_then_roll:
            rospy.loginfo(
                "  Rolling wrist after side bottleneck: %.1f deg",
                math.degrees(self.post_bottleneck_wrist_roll))
            if not self._roll_wrist_joint(self.post_bottleneck_wrist_roll):
                return False
            current_tip_pose = self._current_motion_tip_pose()
            q = [
                current_tip_pose.orientation.x,
                current_tip_pose.orientation.y,
                current_tip_pose.orientation.z,
                current_tip_pose.orientation.w,
            ]
            rospy.loginfo(
                "  Side grasp orientation updated from current tip after wrist roll.")

        bottleneck_pose = self._make_pose(bottleneck_xyz, q)
        approach_center_pose = self._make_pose(approach_center_xyz, q)
        grasp_pose = self._make_pose(grasp_xyz, q)
        lift_pose = self._make_pose(lift_xyz, q)
        self.planned_side_grasp_pose = copy.deepcopy(grasp_pose)
        self.planned_side_approach_center_pose = copy.deepcopy(approach_center_pose)

        rospy.loginfo("  Opening gripper before side approach...")
        self._open_gripper()

        rospy.loginfo("[3/5] Capturing bottleneck RGB-D observation...")
        self._capture_bottleneck(timeout=3.0)
        bottleneck_ee = self._get_end_effector_pose()
        if bottleneck_ee is None:
            rospy.logerr("Failed to get bottleneck end-effector pose.")
            return False

        rospy.loginfo("[4/5] Executing side grasp and recording...")
        self.recording = True
        self.recorded_poses = []
        first_pose = self._stamp_gripper_state(self._get_end_effector_pose())
        if first_pose:
            self.recorded_poses.append(first_pose)

        record_thread = threading.Thread(
            target=self._continuous_record, args=(10.0,))
        record_thread.start()

        if not self._execute_staged_side_approach(grasp_pose, approach_center_pose):
            self.recording = False
            record_thread.join(timeout=2.0)
            return False

        rospy.loginfo("  Closing gripper for side-grasp recording...")
        self._close_gripper()

        if not self._execute_incremental_line_to(lift_pose, "side lift",
                                                 step=0.010, max_error=0.035,
                                                 max_steps=10,
                                                 max_failed_steps=2):
            self.recording = False
            record_thread.join(timeout=2.0)
            return False

        self.recording = False
        record_thread.join(timeout=2.0)
        rospy.loginfo("  Recording stopped. Total poses: %d", len(self.recorded_poses))

        self.move_group.set_max_velocity_scaling_factor(0.6)
        self.move_group.set_max_acceleration_scaling_factor(0.6)

        rospy.loginfo("[5/5] Saving side-grasp demo to recorded library...")
        self._save_demo(bottleneck_ee)
        self._patch_recorded_json()

        rospy.loginfo("=" * 60)
        rospy.loginfo("Side-grasp demo '%s' recorded successfully.", self.demo_name)
        rospy.loginfo("=" * 60)
        return True

    def _patch_recorded_json(self):
        json_path = os.path.join(OUTPUT_DIR, "%s.json" % self.demo_name)
        with open(json_path, "r", encoding="utf-8") as f:
            demo = json.load(f)

        shape = str(rospy.get_param("~object_shape", "cylinder"))
        label = str(rospy.get_param("~object_label", "green_tall_object"))
        demo["language_description"] = (
            "Pick up the green tall object from the side")
        demo["language_tags"] = [
            "grasp",
            "pick up",
            "side grasp",
            "lateral grasp",
            "green tall object",
            "green cylinder",
            "green bottle",
            "tall cylinder",
            "sideways grasp",
            "抓取",
            "侧边抓取",
            "侧面抓取",
            "绿色圆柱",
            "高物体",
        ]
        demo["object_info"]["category"] = shape
        demo["object_info"]["label"] = label
        demo["object_info"]["color"] = "green"
        demo["grasp_strategy"] = "side_grasp"
        demo["approach_axis"] = self.approach_axis
        demo["approach_direction"] = self._axis_vector().tolist()
        demo["retract_direction"] = [0.0, 0.0, 1.0]
        demo["side_grasp_height_fraction"] = float(
            self.side_grasp_height_fraction)
        demo["side_motion_reference_link"] = self.motion_tip_name
        demo["side_reference_convention"] = (
            "right_hand gripper-mouth-center convention"
            if self.motion_tip_name != "right_gripper_tip"
            else "right_gripper_tip fingertip convention")
        demo["side_final_lateral_offset_source"] = (
            self.side_final_lateral_offset_source)
        if getattr(self, "planned_side_grasp_pose", None) is not None:
            p = self.planned_side_grasp_pose
            demo["side_grasp_pose_base_frame"] = {
                "position_m": {
                    "x": float(p.position.x),
                    "y": float(p.position.y),
                    "z": float(p.position.z),
                },
                "orientation_xyzw": {
                    "x": float(p.orientation.x),
                    "y": float(p.orientation.y),
                    "z": float(p.orientation.z),
                    "w": float(p.orientation.w),
                },
            }
            demo["side_grasp_orientation_xyzw"] = [
                float(p.orientation.x),
                float(p.orientation.y),
                float(p.orientation.z),
                float(p.orientation.w),
            ]
            demo["side_final_lateral_offset"] = float(self.side_final_lateral_offset)
        if getattr(self, "planned_side_approach_center_pose", None) is not None:
            p = self.planned_side_approach_center_pose
            demo["side_approach_center_pose_base_frame"] = {
                "position_m": {
                    "x": float(p.position.x),
                    "y": float(p.position.y),
                    "z": float(p.position.z),
                },
                "orientation_xyzw": {
                    "x": float(p.orientation.x),
                    "y": float(p.orientation.y),
                    "z": float(p.orientation.z),
                    "w": float(p.orientation.w),
                },
            }
        demo["gripper_side_orientation_rpy"] = [
            self.gripper_roll, self.gripper_pitch, self.gripper_yaw]
        demo["notes"] = (
            "Recorded side grasp demo. The bottleneck is beside the object; "
            "the interaction phase approaches horizontally at object middle "
            "height, closes the gripper, then lifts upward.")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(demo, f, indent=2, ensure_ascii=False)
        rospy.loginfo("  Patched side-grasp metadata in %s", json_path)

    def _load_langsam_mask(self):
        if not self.mask_path:
            return None
        if not os.path.exists(self.mask_path):
            rospy.logwarn("  LangSAM mask not found: %s", self.mask_path)
            return None
        try:
            mask = np.load(self.mask_path).astype(bool)
            if self.bottleneck_rgb is not None and mask.shape != self.bottleneck_rgb.shape[:2]:
                rospy.logwarn(
                    "  LangSAM mask shape %s does not match RGB shape %s",
                    mask.shape, self.bottleneck_rgb.shape[:2])
                return None
            rospy.loginfo(
                "  Using LangSAM mask for side demo scene package: %s pixels=%d",
                self.mask_path, int(np.count_nonzero(mask)))
            return mask
        except Exception as exc:
            rospy.logwarn("  Failed to load LangSAM mask %s: %s", self.mask_path, exc)
            return None

    def _save_scene_package(self, bottleneck_ee):
        if self.bottleneck_rgb is None or self.bottleneck_depth is None:
            rospy.logwarn("  Scene package skipped: missing bottleneck RGB-D")
            return
        import cv2
        rgb = cv2.cvtColor(self.bottleneck_rgb, cv2.COLOR_BGR2RGB)
        segmap = self._load_langsam_mask()
        if segmap is None:
            segmap = self._green_mask_from_bgr(self.bottleneck_rgb)
            rospy.logwarn(
                "  Falling back to HSV mask for side demo scene package: pixels=%d",
                int(np.count_nonzero(segmap)) if segmap is not None else 0)

        shape = str(rospy.get_param("~object_shape", "cylinder"))
        label = str(rospy.get_param("~object_label", "green_tall_object"))
        scene_data = {
            "rgb": rgb,
            "depth": self.bottleneck_depth,
            "segmap": segmap,
            "intrinsics": np.array([
                [407.391526, 0.0, 640.5],
                [0.0, 407.391526, 400.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64),
            "pose": {
                "position": bottleneck_ee["position"],
                "orientation": bottleneck_ee["orientation"],
                "method": "recorded_side_bottleneck_pose",
                "confidence": 1.0,
            },
        }
        package_root = os.path.join(os.path.dirname(OUTPUT_DIR), "scene_packages")
        package = save_scene_package(
            scene_data,
            package_root,
            name="demo_%s" % self.demo_name,
            role="recorded_demo",
            extra_metadata={
                "demo_id": self.demo_name,
                "object_position_base": self.object_pos,
                "object_size": self.object_size,
                "object_shape": shape,
                "object_label": label,
                "grasp_strategy": "side_grasp",
                "approach_axis": self.approach_axis,
                "motion_reference_link": self.motion_tip_name,
                "side_reference_convention": (
                    "right_hand gripper-mouth-center convention"
                    if self.motion_tip_name != "right_gripper_tip"
                    else "right_gripper_tip fingertip convention"),
                "mask_source": self.mask_path or "hsv_fallback",
            })
        rospy.loginfo(
            "  Scene package saved to %s (points=%d, mask_px=%d)",
            package["package_dir"],
            package["stats"]["pointcloud_points"],
            package["stats"]["segmap_pixels"])


if __name__ == "__main__":
    rospy.init_node("mt3_record_side_grasp_demo", anonymous=True)

    obj_x = rospy.get_param("~object_x", 0.60)
    obj_y = rospy.get_param("~object_y", 0.00)
    obj_z = rospy.get_param("~object_z", -0.58)
    obj_size = rospy.get_param("~object_size", [0.045, 0.045, 0.10])
    demo_name = rospy.get_param(
        "~demo_name", "tall_object_green_side_grasp_v1")
    approach_axis = rospy.get_param("~approach_axis", "-y")
    mask_path = rospy.get_param(
        "~mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")
    gripper_roll = rospy.get_param("~gripper_roll", math.pi / 2.0)
    gripper_pitch = rospy.get_param("~gripper_pitch", 0.0)
    gripper_yaw_deg = rospy.get_param("~gripper_yaw_deg", None)
    pregrasp_distance = rospy.get_param("~pregrasp_distance", 0.12)
    side_surface_margin = rospy.get_param("~side_surface_margin", 0.005)
    flange_approach_offset = rospy.get_param("~flange_approach_offset", 0.040)
    side_flange_z_offset = rospy.get_param("~side_flange_z_offset", 0.040)
    lift_height = rospy.get_param("~lift_height", 0.04)
    auto_try_axes = rospy.get_param("~auto_try_axes", False)
    use_ik_motion = rospy.get_param("~use_ik_motion", True)
    motion_tip_name = rospy.get_param("~motion_tip_name", "right_gripper_tip")
    tip_grasp_offset = rospy.get_param("~tip_grasp_offset", 0.0)
    ik_service_mode = rospy.get_param("~ik_service_mode", "moveit")
    right_hand_to_tip_z = rospy.get_param("~right_hand_to_tip_z", 0.13562)
    reach_then_roll = rospy.get_param("~reach_then_roll", True)
    post_bottleneck_wrist_roll_deg = rospy.get_param(
        "~post_bottleneck_wrist_roll_deg", 90.0)
    staged_side_approach = rospy.get_param("~staged_side_approach", True)
    side_entry_lateral_offset = rospy.get_param(
        "~side_entry_lateral_offset", 0.055)
    side_final_lateral_offset_param = rospy.get_param(
        "~side_final_lateral_offset", "auto")
    side_y_final_lateral_sign = rospy.get_param(
        "~side_y_final_lateral_sign", 1.0)
    stage_c_commit_remaining = rospy.get_param(
        "~stage_c_commit_remaining", 0.045)
    stage_c_cartesian = rospy.get_param("~stage_c_cartesian", True)
    side_grasp_height_fraction = rospy.get_param(
        "~side_grasp_height_fraction", 0.65)
    side_use_mouth_center = rospy.get_param("~side_use_mouth_center", True)
    side_mouth_clearance = rospy.get_param("~side_mouth_clearance", 0.006)
    side_mouth_align_tolerance = rospy.get_param(
        "~side_mouth_align_tolerance", 0.020)
    side_mouth_z_tolerance = rospy.get_param(
        "~side_mouth_z_tolerance", 0.035)
    if str(side_final_lateral_offset_param).strip().lower() in ["", "auto", "none"]:
        side_final_lateral_offset = None
    else:
        side_final_lateral_offset = float(side_final_lateral_offset_param)

    recorder = SideGraspDemoRecorder(
        obj_x, obj_y, obj_z, obj_size, demo_name,
        approach_axis=approach_axis,
        gripper_roll=gripper_roll,
        gripper_pitch=gripper_pitch,
        gripper_yaw_deg=gripper_yaw_deg,
        mask_path=mask_path,
        pregrasp_distance=pregrasp_distance,
        side_surface_margin=side_surface_margin,
        flange_approach_offset=flange_approach_offset,
        side_flange_z_offset=side_flange_z_offset,
        lift_height=lift_height,
        auto_try_axes=auto_try_axes,
        use_ik_motion=use_ik_motion,
        motion_tip_name=motion_tip_name,
        tip_grasp_offset=tip_grasp_offset,
        ik_service_mode=ik_service_mode,
        right_hand_to_tip_z=right_hand_to_tip_z,
        reach_then_roll=reach_then_roll,
        post_bottleneck_wrist_roll_deg=post_bottleneck_wrist_roll_deg,
        staged_side_approach=staged_side_approach,
        side_entry_lateral_offset=side_entry_lateral_offset,
        side_final_lateral_offset=side_final_lateral_offset,
        side_y_final_lateral_sign=side_y_final_lateral_sign,
        stage_c_commit_remaining=stage_c_commit_remaining,
        stage_c_cartesian=stage_c_cartesian,
        side_grasp_height_fraction=side_grasp_height_fraction,
        side_use_mouth_center=side_use_mouth_center,
        side_mouth_clearance=side_mouth_clearance,
        side_mouth_align_tolerance=side_mouth_align_tolerance,
        side_mouth_z_tolerance=side_mouth_z_tolerance)

    try:
        recorder.execute_and_record()
    except rospy.ROSInterruptException:
        rospy.loginfo("Recording interrupted.")
    except Exception as e:
        rospy.logerr("Recording failed: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        moveit_commander.roscpp_shutdown()
