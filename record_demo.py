#!/usr/bin/env python3
"""
MT3 Top-Grasp Demo Recording (Fixed)
基于 auto_grasp_final.py 验证过的笛卡尔下降逻辑
用法: python3 record_demo.py _object_x:=0.60 _object_y:=0.0 _object_z:=-0.58
"""

import rospy, sys, os, time, json, math, threading, copy
import numpy as np
import tf2_ros, geometry_msgs.msg, moveit_commander
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from mt3_scene_package import save_scene_package

ROS_NAMESPACE = "/robot"
PLANNING_GROUP = "right_arm"
END_EFFECTOR_LINK = "right_hand"
RECORD_RATE = 30

JOINT_LIMITS = {
    'right_j0': (-3.05, 3.05), 'right_j1': (-1.92, 1.396),
    'right_j2': (-3.05, 3.05), 'right_j3': (-3.05, 3.05),
    'right_j4': (-3.05, 3.05), 'right_j5': (-3.05, 3.05),
    'right_j6': (-5.23, 5.23),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "demo_library", "recorded")

CART_STEP = 0.004
FLANGE_Z_OFFSET = 0.040
DEFAULT_LEFT_FINGER_TIP_FRAME = "right_gripper_l_finger_tip"
DEFAULT_RIGHT_FINGER_TIP_FRAME = "right_gripper_r_finger_tip"


class DemoRecorder:
    def __init__(self, object_x, object_y, object_z, object_size, demo_name):
        self.object_pos = [object_x, object_y, object_z]
        self.object_size = object_size
        self.demo_name = demo_name
        self.recording = False
        self.recorded_poses = []
        self.gripper_command_state = None
        self.bottleneck_rgb = None
        self.bottleneck_depth = None
        self.mouth_center_calibration = {}
        self.top_grasp_centering_diagnostics = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander(
            robot_description=f"{ROS_NAMESPACE}/robot_description", ns=ROS_NAMESPACE)
        self.move_group = moveit_commander.MoveGroupCommander(
            PLANNING_GROUP,
            robot_description=f"{ROS_NAMESPACE}/robot_description", ns=ROS_NAMESPACE)
        self.move_group.set_end_effector_link(END_EFFECTOR_LINK)
        self.move_group.set_pose_reference_frame("base")

    def _capture_bottleneck(self, timeout=5.0):
        rospy.loginfo("等待相机图像...")
        try:
            rgb_msg = rospy.wait_for_message(
                "/io/internal_camera/head_camera/image_raw", Image, timeout=timeout)
            depth_msg = rospy.wait_for_message(
                "/io/internal_camera/head_camera/depth/image_raw", Image, timeout=timeout)
            self.bottleneck_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self.bottleneck_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            rospy.loginfo(f"Bottleneck captured: RGB {self.bottleneck_rgb.shape}, "
                          f"Depth {self.bottleneck_depth.shape}")
            return True
        except rospy.ROSException as e:
            rospy.logwarn(f"Camera timeout: {e}")
            return False

    def _get_end_effector_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                "base", "right_hand", rospy.Time(0), rospy.Duration(1.0))
            pos = tf.transform.translation
            ori = tf.transform.rotation
            return {
                "position": [pos.x, pos.y, pos.z],
                "orientation": [ori.x, ori.y, ori.z, ori.w],
                "timestamp": tf.header.stamp.to_sec()
            }
        except Exception:
            return None

    def _lookup_frame_point(self, frame, timeout=0.8):
        try:
            tf = self.tf_buffer.lookup_transform(
                "base", frame, rospy.Time(0), rospy.Duration(timeout))
            pos = tf.transform.translation
            return [float(pos.x), float(pos.y), float(pos.z)]
        except Exception as exc:
            rospy.logwarn("TF lookup failed for %s: %s", frame, exc)
            return None

    def _get_gripper_mouth_state(self):
        left_frame = rospy.get_param(
            "~left_finger_tip_frame", DEFAULT_LEFT_FINGER_TIP_FRAME)
        right_frame = rospy.get_param(
            "~right_finger_tip_frame", DEFAULT_RIGHT_FINGER_TIP_FRAME)
        left = self._lookup_frame_point(left_frame)
        right = self._lookup_frame_point(right_frame)
        ee = self._get_end_effector_pose()
        if ee is None:
            return {"available": False}
        hand = ee["position"]
        if left is None or right is None:
            return {
                "available": False,
                "hand": hand,
                "center": hand,
                "offset": [0.0, 0.0, 0.0],
                "opening": 0.0,
            }
        center = [
            0.5 * (left[0] + right[0]),
            0.5 * (left[1] + right[1]),
            0.5 * (left[2] + right[2]),
        ]
        opening = math.sqrt(
            (left[0] - right[0]) ** 2 +
            (left[1] - right[1]) ** 2 +
            (left[2] - right[2]) ** 2)
        return {
            "available": True,
            "hand": hand,
            "left": left,
            "right": right,
            "center": center,
            "offset": [
                center[0] - hand[0],
                center[1] - hand[1],
                center[2] - hand[2],
            ],
            "opening": opening,
        }

    def _compute_top_mouth_xy_offset(self, desired_xy, label):
        state = self._get_gripper_mouth_state()
        if not state.get("available"):
            rospy.logwarn(
                "%s mouth-center TF unavailable; recording will use right_hand XY",
                label)
            self.mouth_center_calibration = {
                "available": False,
                "label": label,
                "reason": "finger_tip_tf_unavailable",
            }
            return [0.0, 0.0]

        offset = state["offset"]
        max_abs = float(rospy.get_param("~top_mouth_xy_correction_max", 0.045))
        raw_dx = float(offset[0])
        raw_dy = float(offset[1])
        dx = max(-max_abs, min(max_abs, raw_dx))
        dy = max(-max_abs, min(max_abs, raw_dy))
        center = state["center"]
        hand = state["hand"]
        err_x = center[0] - float(desired_xy[0])
        err_y = center[1] - float(desired_xy[1])
        err_norm = math.sqrt(err_x * err_x + err_y * err_y)
        center_tol = float(rospy.get_param("~top_grasp_centering_warn_threshold_m", 0.003))
        rospy.loginfo(
            "%s mouth-center calibration: hand=[%.3f, %.3f, %.3f] "
            "mouth=[%.3f, %.3f, %.3f] desired_xy=[%.3f, %.3f] "
            "mouth_err_xy=[%.1f, %.1f]mm norm=%.1fmm offset_xy=[%.1f, %.1f]mm",
            label, hand[0], hand[1], hand[2],
            center[0], center[1], center[2],
            float(desired_xy[0]), float(desired_xy[1]),
            err_x * 1000.0, err_y * 1000.0, err_norm * 1000.0,
            dx * 1000.0, dy * 1000.0)
        self.mouth_center_calibration = {
            "available": True,
            "label": label,
            "hand_xyz": hand,
            "mouth_center_xyz": center,
            "mouth_offset_xyz": state["offset"],
            "used_mouth_offset_xy": [dx, dy],
            "mouth_opening_m": state.get("opening", 0.0),
            "desired_xy": [float(desired_xy[0]), float(desired_xy[1])],
            "mouth_center_xy_error_m": [float(err_x), float(err_y)],
            "mouth_center_xy_error_norm_m": float(err_norm),
            "centering_warn_threshold_m": float(center_tol),
            "centering_passed": bool(err_norm <= center_tol),
        }
        return [dx, dy]

    def _build_top_grasp_centering_diagnostic(self, desired_xy, label, state=None, tcp_pose=None):
        state = state or self._get_gripper_mouth_state()
        desired = [float(desired_xy[0]), float(desired_xy[1])]
        if not state.get("available"):
            return {
                "available": False,
                "label": label,
                "reason": state.get("reason", "mouth_center_tf_unavailable"),
                "object_xy": desired,
            }
        center = state["center"]
        hand = state["hand"]
        if tcp_pose is not None and tcp_pose.get("position"):
            hand = [float(v) for v in tcp_pose["position"][:3]]
        mouth_dx = float(center[0]) - desired[0]
        mouth_dy = float(center[1]) - desired[1]
        tcp_dx = float(hand[0]) - desired[0]
        tcp_dy = float(hand[1]) - desired[1]
        mouth_norm = math.sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy)
        tcp_norm = math.sqrt(tcp_dx * tcp_dx + tcp_dy * tcp_dy)
        threshold = float(rospy.get_param("~top_grasp_centering_warn_threshold_m", 0.003))
        return {
            "available": True,
            "label": label,
            "object_xy": desired,
            "tcp_xyz": [float(v) for v in hand[:3]],
            "mouth_center_xyz": [float(v) for v in center[:3]],
            "tcp_object_offset_xy_m": [float(tcp_dx), float(tcp_dy)],
            "tcp_object_offset_xy_norm_m": float(tcp_norm),
            "mouth_object_offset_xy_m": [float(mouth_dx), float(mouth_dy)],
            "mouth_object_offset_xy_norm_m": float(mouth_norm),
            "centering_warn_threshold_m": float(threshold),
            "centering_passed": bool(mouth_norm <= threshold),
        }

    def _log_top_mouth_alignment(self, desired_xy, label, tcp_pose=None, store_key=None):
        diag = self._build_top_grasp_centering_diagnostic(
            desired_xy, label, tcp_pose=tcp_pose)
        if store_key:
            self.top_grasp_centering_diagnostics[store_key] = diag
        if not diag.get("available"):
            rospy.logwarn("%s mouth-center TF unavailable: %s",
                          label, diag.get("reason", "unknown"))
            return None
        rospy.loginfo(
            "%s CENTERING DEBUG: object_xy=[%.4f, %.4f] tcp_xy=[%.4f, %.4f] "
            "mouth_xy=[%.4f, %.4f] tcp_obj=[%.1f, %.1f]mm |tcp|=%.1fmm "
            "mouth_obj=[%.1f, %.1f]mm |mouth|=%.1fmm threshold=%.1fmm result=%s",
            label,
            diag["object_xy"][0], diag["object_xy"][1],
            diag["tcp_xyz"][0], diag["tcp_xyz"][1],
            diag["mouth_center_xyz"][0], diag["mouth_center_xyz"][1],
            diag["tcp_object_offset_xy_m"][0] * 1000.0,
            diag["tcp_object_offset_xy_m"][1] * 1000.0,
            diag["tcp_object_offset_xy_norm_m"] * 1000.0,
            diag["mouth_object_offset_xy_m"][0] * 1000.0,
            diag["mouth_object_offset_xy_m"][1] * 1000.0,
            diag["mouth_object_offset_xy_norm_m"] * 1000.0,
            diag["centering_warn_threshold_m"] * 1000.0,
            "PASS" if diag["centering_passed"] else "WARN")
        if not diag["centering_passed"]:
            rospy.logwarn(
                "%s demo is not a center top grasp: mouth-object XY norm %.1f mm > %.1f mm",
                label,
                diag["mouth_object_offset_xy_norm_m"] * 1000.0,
                diag["centering_warn_threshold_m"] * 1000.0)
        return diag

    def _correct_top_mouth_xy_at_current_z(self, desired_xy, label):
        """Small final XY correction so the real gripper mouth, not right_hand, is centered."""
        if not rospy.get_param("~use_mouth_center_top_grasp", True):
            return False
        tol = float(rospy.get_param("~top_mouth_xy_tolerance", 0.003))
        max_step = float(rospy.get_param("~top_mouth_xy_final_max_step", 0.018))
        for attempt in range(3):
            state = self._get_gripper_mouth_state()
            if not state.get("available"):
                rospy.logwarn("%s mouth-center final correction unavailable", label)
                return False
            center = state["center"]
            err_x = center[0] - float(desired_xy[0])
            err_y = center[1] - float(desired_xy[1])
            err = math.sqrt(err_x * err_x + err_y * err_y)
            rospy.loginfo(
                "%s correction %d: mouth_err_xy=[%.1f, %.1f]cm",
                label, attempt + 1, err_x * 100.0, err_y * 100.0)
            if abs(err_x) <= tol and abs(err_y) <= tol:
                self.mouth_center_calibration["final_correction"] = {
                    "applied": attempt > 0,
                    "final_mouth_center_xyz": center,
                    "final_mouth_center_xy_error_m": [float(err_x), float(err_y)],
                }
                return True

            scale = min(1.0, max_step / max(err, 1e-6))
            current = self.move_group.get_current_pose().pose
            target = copy.deepcopy(current)
            target.position.x = current.position.x - err_x * scale
            target.position.y = current.position.y - err_y * scale
            plan, fraction = self.move_group.compute_cartesian_path(
                [copy.deepcopy(current), copy.deepcopy(target)], 0.003, True)
            rospy.loginfo(
                "%s correction %d cartesian fraction: %.1f%% target_hand_xy=[%.3f, %.3f]",
                label, attempt + 1, fraction * 100.0,
                target.position.x, target.position.y)
            if fraction >= 0.90 and len(plan.joint_trajectory.points) > 0:
                self.move_group.execute(plan, wait=True)
                rospy.sleep(0.25)
            else:
                rospy.logwarn(
                    "%s correction %d skipped: insufficient Cartesian path %.1f%%",
                    label, attempt + 1, fraction * 100.0)
                break

        state = self._get_gripper_mouth_state()
        if state.get("available"):
            center = state["center"]
            err_x = center[0] - float(desired_xy[0])
            err_y = center[1] - float(desired_xy[1])
            self.mouth_center_calibration["final_correction"] = {
                "applied": True,
                "final_mouth_center_xyz": center,
                "final_mouth_center_xy_error_m": [float(err_x), float(err_y)],
            }
        return False

    def _get_gripper_position(self):
        gripper = getattr(self, "gripper", None)
        if gripper is None:
            return None
        try:
            return float(gripper.get_position())
        except Exception:
            return None

    def _stamp_gripper_state(self, pose):
        if pose is None:
            return None
        pose["gripper_position"] = self._get_gripper_position()
        pose["gripper_state"] = self.gripper_command_state
        pose["gripper_next"] = self.gripper_command_state
        return pose

    def _continuous_record(self, duration_s):
        rate = rospy.Rate(RECORD_RATE)
        start = rospy.get_time()
        while self.recording and (rospy.get_time() - start) < duration_s:
            pose = self._get_end_effector_pose()
            if pose:
                self.recorded_poses.append(self._stamp_gripper_state(pose))
            rate.sleep()

    def _rotate_by_quat(self, q, v):
        x, y, z, w = q[0], q[1], q[2], q[3]
        vx, vy, vz = v[0], v[1], v[2]
        rx = (1-2*y*y-2*z*z)*vx + (2*x*y-2*w*z)*vy + (2*x*z+2*w*y)*vz
        ry = (2*x*y+2*w*z)*vx + (1-2*x*x-2*z*z)*vy + (2*y*z-2*w*x)*vz
        rz = (2*x*z-2*w*y)*vx + (2*y*z+2*w*x)*vy + (1-2*x*x-2*y*y)*vz
        return [rx, ry, rz]

    def _quat_delta_to_angular_velocity(self, q0, q1, dt):
        if dt <= 0:
            return [0.0, 0.0, 0.0]
        dq = [q0[0]*q1[3]+q0[3]*q1[0]+q0[1]*q1[2]-q0[2]*q1[1],
              q0[3]*q1[1]-q0[0]*q1[2]+q0[1]*q1[3]+q0[2]*q1[0],
              q0[3]*q1[2]+q0[0]*q1[1]-q0[1]*q1[0]+q0[2]*q1[3],
              q0[3]*q1[3]-q0[0]*q1[0]-q0[1]*q1[1]-q0[2]*q1[2]]
        norm = math.sqrt(sum(float(v)*float(v) for v in dq))
        if norm <= 1e-8:
            return [0.0, 0.0, 0.0]
        dq = [float(v)/norm for v in dq]
        if dq[3] < 0:
            dq = [-dq[0], -dq[1], -dq[2], -dq[3]]
        axis_norm = math.sqrt(dq[0]*dq[0]+dq[1]*dq[1]+dq[2]*dq[2])
        if axis_norm <= 1e-8:
            return [0.0, 0.0, 0.0]
        angle = 2.0*math.atan2(axis_norm, max(-1.0, min(1.0, dq[3])))
        axis = [dq[0]/axis_norm, dq[1]/axis_norm, dq[2]/axis_norm]
        return [float(axis[j]*angle/dt) for j in range(3)]

    def _poses_to_velocities(self, poses):
        velocities = []
        for i in range(1, len(poses)):
            dt = poses[i]["timestamp"] - poses[i-1]["timestamp"]
            if dt <= 0:
                continue
            p0 = np.array(poses[i-1]["position"])
            p1 = np.array(poses[i]["position"])
            dp_world = (p1 - p0) / dt
            q = poses[i-1]["orientation"]
            q_conj = [-q[0], -q[1], -q[2], q[3]]
            v_ee = self._rotate_by_quat(q_conj, dp_world.tolist())
            q0 = poses[i-1]["orientation"]
            q1 = poses[i]["orientation"]
            w_ee = self._quat_delta_to_angular_velocity(q0, q1, dt)
            w_world = self._rotate_by_quat(q0, w_ee)
            gripper_next = poses[i].get("gripper_next",
                           poses[i].get("gripper_state",
                           poses[i-1].get("gripper_state")))
            velocities.append({
                "timestamp": poses[i]["timestamp"], "dt": float(dt),
                "position": poses[i]["position"],
                "orientation": poses[i]["orientation"],
                "linear_ee": v_ee, "linear_world": dp_world.tolist(),
                "angular_ee": w_ee, "angular_world": w_world,
                "gripper_position": poses[i].get("gripper_position"),
                "gripper_state": poses[i].get("gripper_state"),
                "gripper_next": gripper_next,
            })
        return velocities

    # ════════════════════════════════════════════════════════
    # 笛卡尔下降（XY对齐 + Z垂直降）— 来自 auto_grasp_final.py
    # ════════════════════════════════════════════════════════
    def _cartesian_descend(self, target_pose, label):
        """笛卡尔直线下降：先 XY 对齐再 Z 降"""
        # Step 1: Cartesian XY align
        current = self.move_group.get_current_pose().pose
        xy_target = copy.deepcopy(target_pose)
        xy_target.position.z = current.position.z
        wp_xy = [copy.deepcopy(current), copy.deepcopy(xy_target)]
        plan_xy, frac_xy = self.move_group.compute_cartesian_path(wp_xy, CART_STEP, True)
        rospy.loginfo("  %s XY fraction: %.1f%%", label, frac_xy*100)
        if frac_xy >= 0.9 and len(plan_xy.joint_trajectory.points) > 0:
            self.move_group.execute(plan_xy, wait=True)
            rospy.sleep(0.3)

        # Step 2: Cartesian Z descent
        current = self.move_group.get_current_pose().pose
        z_target = copy.deepcopy(target_pose)
        z_target.position.x = current.position.x
        z_target.position.y = current.position.y
        wp_z = [copy.deepcopy(current), copy.deepcopy(z_target)]
        plan_z, frac_z = self.move_group.compute_cartesian_path(wp_z, CART_STEP, True)
        rospy.loginfo("  %s Z fraction: %.1f%%", label, frac_z*100)
        if frac_z >= 0.9 and len(plan_z.joint_trajectory.points) > 0:
            self.move_group.execute(plan_z, wait=True)
            rospy.sleep(0.3)
            return True
        # Fallback: small step descent
        rospy.logwarn("  %s Z fallback: small steps", label)
        for _ in range(15):
            c = self.move_group.get_current_pose().pose
            remaining = z_target.position.z - c.position.z
            if abs(remaining) < 0.005:
                return True
            s = copy.deepcopy(c)
            s.position.z += max(-0.012, min(0.012, remaining))
            s.orientation = copy.deepcopy(z_target.orientation)
            wp = [copy.deepcopy(c), copy.deepcopy(s)]
            plan, frac = self.move_group.compute_cartesian_path(wp, 0.003, True)
            if frac >= 0.8 and len(plan.joint_trajectory.points) > 0:
                self.move_group.execute(plan, wait=True)
                rospy.sleep(0.15)
            else:
                break
        return abs(self.move_group.get_current_pose().pose.position.z -
                   target_pose.position.z) < 0.015

    # ════════════════════════════════════════════════════════
    # 主流程
    # ════════════════════════════════════════════════════════
    def execute_and_record(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"MT3 Demo Recording: {self.demo_name}")
        rospy.loginfo(f"Object at: {self.object_pos}")
        rospy.loginfo("=" * 60)

        bx, by, bz = self.object_pos
        obj_h = self.object_size[2]  # 物体全高
        obj_top = bz + obj_h         # 物体顶面 (bz=底部)
        grasp_contact_z = obj_top + 0.005          # 指尖接触高度
        grasp_flange_z = grasp_contact_z + FLANGE_Z_OFFSET  # 法兰高度

        # [1/5] Safe pose
        rospy.loginfo("[1/5] Safe pose...")
        safe_joints = {
            'right_j0': 0.0, 'right_j1': -0.8, 'right_j2': 0.0,
            'right_j3': 1.8, 'right_j4': 0.0, 'right_j5': 0.0, 'right_j6': 0.0,
        }
        self.move_group.set_joint_value_target(safe_joints)
        self.move_group.go(wait=True)
        rospy.sleep(1.0)
        # Open gripper
        try:
            from intera_interface import Gripper
            self.gripper = Gripper('right_gripper')
            if not self.gripper.is_calibrated():
                self.gripper.calibrate()
                rospy.sleep(1.0)
            self.gripper.open()
            rospy.sleep(0.5)
        except Exception:
            pass

        # [2/5] Transition then overhead
        rospy.loginfo("[2/5] Transition + overhead...")
        # Transition (x=0.5)
        trans_pose = geometry_msgs.msg.Pose()
        trans_pose.position.x = 0.50
        trans_pose.position.y = by
        trans_pose.position.z = bz + 0.6
        trans_pose.orientation.x = 1.0
        trans_pose.orientation.y = 0.0
        trans_pose.orientation.z = 0.0
        trans_pose.orientation.w = 0.0
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_pose_target(trans_pose)
        self.move_group.go(wait=True)
        rospy.sleep(0.5)

        # Overhead
        overhead = geometry_msgs.msg.Pose()
        overhead.position.x = bx
        overhead.position.y = by
        overhead.position.z = obj_top + 0.15
        overhead.orientation = copy.deepcopy(trans_pose.orientation)
        self.move_group.set_pose_target(overhead)
        self.move_group.go(wait=True)
        rospy.sleep(1.0)

        use_mouth_center = rospy.get_param("~use_mouth_center_top_grasp", True)
        mouth_xy_offset = [0.0, 0.0]
        if use_mouth_center:
            mouth_xy_offset = self._compute_top_mouth_xy_offset(
                [bx, by], "Top demo overhead")
            overhead.position.x = bx - mouth_xy_offset[0]
            overhead.position.y = by - mouth_xy_offset[1]
            rospy.loginfo(
                "  Corrected overhead right_hand XY: [%.3f, %.3f] "
                "so mouth center targets object XY [%.3f, %.3f]",
                overhead.position.x, overhead.position.y, bx, by)
            self.move_group.set_pose_target(overhead)
            self.move_group.go(wait=True)
            rospy.sleep(0.5)
            self._log_top_mouth_alignment(
                [bx, by], "Top demo corrected overhead", store_key="corrected_overhead")
        else:
            self.mouth_center_calibration = {
                "available": False,
                "reason": "disabled_by_param",
            }

        # [3/5] Bottleneck
        rospy.loginfo("[3/5] Bottleneck...")
        self._capture_bottleneck(timeout=3.0)
        bottleneck_ee = self._get_end_effector_pose()
        if bottleneck_ee is None:
            rospy.logerr("No bottleneck EE!")
            return False
        rospy.loginfo("  Bottleneck EE: %s", bottleneck_ee["position"])

        # [4/5] Grasp + Record
        rospy.loginfo("[4/5] Grasp + Record...")
        self.recording = True
        self.gripper_command_state = 0
        self.recorded_poses = []
        fp = self._stamp_gripper_state(self._get_end_effector_pose())
        if fp:
            self.recorded_poses.append(fp)
        record_thread = threading.Thread(target=self._continuous_record, args=(25.0,))
        record_thread.start()

        # Grasp target
        grasp_pose = geometry_msgs.msg.Pose()
        grasp_pose.position.x = bx - mouth_xy_offset[0]
        grasp_pose.position.y = by - mouth_xy_offset[1]
        grasp_pose.position.z = grasp_flange_z
        grasp_pose.orientation = copy.deepcopy(overhead.orientation)

        self.move_group.set_max_velocity_scaling_factor(0.10)
        self.move_group.set_max_acceleration_scaling_factor(0.10)
        if not self._cartesian_descend(grasp_pose, "grasp"):
            rospy.logwarn("Grasp descent incomplete, attempting close anyway")
        rospy.sleep(0.3)
        self._correct_top_mouth_xy_at_current_z(
            [bx, by], "Top demo before-close mouth XY")

        grasp_ee = self._get_end_effector_pose()
        self._log_top_mouth_alignment(
            [bx, by], "Top demo before close",
            tcp_pose=grasp_ee, store_key="before_close")

        # Close gripper
        rospy.loginfo("  Closing gripper...")
        self.gripper_command_state = 1
        try:
            self.gripper.close()
            rospy.sleep(1.5)
        except Exception:
            rospy.sleep(1.5)

        # Lift
        rospy.loginfo("  Lifting...")
        lift_pose = copy.deepcopy(grasp_pose)
        lift_pose.position.z = obj_top + 0.15
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_pose_target(lift_pose)
        self.move_group.go(wait=True)
        rospy.sleep(0.5)

        self.recording = False
        record_thread.join(timeout=2.0)
        rospy.loginfo("  Frames: %d", len(self.recorded_poses))

        self.move_group.set_max_velocity_scaling_factor(0.6)
        self.move_group.set_max_acceleration_scaling_factor(0.6)

        # [5/5] Save
        rospy.loginfo("[5/5] Saving...")
        self._save_demo(bottleneck_ee, grasp_ee)
        rospy.loginfo("=" * 60)
        rospy.loginfo("Demo '%s' done!", self.demo_name)
        rospy.loginfo("=" * 60)
        return True

    def _save_demo(self, bottleneck_ee, grasp_ee):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        seen_ts = set()
        unique_poses = []
        for p in self.recorded_poses:
            ts = round(p.get("timestamp", 0), 4)
            if ts not in seen_ts:
                seen_ts.add(ts)
                unique_poses.append(p)
        velocities = self._poses_to_velocities(unique_poses)

        demo = {
            "id": self.demo_name,
            "format": "mt3_recorded_v2",
            "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "object_info": {
                "position_base": self.object_pos,
                "size_m": self.object_size,
                "category": "cube", "color": "green",
            },
            "bottleneck_pose_base_frame": {
                "position_m": {"x": bottleneck_ee["position"][0],
                               "y": bottleneck_ee["position"][1],
                               "z": bottleneck_ee["position"][2]},
                "orientation_xyzw": {"x": bottleneck_ee["orientation"][0],
                                     "y": bottleneck_ee["orientation"][1],
                                     "z": bottleneck_ee["orientation"][2],
                                     "w": bottleneck_ee["orientation"][3]},
                "timestamp": bottleneck_ee.get("timestamp", 0),
            },
            "grasp_pose_base_frame": {
                "position_m": {"x": grasp_ee["position"][0] if grasp_ee else 0,
                               "y": grasp_ee["position"][1] if grasp_ee else 0,
                               "z": grasp_ee["position"][2] if grasp_ee else 0},
            },
            "top_grasp_reference": "gripper_mouth_center",
            "top_grasp_mouth_center_calibration": self.mouth_center_calibration,
            "top_grasp_centering_diagnostics": self.top_grasp_centering_diagnostics,
            "trajectory": {
                "format": "end_effector_pose_twist_gripper_v2",
                "frame": "end_effector", "pose_frame": "base",
                "sample_rate_hz": RECORD_RATE,
                "num_waypoints": len(velocities),
                "poses": unique_poses,
                "velocities": velocities,
                "gripper_convention": "gripper_next: 1=close, 0=open, null=unknown",
            },
            "language_tags": ["grasp", "pick up", "cube", "green cube",
                            "top-down grasp", "抓取", "正方体", "绿色方块"],
            "language_description": "Pick up the green cube from above",
            "approach_direction": [0.0, 0.0, -1.0],
            "retract_direction": [0.0, 0.0, 1.0],
            "gripper_opening_m": 0.07,
        }

        json_path = os.path.join(OUTPUT_DIR, f"{self.demo_name}.json")
        with open(json_path, "w") as f:
            json.dump(demo, f, indent=2)

        if self.bottleneck_rgb is not None:
            import cv2
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"{self.demo_name}_bottleneck_rgb.png"),
                self.bottleneck_rgb)
        self._save_scene_package(bottleneck_ee)
        rospy.loginfo("  Saved: %s (%d wp)", json_path, len(velocities))

    def _green_mask_from_bgr(self, bgr):
        if bgr is None:
            return None
        import cv2
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 55, 55), (80, 255, 255))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask.astype(bool)

    def _save_scene_package(self, bottleneck_ee):
        if self.bottleneck_rgb is None or self.bottleneck_depth is None:
            return
        import cv2
        rgb = cv2.cvtColor(self.bottleneck_rgb, cv2.COLOR_BGR2RGB)
        scene_data = {
            "rgb": rgb, "depth": self.bottleneck_depth,
            "segmap": self._green_mask_from_bgr(self.bottleneck_rgb),
            "intrinsics": np.array([
                [407.391526, 0.0, 640.5],
                [0.0, 407.391526, 400.5],
                [0.0, 0.0, 1.0]], dtype=np.float64),
            "pose": {"position": bottleneck_ee["position"],
                     "orientation": bottleneck_ee["orientation"],
                     "method": "recorded_bottleneck_pose", "confidence": 1.0},
        }
        pkg_root = os.path.join(os.path.dirname(OUTPUT_DIR), "scene_packages")
        save_scene_package(scene_data, pkg_root,
                          name=f"demo_{self.demo_name}", role="recorded_demo",
                          extra_metadata={"demo_id": self.demo_name,
                                         "object_position_base": self.object_pos,
                                         "object_size": self.object_size})


if __name__ == "__main__":
    rospy.init_node("mt3_record_demo", anonymous=True)
    obj_x = rospy.get_param("~object_x", 0.60)
    obj_y = rospy.get_param("~object_y", 0.00)
    obj_z = rospy.get_param("~object_z", -0.58)
    obj_size = rospy.get_param("~object_size", [0.045, 0.045, 0.045])
    demo_name = rospy.get_param("~demo_name", "cube_green_top_grasp_recorded")
    recorder = DemoRecorder(obj_x, obj_y, obj_z, obj_size, demo_name)
    try:
        recorder.execute_and_record()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr("Failed: %s", e)
        import traceback; traceback.print_exc()
    finally:
        moveit_commander.roscpp_shutdown()
