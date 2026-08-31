#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import copy
import json
import os
import sys
import threading
import math
import time
import moveit_commander
import geometry_msgs.msg
import tf
from intera_interface import Gripper, RobotEnable
import moveit_msgs.msg
import subprocess
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import OrientationConstraint, Constraints, PositionConstraint, BoundingVolume

# 【关键配置】与Sawyer仿真环境完全匹配
ROS_NAMESPACE = "/robot"                # move_group节点命名空间
PLANNING_GROUP = "right_arm"            # SRDF机械臂规划组名称
END_EFFECTOR_LINK = "right_hand"        # 法兰中心link（官方SRDF默认）

# 通用配置【完全未修改】
FINGER_LENGTH = 0.03  # 夹爪指尖长度（法兰→指尖Z轴偏移，实测值）
ALLOWED_ERROR = 0.005 # 【优化】XY对齐误差阈值从0.002→0.005，放宽要求
MAX_RETRY = 3         # 【优化】对齐最大重试次数从10→3
CART_STEP = 0.01      # 笛卡尔步长（改0.01解决minjerk数值警告）
CART_VEL_SCALE = 0.1  # 笛卡尔执行速度（改0.1解决速度超限ABORTED）

# 【优化】速度缩放因子：从保守的0.1/0.3→高效的0.6/0.8
ORI_VEL_SCALE = 0.3    # 常规运动速度，保持和已验证录制脚本接近
ORI_ACC_SCALE = 0.3
DOWN_VEL_SCALE = 0.1   # 下降阶段低速，减少接触时把方块推走
DOWN_ACC_SCALE = 0.1


def install_moveit_timing(move_group):
    """Accumulate MoveIt planning/execution wall time in ROS params."""
    plan_param = "/sawyer_auto_grasp/planning_time_s"
    exec_param = "/sawyer_auto_grasp/robot_execution_time_s"
    count_plan_param = "/sawyer_auto_grasp/planning_call_count"
    count_exec_param = "/sawyer_auto_grasp/robot_execution_call_count"
    rospy.set_param(plan_param, 0.0)
    rospy.set_param(exec_param, 0.0)
    rospy.set_param(count_plan_param, 0)
    rospy.set_param(count_exec_param, 0)
    rospy.set_param("/sawyer_auto_grasp/timing_source", "moveit_wrapper_v1")

    def _accumulate(param, count_param, dt):
        try:
            rospy.set_param(param, float(rospy.get_param(param, 0.0)) + float(dt))
            rospy.set_param(count_param, int(rospy.get_param(count_param, 0)) + 1)
        except Exception:
            pass

    original_plan = move_group.plan
    original_execute = move_group.execute
    original_go = move_group.go

    def timed_plan(*args, **kwargs):
        t0 = time.time()
        try:
            return original_plan(*args, **kwargs)
        finally:
            _accumulate(plan_param, count_plan_param, time.time() - t0)

    def timed_execute(*args, **kwargs):
        t0 = time.time()
        try:
            return original_execute(*args, **kwargs)
        finally:
            _accumulate(exec_param, count_exec_param, time.time() - t0)

    def timed_go(*args, **kwargs):
        t0 = time.time()
        try:
            return original_go(*args, **kwargs)
        finally:
            _accumulate(exec_param, count_exec_param, time.time() - t0)

    move_group.plan = timed_plan
    move_group.execute = timed_execute
    move_group.go = timed_go

class EndEffectorTrajectoryRecorder(object):
    """Sample right_hand pose during execution and export pose/twist sequence."""

    def __init__(self, move_group, gripper=None, rate_hz=10.0):
        self.move_group = move_group
        self.gripper = gripper
        self.rate_hz = float(rate_hz)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self._stop.is_set():
            try:
                pose = self.move_group.get_current_pose().pose
                gripper_pos = None
                if self.gripper is not None:
                    try:
                        gripper_pos = float(self.gripper.get_position())
                    except Exception:
                        gripper_pos = None
                self.samples.append({
                    "t": float(rospy.get_time()),
                    "position": [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ],
                    "orientation": [
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ],
                    "gripper_position": gripper_pos,
                })
            except Exception:
                pass
            rate.sleep()

    def to_trajectory(self):
        velocities = []
        for i, sample in enumerate(self.samples):
            if i == 0:
                linear = [0.0, 0.0, 0.0]
            else:
                prev = self.samples[i - 1]
                dt = max(1e-6, sample["t"] - prev["t"])
                linear = [
                    (sample["position"][j] - prev["position"][j]) / dt
                    for j in range(3)
                ]
            velocities.append({
                "t": sample["t"],
                "linear": [float(v) for v in linear],
                "angular": [0.0, 0.0, 0.0],
                "gripper_position": sample.get("gripper_position"),
            })

        return {
            "format": "sampled_end_effector_pose_twist",
            "frame": "base",
            "sample_rate_hz": self.rate_hz,
            "num_waypoints": len(self.samples),
            "poses": self.samples,
            "velocities": velocities,
            "notes": "Sampled during scripted MoveIt grasp execution; angular velocity is not estimated yet.",
        }

    def save(self, path, success):
        if not path:
            return None
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        data = self.to_trajectory()
        data["success"] = bool(success)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

# Sawyer官方关节物理限位（禁止修改）【完全未修改】
JOINT_LIMITS = {
    'right_j0': (-3.05, 3.05),
    'right_j1': (-1.92, 1.396),
    'right_j2': (-3.05, 3.05),
    'right_j3': (-3.05, 3.05),
    'right_j4': (-3.05, 3.05),
    'right_j5': (-3.05, 3.05),
    'right_j6': (-5.23, 5.23)
}

# 关节限位修正函数（防止关节超界）【完全未修改】
def clamp_joint_value(joint_name, value):
    if joint_name not in JOINT_LIMITS:
        return value
    min_val, max_val = JOINT_LIMITS[joint_name]
    clamped = max(min_val, min(value, max_val))
    if abs(value - clamped) > 0.01:
        rospy.logwarn(f"关节{joint_name}超界修正：{value:.3f}→{clamped:.3f}")
    return clamped

# 等待move_group节点启动（带超时检测）【完全未修改】
def quat_rotate(q, v):
    """Rotate vector v by quaternion q=[x,y,z,w]."""
    x, y, z, w = q
    vx, vy, vz = v
    return [
        (1 - 2*y*y - 2*z*z) * vx + (2*x*y - 2*w*z) * vy + (2*x*z + 2*w*y) * vz,
        (2*x*y + 2*w*z) * vx + (1 - 2*x*x - 2*z*z) * vy + (2*y*z - 2*w*x) * vz,
        (2*x*z - 2*w*y) * vx + (2*y*z + 2*w*x) * vy + (1 - 2*x*x - 2*y*y) * vz,
    ]


def load_demo_replay_velocities(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    trajectory = payload.get("trajectory", payload)
    velocities = trajectory.get("velocities", [])
    if not velocities:
        poses = trajectory.get("poses", [])
        velocities = []
        for i in range(1, len(poses)):
            prev = poses[i - 1]
            cur = poses[i]
            dt = max(1e-3, float(cur.get("t", cur.get("timestamp", i))) -
                     float(prev.get("t", prev.get("timestamp", i - 1))))
            p0 = prev.get("position", [0.0, 0.0, 0.0])
            p1 = cur.get("position", [0.0, 0.0, 0.0])
            velocities.append({
                "timestamp": float(cur.get("t", cur.get("timestamp", i))),
                "linear": [(float(p1[j]) - float(p0[j])) / dt for j in range(3)],
                "angular": [0.0, 0.0, 0.0],
            })
    return payload, velocities


def _pose_position_list(sample):
    pos = sample.get("position", [0.0, 0.0, 0.0])
    if isinstance(pos, dict):
        return [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        ]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _find_replay_close_pose_index(poses, velocities):
    for i, sample in enumerate(poses):
        if sample.get("gripper_next", None) == 1:
            return i
        if sample.get("gripper_state", None) == 1:
            return i
    for i, v in enumerate(velocities or []):
        if v.get("gripper_next", None) == 1:
            return min(i + 1, len(poses) - 1)
        if v.get("gripper_state", None) == 1:
            return min(i + 1, len(poses) - 1)
    if poses:
        z_values = [_pose_position_list(p)[2] for p in poses]
        return int(z_values.index(min(z_values)))
    return None


def make_replay_waypoints_from_poses(start_pose, target_orientation, poses,
                                     velocities, yaw_delta=0.0,
                                     close_anchor_pose=None):
    """Replay recorded pose sequence as relative displacement from bottleneck."""
    if not poses or len(poses) < 2:
        return [], None

    base = _pose_position_list(poses[0])
    anchor_pose = start_pose
    cos_yaw = math.cos(float(yaw_delta))
    sin_yaw = math.sin(float(yaw_delta))
    waypoints = []
    gripper_close_index = None
    for i, sample in enumerate(poses[1:], start=1):
        p = _pose_position_list(sample)
        dx = p[0] - base[0]
        dy = p[1] - base[1]
        dz = p[2] - base[2]
        rotated_dx = cos_yaw * dx - sin_yaw * dy
        rotated_dy = sin_yaw * dx + cos_yaw * dy
        waypoint = copy.deepcopy(anchor_pose)
        waypoint.position.x += rotated_dx
        waypoint.position.y += rotated_dy
        waypoint.position.z += dz
        waypoint.orientation = copy.deepcopy(target_orientation)
        waypoints.append(waypoint)

        # In recorded v2 demos, velocity[i-1].gripper_next marks the command
        # that should be applied after moving to pose i.
        v_index = i - 1
        if (gripper_close_index is None and velocities and
                v_index < len(velocities) and
                velocities[v_index].get("gripper_next", None) == 1):
            gripper_close_index = len(waypoints) - 1

    if gripper_close_index is None and waypoints:
        z_values = [p.position.z for p in waypoints]
        gripper_close_index = int(z_values.index(min(z_values)))
        rospy.logwarn(
            "Pose replay has no gripper_next labels; closing at lowest replay z")

    if close_anchor_pose is not None and gripper_close_index is not None and waypoints:
        close_index = max(0, min(len(waypoints) - 1, int(gripper_close_index)))
        close_wp = waypoints[close_index]
        offset_x = float(close_anchor_pose.position.x) - float(close_wp.position.x)
        offset_y = float(close_anchor_pose.position.y) - float(close_wp.position.y)
        # Keep the demonstrated vertical profile.  The scripted grasp target z
        # includes flange/contact offsets that can be several centimeters deeper
        # than the recorded replay close point; applying that z shift makes the
        # Cartesian path discontinuous from the current bottleneck pose.
        offset_z = 0.0
        for waypoint in waypoints:
            waypoint.position.x += offset_x
            waypoint.position.y += offset_y
            waypoint.position.z += offset_z

    return waypoints, gripper_close_index


def velocity_linear_ee(v):
    if "linear_ee" in v:
        return v.get("linear_ee") or [0.0, 0.0, 0.0]
    if "linear" in v:
        return v.get("linear") or [0.0, 0.0, 0.0]
    if "twist" in v:
        twist = v.get("twist") or []
        if len(twist) >= 3:
            return twist[:3]
    return [v.get("vx", 0.0), v.get("vy", 0.0), v.get("vz", 0.0)]


def velocity_angular_ee(v):
    if "angular_ee" in v:
        return v.get("angular_ee") or [0.0, 0.0, 0.0]
    if "angular" in v:
        return v.get("angular") or [0.0, 0.0, 0.0]
    if "twist" in v:
        twist = v.get("twist") or []
        if len(twist) >= 6:
            return twist[3:6]
    return [v.get("wx", 0.0), v.get("wy", 0.0), v.get("wz", 0.0)]


def velocity_dt(velocities, index, default_dt=0.10):
    cur = velocities[index]
    if index == 0:
        return default_dt
    prev = velocities[index - 1]
    t0 = prev.get("timestamp", prev.get("t", None))
    t1 = cur.get("timestamp", cur.get("t", None))
    if t0 is None or t1 is None:
        return default_dt
    return max(0.02, min(0.20, float(t1) - float(t0)))


def make_replay_waypoints(start_pose, target_orientation, velocities):
    """Integrate recorded end-effector-frame velocities into Cartesian waypoints."""
    q = [target_orientation.x, target_orientation.y,
         target_orientation.z, target_orientation.w]
    current = copy.deepcopy(start_pose)
    current.orientation = copy.deepcopy(target_orientation)
    waypoints = []
    gripper_close_index = None

    for i, v in enumerate(velocities):
        dt = velocity_dt(velocities, i)
        lin_ee_step = [float(x) * dt for x in velocity_linear_ee(v)]
        lin_world_step = quat_rotate(q, lin_ee_step)
        current = copy.deepcopy(current)
        current.position.x += lin_world_step[0]
        current.position.y += lin_world_step[1]
        current.position.z += lin_world_step[2]
        current.orientation = copy.deepcopy(target_orientation)
        waypoints.append(copy.deepcopy(current))
        if gripper_close_index is None and v.get("gripper_next", None) == 1:
            gripper_close_index = i

    if gripper_close_index is None and waypoints:
        z_values = [p.position.z for p in waypoints]
        gripper_close_index = int(z_values.index(min(z_values)))
        rospy.logwarn(
            "Replay trajectory has no gripper_next labels; closing at lowest integrated z")

    return waypoints, gripper_close_index


def execute_cartesian_waypoint_segment(move_group, waypoints, label,
                                       min_fraction=0.75,
                                       allow_partial=True):
    if not waypoints:
        return True
    rospy.loginfo("Replay Cartesian %s: %d waypoints", label, len(waypoints))
    try:
        plan, fraction = move_group.compute_cartesian_path(
            waypoints, CART_STEP, True)
    except TypeError:
        plan, fraction = move_group.compute_cartesian_path(
            waypoints, CART_STEP, 0.0)
    rospy.loginfo("Replay Cartesian %s fraction: %.1f%%", label, fraction * 100.0)
    if fraction < min_fraction:
        rospy.logwarn(
            "Replay Cartesian %s only followed %.1f%% < %.1f%%; executing reachable prefix",
            label, fraction * 100.0, min_fraction * 100.0)
        if not allow_partial:
            rospy.logerr(
                "Replay Cartesian %s aborted before execution because partial "
                "grasp replay would move the gripper off the object.",
                label)
            return False
    if fraction <= 0.05:
        rospy.logerr("Replay Cartesian %s failed: no usable path", label)
        return False
    ok = move_group.execute(plan, wait=True)
    rospy.sleep(0.2)
    return bool(ok)


def execute_cartesian_waypoint_segmented(move_group, waypoints, label,
                                         chunk_size=12,
                                         min_fraction=0.75,
                                         return_progress=False):
    """Execute replay waypoints in short Cartesian chunks.

    Planning a long dense replay segment as one Cartesian path can fail near
    the robot workspace boundary even when each local motion is feasible.  This
    keeps the MT3 replay trajectory but asks MoveIt to solve it piece by piece.
    """
    if not waypoints:
        return (True, 1.0) if return_progress else True
    try:
        chunk_size = int(rospy.get_param(
            '/sawyer_auto_grasp/replay_chunk_size', chunk_size))
    except Exception:
        chunk_size = int(chunk_size)
    chunk_size = max(2, min(40, chunk_size))
    rospy.loginfo(
        "Replay Cartesian %s segmented execution: %d waypoints, chunk_size=%d",
        label, len(waypoints), chunk_size)
    chunk_no = 0
    total = len(waypoints)
    executed_count = 0
    for start in range(0, total, chunk_size):
        chunk_no += 1
        end = min(total, start + chunk_size)
        chunk = waypoints[start:end]
        chunk_label = "%s chunk %02d [%d-%d/%d]" % (
            label, chunk_no, start + 1, end, total)
        ok = execute_cartesian_waypoint_segment(
            move_group, chunk, chunk_label,
            min_fraction=min_fraction,
            allow_partial=False)
        if not ok:
            progress = float(executed_count) / float(total) if total else 0.0
            rospy.logerr(
                "Replay Cartesian %s segmented execution failed at chunk %02d "
                "(progress=%.1f%%)",
                label, chunk_no, progress * 100.0)
            return (False, progress) if return_progress else False
        executed_count = end
    return (True, 1.0) if return_progress else True


def _set_pose_position(pose, x=None, y=None, z=None):
    next_pose = copy.deepcopy(pose)
    if x is not None:
        next_pose.position.x = float(x)
    if y is not None:
        next_pose.position.y = float(y)
    if z is not None:
        next_pose.position.z = float(z)
    return next_pose


def side_contact_pose_to_flange_pose(contact_pose, approach_sign,
                                     tcp_forward_offset, label):
    """Convert side-grasp contact target to right_hand flange target.

    Recorded MT3 demos already store right_hand flange poses.  The offset is
    therefore disabled by default and is only used for explicitly contact-based
    external targets.
    """
    flange_pose = copy.deepcopy(contact_pose)
    flange_pose.position.x = (
        float(contact_pose.position.x) -
        float(approach_sign) * float(tcp_forward_offset)
    )
    rospy.loginfo(
        "%s side TCP->flange: contact_x=%.3f -> flange_x=%.3f "
        "(approach_sign=%+.0f tcp_forward=%.3f)",
        label, contact_pose.position.x, flange_pose.position.x,
        approach_sign, tcp_forward_offset)
    return flange_pose


def _safe_float_param(name, default):
    try:
        return float(rospy.get_param(name, default))
    except Exception:
        return float(default)


def _safe_size_param(name, default):
    value = rospy.get_param(name, default)
    try:
        if value and len(value) >= 3:
            return [abs(float(value[0])), abs(float(value[1])), abs(float(value[2]))]
    except Exception:
        pass
    return [float(default[0]), float(default[1]), float(default[2])]


def _axis_value(point, axis):
    if axis == "x":
        return float(point[0])
    if axis == "y":
        return float(point[1])
    return float(point[2])


def _set_pose_axis(pose, axis, value):
    if axis == "x":
        pose.position.x = float(value)
    elif axis == "y":
        pose.position.y = float(value)
    else:
        pose.position.z = float(value)


def _pose_position_array(pose):
    return [float(pose.position.x), float(pose.position.y), float(pose.position.z)]


def _array_to_pose_like(reference_pose, xyz):
    pose = copy.deepcopy(reference_pose)
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    return pose


def _lookup_tf_point(listener, frame):
    try:
        listener.waitForTransform("base", frame, rospy.Time(0), rospy.Duration(0.3))
        trans, _ = listener.lookupTransform("base", frame, rospy.Time(0))
        return [float(trans[0]), float(trans[1]), float(trans[2])]
    except Exception:
        return None


def get_gripper_mouth_state(listener, move_group):
    """Measure the open gripper mouth from the two finger-tip TF frames."""
    left_frame = str(rospy.get_param(
        '/sawyer_auto_grasp/left_finger_tip_frame',
        'right_gripper_l_finger_tip'))
    right_frame = str(rospy.get_param(
        '/sawyer_auto_grasp/right_finger_tip_frame',
        'right_gripper_r_finger_tip'))
    left = _lookup_tf_point(listener, left_frame)
    right = _lookup_tf_point(listener, right_frame)
    hand_pose = move_group.get_current_pose().pose
    hand = _pose_position_array(hand_pose)
    if left is None or right is None:
        rospy.logwarn(
            "Gripper mouth TF unavailable; using right_hand as fallback center")
        return {
            "available": False,
            "left": None,
            "right": None,
            "center": hand,
            "opening": 0.0,
            "hand": hand,
            "offset": [0.0, 0.0, 0.0],
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
        "left": left,
        "right": right,
        "center": center,
        "opening": opening,
        "hand": hand,
        "offset": [
            center[0] - hand[0],
            center[1] - hand[1],
            center[2] - hand[2],
        ],
    }


def command_pose_for_mouth_center(reference_pose, desired_mouth_center, mouth_offset):
    return _array_to_pose_like(reference_pose, [
        float(desired_mouth_center[0]) - float(mouth_offset[0]),
        float(desired_mouth_center[1]) - float(mouth_offset[1]),
        float(desired_mouth_center[2]) - float(mouth_offset[2]),
    ])


def log_side_mouth_check(listener, move_group, label, object_center,
                         object_radius, lateral_axis):
    state = get_gripper_mouth_state(listener, move_group)
    center = state["center"]
    mouth_lateral = _axis_value(center, lateral_axis)
    object_lateral = _axis_value(object_center, lateral_axis)
    lateral_error = mouth_lateral - object_lateral
    opening = float(state.get("opening", 0.0))
    clearance = opening * 0.5 - float(object_radius) - abs(lateral_error)
    rospy.loginfo(
        "%s mouth-center check: mouth=[%.3f, %.3f, %.3f] "
        "object=[%.3f, %.3f, %.3f] d%s=%.1fcm opening=%.1fcm "
        "finger_clearance=%.1fcm",
        label, center[0], center[1], center[2],
        object_center[0], object_center[1], object_center[2],
        lateral_axis, lateral_error * 100.0, opening * 100.0,
        clearance * 100.0)
    return state, lateral_error, clearance


def log_top_mouth_xy_check(listener, move_group, label, desired_xy):
    """Log the real gripper mouth center against a top-grasp XY target."""
    state = get_gripper_mouth_state(listener, move_group)
    center = state["center"]
    hand = state["hand"]
    dx = center[0] - float(desired_xy[0])
    dy = center[1] - float(desired_xy[1])
    offset = state.get("offset", [0.0, 0.0, 0.0])
    rospy.loginfo(
        "%s top mouth-center check: hand=[%.3f, %.3f, %.3f] "
        "mouth=[%.3f, %.3f, %.3f] desired_xy=[%.3f, %.3f] "
        "mouth_err_xy=[%.1f, %.1f]cm hand_to_mouth_xy=[%.1f, %.1f]cm "
        "tf_available=%s",
        label, hand[0], hand[1], hand[2],
        center[0], center[1], center[2],
        float(desired_xy[0]), float(desired_xy[1]),
        dx * 100.0, dy * 100.0,
        float(offset[0]) * 100.0, float(offset[1]) * 100.0,
        state.get("available", False))
    return state, dx, dy


def correct_top_mouth_xy_before_close(listener, move_group, label, desired_xy):
    """For top_grasp only: make a small final XY correction using finger-tip TF."""
    group = str(rospy.get_param(
        '/sawyer_auto_grasp/experiment_group', '')).strip().lower()
    if group != "top_grasp":
        return False
    if not rospy.get_param(
            '/sawyer_auto_grasp/use_top_mouth_center_final_correction', True):
        return False

    tol = float(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_tolerance', 0.003))
    max_step = float(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_final_max_step', 0.018))
    attempts = int(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_final_attempts', 3))

    for attempt in range(max(1, attempts)):
        state, dx, dy = log_top_mouth_xy_check(
            listener, move_group, "%s correction %d" % (label, attempt + 1),
            desired_xy)
        if not state.get("available", False):
            rospy.logwarn(
                "%s correction skipped: gripper mouth TF unavailable", label)
            return False
        if abs(dx) <= tol and abs(dy) <= tol:
            rospy.loginfo(
                "%s correction done: mouth_err_xy=[%.1f, %.1f]cm",
                label, dx * 100.0, dy * 100.0)
            rospy.set_param(
                '/sawyer_auto_grasp/top_mouth_xy_final_error_m',
                [float(dx), float(dy)])
            return True

        err = math.sqrt(dx * dx + dy * dy)
        scale = min(1.0, max_step / max(err, 1e-6))
        current = move_group.get_current_pose().pose
        target = copy.deepcopy(current)
        target.position.x = current.position.x - dx * scale
        target.position.y = current.position.y - dy * scale
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(current), copy.deepcopy(target)], 0.003, True)
        rospy.loginfo(
            "%s correction %d cartesian fraction: %.1f%% target_hand_xy=[%.3f, %.3f]",
            label, attempt + 1, fraction * 100.0,
            target.position.x, target.position.y)
        if fraction >= 0.90 and len(plan.joint_trajectory.points) > 0:
            ok = move_group.execute(plan, wait=True)
            try:
                move_group.stop()
            except Exception:
                pass
            rospy.sleep(0.25)
            if not ok:
                rospy.logwarn("%s correction %d execute returned false",
                              label, attempt + 1)
                break
        else:
            rospy.logwarn(
                "%s correction %d skipped: insufficient Cartesian path %.1f%%",
                label, attempt + 1, fraction * 100.0)
            break

    state, dx, dy = log_top_mouth_xy_check(
        listener, move_group, "%s correction final" % label, desired_xy)
    rospy.set_param(
        '/sawyer_auto_grasp/top_mouth_xy_final_error_m',
        [float(dx), float(dy)])
    return abs(dx) <= tol and abs(dy) <= tol


def record_before_close_mouth_xy(listener, move_group, label, desired_xy):
    """Record the real gripper mouth center immediately before closing."""
    try:
        state = get_gripper_mouth_state(listener, move_group)
        if not state.get("available", False):
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy', ["", ""])
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy', ["", ""])
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', "")
            rospy.logwarn(
                "%s before-close mouth record skipped: gripper mouth TF unavailable",
                label)
            return False

        center = state["center"]
        dx = float(center[0]) - float(desired_xy[0])
        dy = float(center[1]) - float(desired_xy[1])
        err = math.sqrt(dx * dx + dy * dy)
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy',
                        [float(center[0]), float(center[1])])
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy',
                        [float(dx), float(dy)])
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', float(center[0]))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', float(center[1]))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', float(dx))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', float(dy))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', float(err))
        rospy.loginfo(
            "%s before-close mouth record: mouth=[%.3f, %.3f] "
            "desired_xy=[%.3f, %.3f] err_xy=[%.1f, %.1f]cm norm=%.1fcm",
            label, center[0], center[1], desired_xy[0], desired_xy[1],
            dx * 100.0, dy * 100.0, err * 100.0)
        return True
    except Exception as exc:
        rospy.logwarn("%s before-close mouth record failed: %s", label, exc)
        return False


def add_side_target_collision(scene):
    """Add the currently perceived target only during side-grasp approach."""
    enabled = rospy.get_param('/sawyer_auto_grasp/enable_side_target_collision', True)
    if not enabled:
        return None

    name = str(rospy.get_param(
        '/sawyer_auto_grasp/target_collision_name', 'mt3_target_object'))
    object_shape = str(rospy.get_param(
        '/sawyer_auto_grasp/object_shape', 'unknown')).lower()
    size = _safe_size_param(
        '/sawyer_auto_grasp/object_size', [0.045, 0.045, 0.045])
    sx, sy, sz = size
    x = _safe_float_param(
        '/sawyer_auto_grasp/object_base_x',
        rospy.get_param('/sawyer_auto_grasp/obj_base_x',
                        rospy.get_param('/sawyer_auto_grasp/grasp_x', 0.60)))
    y = _safe_float_param(
        '/sawyer_auto_grasp/object_base_y',
        rospy.get_param('/sawyer_auto_grasp/obj_base_y',
                        rospy.get_param('/sawyer_auto_grasp/grasp_y', 0.0)))
    z_bottom = _safe_float_param(
        '/sawyer_auto_grasp/object_base_z',
        rospy.get_param('/sawyer_auto_grasp/obj_base_z',
                        rospy.get_param('/sawyer_auto_grasp/grasp_z', -0.58)))

    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = 'base'
    pose.pose.orientation.w = 1.0
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z_bottom + sz / 2.0

    try:
        scene.remove_world_object(name)
        rospy.sleep(0.2)
        if 'cylinder' in object_shape:
            radius = max(0.5 * sx, 0.5 * sy, 0.01)
            try:
                scene.add_cylinder(name, pose, sz, radius)
                rospy.loginfo(
                    "Side target collision added as cylinder: name=%s center=[%.3f, %.3f, %.3f] h=%.3f r=%.3f",
                    name, pose.pose.position.x, pose.pose.position.y,
                    pose.pose.position.z, sz, radius)
            except Exception as exc:
                rospy.logwarn(
                    "add_cylinder unavailable (%s); using box collision fallback",
                    exc)
                scene.add_box(name, pose, size=(sx, sy, sz))
        else:
            scene.add_box(name, pose, size=(sx, sy, sz))
            rospy.loginfo(
                "Side target collision added as box: name=%s center=[%.3f, %.3f, %.3f] size=[%.3f, %.3f, %.3f]",
                name, pose.pose.position.x, pose.pose.position.y,
                pose.pose.position.z, sx, sy, sz)
        rospy.sleep(0.5)
        return name
    except Exception as exc:
        rospy.logwarn("Failed to add side target collision: %s", exc)
        return None


def remove_side_target_collision(scene, name, label):
    if scene is None or not name:
        return
    try:
        scene.remove_world_object(name)
        rospy.sleep(0.3)
        rospy.loginfo("%s: removed target collision object %s", label, name)
    except Exception as exc:
        rospy.logwarn("%s: failed to remove target collision object %s: %s",
                      label, name, exc)


def execute_pose_target(move_group, pose, label, velocity=0.12,
                        acceleration=0.12, attempts=3, planning_time=8.0):
    move_group.set_max_velocity_scaling_factor(velocity)
    move_group.set_max_acceleration_scaling_factor(acceleration)
    move_group.set_num_planning_attempts(attempts)
    move_group.set_planning_time(planning_time)
    move_group.set_goal_position_tolerance(0.012)
    move_group.set_goal_orientation_tolerance(0.10)
    try:
        move_group.set_start_state_to_current_state()
        move_group.clear_pose_targets()
    except Exception:
        pass
    rospy.loginfo("%s pose target: [%.3f, %.3f, %.3f]",
                  label, pose.position.x, pose.position.y, pose.position.z)
    move_group.set_pose_target(pose)
    ok = move_group.go(wait=True)
    try:
        move_group.stop()
        move_group.clear_pose_targets()
    except Exception:
        pass
    rospy.sleep(0.4)
    if not ok:
        rospy.logerr("%s planning/execution failed", label)
    return bool(ok)


def execute_position_target(move_group, pose, label, velocity=0.10,
                            acceleration=0.10, attempts=8,
                            planning_time=12.0):
    """Move to xyz only, leaving wrist orientation free for easier staging."""
    move_group.set_max_velocity_scaling_factor(velocity)
    move_group.set_max_acceleration_scaling_factor(acceleration)
    move_group.set_num_planning_attempts(attempts)
    move_group.set_planning_time(planning_time)
    move_group.set_goal_position_tolerance(0.015)
    try:
        move_group.set_start_state_to_current_state()
        move_group.clear_pose_targets()
    except Exception:
        pass
    xyz = [pose.position.x, pose.position.y, pose.position.z]
    rospy.loginfo("%s position target only: [%.3f, %.3f, %.3f]",
                  label, xyz[0], xyz[1], xyz[2])
    move_group.set_position_target(xyz, END_EFFECTOR_LINK)
    ok = move_group.go(wait=True)
    try:
        move_group.stop()
        move_group.clear_pose_targets()
    except Exception:
        pass
    rospy.sleep(0.4)
    if not ok:
        rospy.logerr("%s position-only planning/execution failed", label)
    return bool(ok)


def execute_cartesian_pose(move_group, pose, label, eef_step=0.010,
                           min_fraction=0.45, accept_error=0.045,
                           strict_error=False):
    move_group.set_max_velocity_scaling_factor(0.05)
    move_group.set_max_acceleration_scaling_factor(0.05)
    current = move_group.get_current_pose().pose
    plan, fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(current), copy.deepcopy(pose)], eef_step, True)
    rospy.loginfo("%s cartesian fraction: %.1f%%",
                  label, fraction * 100.0)
    if fraction > 0.05 and len(plan.joint_trajectory.points) > 0:
        ok = move_group.execute(plan, wait=True)
        rospy.sleep(0.4)
    else:
        ok = False
    actual = move_group.get_current_pose().pose
    dx = actual.position.x - pose.position.x
    dy = actual.position.y - pose.position.y
    dz = actual.position.z - pose.position.z
    error = math.sqrt(dx * dx + dy * dy + dz * dz)
    rospy.loginfo(
        "%s result error=%.1fcm (dx=%.1f dy=%.1f dz=%.1fcm)",
        label, error * 100.0, dx * 100.0, dy * 100.0, dz * 100.0)
    if error <= accept_error:
        return True
    if strict_error:
        rospy.logerr("%s failed: result error %.1fcm > %.1fcm",
                     label, error * 100.0, accept_error * 100.0)
        return False
    if not ok or fraction < min_fraction:
        rospy.logerr("%s failed: fraction %.1f%% error %.1fcm",
                     label, fraction * 100.0, error * 100.0)
        return False
    return True


def execute_incremental_cartesian_pose(move_group, target_pose, label,
                                       max_step=0.035, eef_step=0.006,
                                       accept_error=0.025,
                                       max_steps=30,
                                       step_accept_error=None,
                                       min_progress=0.004,
                                       axes=("x", "y", "z")):
    """Move toward target pose through short Cartesian segments."""
    axes = set(axes)
    rospy.loginfo("%s incremental move: target=[%.3f, %.3f, %.3f] step=%.1fcm",
                  label, target_pose.position.x, target_pose.position.y,
                  target_pose.position.z, max_step * 100.0)
    for i in range(max_steps):
        current = move_group.get_current_pose().pose
        dx = target_pose.position.x - current.position.x if "x" in axes else 0.0
        dy = target_pose.position.y - current.position.y if "y" in axes else 0.0
        dz = target_pose.position.z - current.position.z if "z" in axes else 0.0
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= accept_error:
            rospy.loginfo("%s reached: remaining=%.1fcm", label, dist * 100.0)
            return True

        ratio = min(1.0, max_step / max(dist, 1e-6))
        next_pose = copy.deepcopy(target_pose)
        next_pose.position.x = current.position.x + dx * ratio if "x" in axes else current.position.x
        next_pose.position.y = current.position.y + dy * ratio if "y" in axes else current.position.y
        next_pose.position.z = current.position.z + dz * ratio if "z" in axes else current.position.z
        before_remaining = dist
        local_accept = step_accept_error
        if local_accept is None:
            local_accept = max(0.006, min(0.012, max_step * 0.30))
        ok = execute_cartesian_pose(
            move_group, next_pose,
            "%s step %02d" % (label, i + 1),
            eef_step=eef_step, min_fraction=0.55,
            accept_error=local_accept,
            strict_error=True)
        current_after = move_group.get_current_pose().pose
        remaining = math.sqrt(
            ((target_pose.position.x - current_after.position.x) if "x" in axes else 0.0) ** 2 +
            ((target_pose.position.y - current_after.position.y) if "y" in axes else 0.0) ** 2 +
            ((target_pose.position.z - current_after.position.z) if "z" in axes else 0.0) ** 2)
        progress = before_remaining - remaining
        rospy.loginfo("%s step %02d ok=%s remaining=%.1fcm",
                      label, i + 1, ok, remaining * 100.0)
        if not ok:
            return False
        if remaining > accept_error and progress < min_progress:
            rospy.logerr(
                "%s stalled at step %02d: progress %.1fcm < %.1fcm",
                label, i + 1, progress * 100.0, min_progress * 100.0)
            return False

    current = move_group.get_current_pose().pose
    remaining = math.sqrt(
        ((target_pose.position.x - current.position.x) if "x" in axes else 0.0) ** 2 +
        ((target_pose.position.y - current.position.y) if "y" in axes else 0.0) ** 2 +
        ((target_pose.position.z - current.position.z) if "z" in axes else 0.0) ** 2)
    if remaining <= accept_error:
        rospy.loginfo("%s reached after max steps: remaining=%.1fcm",
                      label, remaining * 100.0)
        return True
    rospy.logerr("%s failed: remaining %.1fcm > %.1fcm",
                 label, remaining * 100.0, accept_error * 100.0)
    return False


def side_yz_ready(move_group, target_pose, label,
                  accept_y=0.008, accept_z=0.012):
    current = move_group.get_current_pose().pose
    dy = current.position.y - target_pose.position.y
    dz = current.position.z - target_pose.position.z
    rospy.loginfo(
        "%s yz readiness: dy=%.1fcm dz=%.1fcm",
        label, dy * 100.0, dz * 100.0)
    if abs(dy) <= accept_y and abs(dz) <= accept_z:
        return True
    rospy.logerr(
        "%s not ready for x approach: |dy| %.1fcm / %.1fcm, "
        "|dz| %.1fcm / %.1fcm",
        label, abs(dy) * 100.0, accept_y * 100.0,
        abs(dz) * 100.0, accept_z * 100.0)
    return False


def side_xy_ready(move_group, target_pose, label,
                  accept_x=0.014, accept_y=0.012):
    current = move_group.get_current_pose().pose
    dx = current.position.x - target_pose.position.x
    dy = current.position.y - target_pose.position.y
    rospy.loginfo(
        "%s xy readiness: dx=%.1fcm dy=%.1fcm",
        label, dx * 100.0, dy * 100.0)
    if abs(dx) <= accept_x and abs(dy) <= accept_y:
        return True
    rospy.logerr(
        "%s not ready before lowering: |dx| %.1fcm / %.1fcm, "
        "|dy| %.1fcm / %.1fcm",
        label, abs(dx) * 100.0, accept_x * 100.0,
        abs(dy) * 100.0, accept_y * 100.0)
    return False


def side_close_enough_for_grasp(move_group, grasp_pose, label,
                                accept_x=0.020, accept_y=0.012,
                                accept_z=0.018):
    current = move_group.get_current_pose().pose
    dx = current.position.x - grasp_pose.position.x
    dy = current.position.y - grasp_pose.position.y
    dz = current.position.z - grasp_pose.position.z
    rospy.loginfo(
        "%s close-window check: dx=%.1fcm dy=%.1fcm dz=%.1fcm",
        label, dx * 100.0, dy * 100.0, dz * 100.0)
    if abs(dx) <= accept_x and abs(dy) <= accept_y and abs(dz) <= accept_z:
        rospy.logwarn(
            "%s accepted for gripper close inside side-grasp window",
            label)
        return True
    return False


def stop_and_save_trajectory(trajectory_recorder, trajectory_record_path, success):
    if trajectory_recorder is None:
        return
    try:
        trajectory_recorder.stop()
        saved_path = trajectory_recorder.save(
            trajectory_record_path, success=success)
        rospy.loginfo("End-effector trajectory saved: %s samples=%d success=%s",
                      saved_path, len(trajectory_recorder.samples), success)
    except Exception as exc:
        rospy.logwarn("Failed to save trajectory: %s", exc)


def execute_side_grasp_staged(move_group, gripper, bottleneck_pose, grasp_pose,
                              trajectory_recorder=None,
                              trajectory_record_path="",
                              target_collision_scene=None,
                              target_collision_name=None):
    """Execute side grasp with staged approach instead of full replay IK."""
    rospy.loginfo("MT3 side grasp staged executor enabled")
    tf_listener = tf.TransformListener()
    try:
        gripper.open()
        rospy.sleep(0.5)
    except Exception:
        pass

    current = move_group.get_current_pose().pose
    demo_dx = grasp_pose.position.x - bottleneck_pose.position.x
    demo_dy = grasp_pose.position.y - bottleneck_pose.position.y
    side_approach_axis = "y" if abs(demo_dy) > abs(demo_dx) else "x"
    side_approach_sign = 1.0
    if side_approach_axis == "x":
        if demo_dx < 0.0:
            side_approach_sign = -1.0
        lateral_axis = "y"
    else:
        if demo_dy < 0.0:
            side_approach_sign = -1.0
        lateral_axis = "x"
    rospy.loginfo(
        "Side staged approach axis selected from demo delta: axis=%s sign=%+.0f "
        "demo_delta=[%.3f, %.3f]",
        side_approach_axis, side_approach_sign, demo_dx, demo_dy)
    side_tcp_forward_offset = float(rospy.get_param(
        '/sawyer_auto_grasp/side_tcp_forward_offset', 0.0))
    side_fingertip_clearance = float(rospy.get_param(
        '/sawyer_auto_grasp/side_fingertip_clearance', 0.035))
    object_x = float(rospy.get_param(
        '/sawyer_auto_grasp/object_base_x',
        rospy.get_param('/sawyer_auto_grasp/object_x', grasp_pose.position.x)))
    object_y = float(rospy.get_param(
        '/sawyer_auto_grasp/object_base_y',
        rospy.get_param('/sawyer_auto_grasp/object_y', grasp_pose.position.y)))
    object_size = rospy.get_param(
        '/sawyer_auto_grasp/object_size', [0.045, 0.045, 0.045])
    object_radius = 0.025
    try:
        if object_size and len(object_size) >= 2:
            object_radius = 0.5 * max(abs(float(object_size[0])),
                                      abs(float(object_size[1])))
            object_radius = max(0.020, min(0.070, object_radius))
    except Exception:
        object_radius = 0.025

    side_final_x_extra = float(rospy.get_param(
        '/sawyer_auto_grasp/side_final_x_extra', 0.0))
    side_final_z_bias = float(rospy.get_param(
        '/sawyer_auto_grasp/side_final_z_bias', 0.0))
    if abs(side_final_x_extra) > 1e-6 or abs(side_final_z_bias) > 1e-6:
        original_grasp = copy.deepcopy(grasp_pose)
        grasp_pose = copy.deepcopy(grasp_pose)
        if side_approach_axis == "x":
            grasp_pose.position.x += side_approach_sign * side_final_x_extra
        else:
            grasp_pose.position.y += side_approach_sign * side_final_x_extra
        grasp_pose.position.z += side_final_z_bias
        rospy.loginfo(
            "Side final contact refinement along %s: pos %.3f -> %.3f "
            "(extra=%+.3f, radius=%.3f), z %.3f -> %.3f (bias=%+.3f)",
            side_approach_axis,
            (original_grasp.position.x if side_approach_axis == "x" else original_grasp.position.y),
            (grasp_pose.position.x if side_approach_axis == "x" else grasp_pose.position.y),
            side_approach_sign * side_final_x_extra, object_radius,
            original_grasp.position.z, grasp_pose.position.z,
            side_final_z_bias)

    object_approach_coord = object_x if side_approach_axis == "x" else object_y
    grasp_approach_coord = (
        grasp_pose.position.x if side_approach_axis == "x"
        else grasp_pose.position.y)
    bottleneck_approach_coord = (
        bottleneck_pose.position.x if side_approach_axis == "x"
        else bottleneck_pose.position.y)
    contact_at_grasp = (
        grasp_approach_coord + side_approach_sign * side_tcp_forward_offset)
    safe_contact_coord = (
        object_approach_coord - side_approach_sign *
        (object_radius + side_fingertip_clearance))
    fingertip_safe_retreat = abs(contact_at_grasp - safe_contact_coord)
    configured_retreat = float(rospy.get_param(
        '/sawyer_auto_grasp/side_min_entry_retreat', -1.0))
    current_retreat = abs(grasp_approach_coord - bottleneck_approach_coord)
    demo_retreat = max(current_retreat, 0.0)
    if configured_retreat > 0.0:
        side_min_entry_retreat = max(configured_retreat, fingertip_safe_retreat)
    else:
        # Prefer the bottleneck-to-grasp retreat that came from the mapped demo.
        # Only expand it if the current fingertip geometry would otherwise hit
        # the object before the lateral/z alignment is complete.
        side_min_entry_retreat = max(demo_retreat, fingertip_safe_retreat)
    rospy.loginfo(
        "Side fingertip safety: axis=%s object_coord=%.3f radius=%.3f clearance=%.3f "
        "contact_grasp=%.3f safe_contact=%.3f min_retreat=%.3f",
        side_approach_axis, object_approach_coord, object_radius,
        side_fingertip_clearance, contact_at_grasp, safe_contact_coord,
        side_min_entry_retreat)
    rospy.loginfo(
        "Side retreat source: demo_retreat=%.3f fingertip_safe=%.3f "
        "configured=%.3f -> used=%.3f",
        demo_retreat, fingertip_safe_retreat,
        configured_retreat, side_min_entry_retreat)
    if current_retreat < side_min_entry_retreat:
        old_coord = bottleneck_approach_coord
        bottleneck_pose = copy.deepcopy(bottleneck_pose)
        new_coord = grasp_approach_coord - side_approach_sign * side_min_entry_retreat
        if side_approach_axis == "x":
            bottleneck_pose.position.x = new_coord
        else:
            bottleneck_pose.position.y = new_coord
        rospy.logwarn(
            "Side bottleneck %s retreated for safe lateral/z alignment: "
            "%.3f -> %.3f (retreat %.3f -> %.3f)",
            side_approach_axis, old_coord, new_coord,
            current_retreat, side_min_entry_retreat)

    stage_a = copy.deepcopy(bottleneck_pose)
    side_entry_clearance = rospy.get_param(
        '/sawyer_auto_grasp/side_entry_clearance', 0.08)
    side_entry_max_z = rospy.get_param(
        '/sawyer_auto_grasp/side_entry_max_z', 0.16)
    stage_a.position.z = min(
        max(current.position.z, grasp_pose.position.z + side_entry_clearance),
        side_entry_max_z)
    rospy.loginfo(
        "Side Step A entry z bounded: current=%.3f grasp=%.3f -> %.3f "
        "(clearance=%.3f max_z=%.3f)",
        current.position.z, grasp_pose.position.z, stage_a.position.z,
        side_entry_clearance, side_entry_max_z)
    entry_ok = execute_pose_target(
            move_group, stage_a, "Side Step A: safe lateral entry",
            velocity=0.12, acceleration=0.12, attempts=5, planning_time=10.0)
    stage_a_entry = copy.deepcopy(stage_a)
    if not entry_ok:
        rospy.logwarn(
            "Side Step A direct side-orientation entry failed; "
            "trying nearby lateral entry candidates.")
        for y_offset in [0.0, -0.02, 0.02, -0.04, 0.04]:
            candidate = copy.deepcopy(stage_a)
            candidate.position.y = stage_a.position.y + y_offset
            bridge = move_group.get_current_pose().pose
            bridge.position.x = candidate.position.x
            bridge.position.y = candidate.position.y
            bridge.position.z = candidate.position.z
            rospy.loginfo(
                "Side Step A candidate y_offset=%.3f target=[%.3f, %.3f, %.3f]",
                y_offset, candidate.position.x, candidate.position.y,
                candidate.position.z)
            if not execute_position_target(
                    move_group, bridge, "Side Step A bridge",
                    velocity=0.10, acceleration=0.10, attempts=8,
                    planning_time=12.0):
                continue
            if execute_pose_target(
                    move_group, candidate, "Side Step A bridge: rotate to side",
                    velocity=0.08, acceleration=0.08, attempts=8,
                    planning_time=12.0):
                stage_a_entry = candidate
                entry_ok = True
                break

        if not entry_ok:
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False

    current_after_entry = move_group.get_current_pose().pose
    entry_dx = current_after_entry.position.x - stage_a.position.x
    entry_dy = current_after_entry.position.y - stage_a.position.y
    if abs(entry_dx) > 0.014 or abs(entry_dy) > 0.012:
        rospy.logwarn(
            "Side Step A2 needed: actual entry is off by dx=%.1fcm dy=%.1fcm",
            entry_dx * 100.0, entry_dy * 100.0)
        stage_a_center = copy.deepcopy(current_after_entry)
        stage_a_center.position.x = stage_a.position.x
        stage_a_center.position.y = stage_a.position.y
        stage_a_center.position.z = stage_a.position.z
        stage_a_center.orientation = copy.deepcopy(stage_a.orientation)
        if not execute_incremental_cartesian_pose(
                move_group, stage_a_center,
                "Side Step A2: enforce safe entry xy",
                max_step=0.015, eef_step=0.006,
                accept_error=0.010, max_steps=24,
                step_accept_error=0.012, min_progress=0.002,
                axes=("x", "y")):
            rospy.logwarn(
                "Side Step A2 Cartesian xy correction failed; trying pose fallback.")
            if not execute_pose_target(
                    move_group, stage_a, "Side Step A2 fallback",
                    velocity=0.08, acceleration=0.08, attempts=5,
                    planning_time=10.0):
                stop_and_save_trajectory(
                    trajectory_recorder, trajectory_record_path, False)
                return False

    if not side_xy_ready(
            move_group, stage_a, "Side Step A ready check before lowering",
            accept_x=0.018, accept_y=0.014):
        stop_and_save_trajectory(
            trajectory_recorder, trajectory_record_path, False)
        return False

    object_base_z = float(rospy.get_param(
        '/sawyer_auto_grasp/object_base_z',
        rospy.get_param('/sawyer_auto_grasp/object_z', grasp_pose.position.z)))
    try:
        object_height = abs(float(object_size[2]))
    except Exception:
        object_height = 0.10
    side_grasp_height_ratio = float(rospy.get_param(
        '/sawyer_auto_grasp/side_grasp_height_fraction',
        rospy.get_param('/sawyer_auto_grasp/side_grasp_height_ratio', 0.55)))
    side_grasp_height_ratio = max(0.30, min(0.80, side_grasp_height_ratio))
    use_mouth_center_z = rospy.get_param(
        '/sawyer_auto_grasp/side_use_mouth_center_z', True)
    desired_grasp_z = (
        object_base_z + object_height * side_grasp_height_ratio
        if use_mouth_center_z else grasp_pose.position.z)

    mouth_state = get_gripper_mouth_state(tf_listener, move_group)
    mouth_offset = mouth_state["offset"]
    final_mouth_center = [float(object_x), float(object_y), float(desired_grasp_z)]
    precontact_clearance = float(rospy.get_param(
        '/sawyer_auto_grasp/side_precontact_clearance', 0.018))
    precontact_mouth_center = list(final_mouth_center)
    precontact_coord = (
        object_approach_coord -
        side_approach_sign * (object_radius + precontact_clearance))
    if side_approach_axis == "x":
        precontact_mouth_center[0] = precontact_coord
    else:
        precontact_mouth_center[1] = precontact_coord

    old_grasp_for_mouth = copy.deepcopy(grasp_pose)
    grasp_pose = command_pose_for_mouth_center(
        grasp_pose, final_mouth_center, mouth_offset)
    bottleneck_pose = command_pose_for_mouth_center(
        bottleneck_pose, precontact_mouth_center, mouth_offset)
    rospy.loginfo(
        "Side mouth-centered target rebuilt from finger TF: "
        "object_center=[%.3f, %.3f, %.3f] final_mouth=[%.3f, %.3f, %.3f] "
        "precontact_mouth=[%.3f, %.3f, %.3f] hand_offset=[%.3f, %.3f, %.3f] "
        "old_grasp=[%.3f, %.3f, %.3f] new_grasp=[%.3f, %.3f, %.3f]",
        object_x, object_y, desired_grasp_z,
        final_mouth_center[0], final_mouth_center[1], final_mouth_center[2],
        precontact_mouth_center[0], precontact_mouth_center[1],
        precontact_mouth_center[2],
        mouth_offset[0], mouth_offset[1], mouth_offset[2],
        old_grasp_for_mouth.position.x, old_grasp_for_mouth.position.y,
        old_grasp_for_mouth.position.z,
        grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z)
    _, _, entry_finger_clearance = log_side_mouth_check(
        tf_listener, move_group, "Side Step A mouth geometry",
        final_mouth_center, object_radius, lateral_axis)

    # Do not lower all the way while the gripper is still far from the object.
    # Pick the final approach axis from the mapped demo:
    # x-side grasp: align y/z first, then approach along x.
    # y-side grasp: align x/z first, then approach along y.
    grasp_approach_coord = (
        grasp_pose.position.x if side_approach_axis == "x"
        else grasp_pose.position.y)
    bottleneck_approach_coord = (
        bottleneck_pose.position.x if side_approach_axis == "x"
        else bottleneck_pose.position.y)
    precontact_retreat = abs(grasp_approach_coord - bottleneck_approach_coord)
    current_after_entry = move_group.get_current_pose().pose
    precontact_pose = copy.deepcopy(current_after_entry)
    precontact_pose.position.x = bottleneck_pose.position.x
    precontact_pose.position.y = bottleneck_pose.position.y
    precontact_pose.orientation = copy.deepcopy(grasp_pose.orientation)
    rospy.loginfo(
        "Side Step B: high pre-approach before lowering along %s "
        "(retreat=%.1fcm, target=[%.3f, %.3f, %.3f])",
        side_approach_axis,
        precontact_retreat * 100.0,
        precontact_pose.position.x,
        precontact_pose.position.y,
        precontact_pose.position.z)
    pre_ok = execute_incremental_cartesian_pose(
            move_group, precontact_pose,
            "Side Step B: high lateral/approach pre-approach",
            max_step=0.012, eef_step=0.004,
            accept_error=0.018, max_steps=28,
            step_accept_error=0.014, min_progress=0.0002,
            axes=("x", "y"))
    if not pre_ok:
        current_pre = move_group.get_current_pose().pose
        pre_dx = abs(current_pre.position.x - precontact_pose.position.x)
        pre_dy = abs(current_pre.position.y - precontact_pose.position.y)
        if pre_dx <= 0.030 and pre_dy <= 0.022:
            rospy.logwarn(
                "Side Step B pre-approach not exact but close enough "
                "(dx=%.1fcm dy=%.1fcm); continuing.",
                pre_dx * 100.0, pre_dy * 100.0)
        else:
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False

    stage_c1 = move_group.get_current_pose().pose
    stage_c1.position.x = grasp_pose.position.x
    stage_c1.position.y = grasp_pose.position.y
    stage_c1.position.z = grasp_pose.position.z
    stage_c1.orientation = copy.deepcopy(grasp_pose.orientation)
    lower_axes = (lateral_axis, "z")
    lower_ok = execute_incremental_cartesian_pose(
            move_group, stage_c1,
            "Side Step C1: align lateral/z near object",
            max_step=0.018, eef_step=0.004,
            accept_error=0.024, max_steps=40,
            step_accept_error=0.016, min_progress=0.0005,
            axes=lower_axes)
    if not lower_ok:
        current_low = move_group.get_current_pose().pose
        low_lateral_error = (
            abs(current_low.position.y - grasp_pose.position.y)
            if lateral_axis == "y"
            else abs(current_low.position.x - grasp_pose.position.x))
        low_z_error = abs(current_low.position.z - grasp_pose.position.z)
        if low_lateral_error <= 0.024 and low_z_error <= 0.055:
            rospy.logwarn(
                "Side Step C1 lower not exact but inside side-grasp window "
                "(d%s=%.1fcm dz=%.1fcm); continuing to final %s.",
                lateral_axis, low_lateral_error * 100.0,
                low_z_error * 100.0, side_approach_axis)
        else:
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False

    current_before_final = move_group.get_current_pose().pose
    mouth_state, mouth_lateral_error, finger_clearance = log_side_mouth_check(
        tf_listener, move_group, "Side Step C1 mouth ready before final",
        final_mouth_center, object_radius, lateral_axis)
    if mouth_state.get("available"):
        lateral_error = abs(mouth_lateral_error)
        z_error = abs(mouth_state["center"][2] - final_mouth_center[2])
    else:
        lateral_error = (
            abs(current_before_final.position.y - grasp_pose.position.y)
            if lateral_axis == "y"
            else abs(current_before_final.position.x - grasp_pose.position.x))
        z_error = abs(current_before_final.position.z - grasp_pose.position.z)
    rospy.loginfo(
        "Side Step C1 ready check before final %s: mouth_d%s=%.1fcm dz=%.1fcm",
        side_approach_axis, lateral_axis, lateral_error * 100.0,
        z_error * 100.0)
    if lateral_error > 0.018 or z_error > 0.035:
        rospy.logerr(
            "Side Step C blocked: mouth center is not aligned enough "
            "for safe straight approach")
        stop_and_save_trajectory(
            trajectory_recorder, trajectory_record_path, False)
        return False
    if mouth_state.get("available") and finger_clearance < -0.004:
        rospy.logerr(
            "Side Step C blocked: object is outside opened finger corridor "
            "by %.1fcm", -finger_clearance * 100.0)
        stop_and_save_trajectory(
            trajectory_recorder, trajectory_record_path, False)
        return False

    remove_side_target_collision(
        target_collision_scene, target_collision_name,
        "Side Step C: before straight side approach")
    if not execute_incremental_cartesian_pose(
            move_group, grasp_pose,
            "Side Step C: straight %s approach" % side_approach_axis,
            max_step=0.010, eef_step=0.004,
            accept_error=0.018, max_steps=22,
            step_accept_error=0.012, min_progress=0.0005,
            axes=(side_approach_axis,)):
        if not side_close_enough_for_grasp(
                move_group, grasp_pose,
                "Side Step C close-window fallback",
                accept_x=0.030, accept_y=0.022,
                accept_z=0.035):
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False

    mouth_state, mouth_lateral_error, finger_clearance = log_side_mouth_check(
        tf_listener, move_group, "Side Step C final mouth before close",
        final_mouth_center, object_radius, lateral_axis)
    if mouth_state.get("available"):
        mouth_center = mouth_state["center"]
        approach_error = abs(
            _axis_value(mouth_center, side_approach_axis) -
            _axis_value(final_mouth_center, side_approach_axis))
        z_error = abs(mouth_center[2] - final_mouth_center[2])
        if (abs(mouth_lateral_error) > 0.022 or
                approach_error > 0.030 or z_error > 0.040):
            rospy.logerr(
                "Side Step D blocked: mouth center not at object before close "
                "(approach=%.1fcm lateral=%.1fcm z=%.1fcm)",
                approach_error * 100.0,
                abs(mouth_lateral_error) * 100.0,
                z_error * 100.0)
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False
        if finger_clearance < -0.004:
            rospy.logerr(
                "Side Step D blocked: finger corridor too narrow/misaligned "
                "by %.1fcm", -finger_clearance * 100.0)
            stop_and_save_trajectory(
                trajectory_recorder, trajectory_record_path, False)
            return False

    rospy.loginfo("Side Step D: close gripper")
    try:
        gripper.close()
        rospy.sleep(1.5)
    except Exception as exc:
        rospy.logwarn("Side gripper close failed: %s", exc)

    lift_pose = move_group.get_current_pose().pose
    lift_pose.position.z += 0.08
    lift_ok = execute_pose_target(
        move_group, lift_pose, "Side Step E: lift",
        velocity=0.10, acceleration=0.10, attempts=3, planning_time=8.0)
    rospy.sleep(1.0)
    stop_and_save_trajectory(
        trajectory_recorder, trajectory_record_path, bool(lift_ok))
    return bool(lift_ok)


def execute_demo_replay(move_group, gripper, bottleneck_pose, target_orientation,
                        replay_path, trajectory_recorder=None,
                        trajectory_record_path="",
                        target_collision_scene=None,
                        target_collision_name=None,
                        close_anchor_pose=None):
    """Reach aligned bottleneck, then replay retrieved demo end-effector motion."""
    rospy.set_param('/sawyer_auto_grasp/used_recovery_logic', False)
    rospy.set_param('/sawyer_auto_grasp/replay_recovery_progress', "")
    rospy.set_param('/sawyer_auto_grasp/replay_recovery_stage', "")
    rospy.set_param('/sawyer_auto_grasp/execution_variant', "standard_replay")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy', ["", ""])
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy', ["", ""])
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', "")
    tf_listener = tf.TransformListener()
    payload, velocities = load_demo_replay_velocities(replay_path)
    trajectory = payload.get("trajectory", payload)
    poses = trajectory.get("poses", [])
    replay_yaw_delta = float(payload.get("replay_yaw_delta_rad", 0.0) or 0.0)
    rospy.loginfo(
        "MT3 demo replay enabled: source=%s poses=%d velocities=%d yaw_delta=%.1fdeg",
        payload.get("source_demo", "unknown"), len(poses), len(velocities),
        math.degrees(replay_yaw_delta))
    if not velocities and len(poses) < 2:
        rospy.logerr("Demo replay trajectory is empty.")
        return False

    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
    move_group.set_goal_position_tolerance(0.008)
    move_group.set_goal_orientation_tolerance(0.06)

    transition = copy.deepcopy(bottleneck_pose)
    transition.position.x = bottleneck_pose.position.x
    transition.position.y = bottleneck_pose.position.y
    transition.position.z = bottleneck_pose.position.z + 0.10
    rospy.loginfo("Replay Step1: move to transition above bottleneck")
    move_group.set_pose_target(transition)
    if not move_group.go(wait=True):
        rospy.logerr("Replay transition planning failed.")
        return False
    rospy.sleep(0.5)

    rospy.loginfo("Replay Step2: reach aligned bottleneck pose")
    if not execute_incremental_cartesian_pose(
            move_group, bottleneck_pose,
            "Replay Step2: Cartesian lower to bottleneck",
            max_step=0.015, eef_step=0.004,
            accept_error=0.015, max_steps=24,
            step_accept_error=0.014, min_progress=0.002,
            axes=("z",)):
        rospy.logwarn(
            "Replay Cartesian bottleneck lower failed; trying slow pose fallback.")
        if not execute_pose_target(
                move_group, bottleneck_pose,
                "Replay Step2 fallback: reach bottleneck",
                velocity=0.06, acceleration=0.06,
                attempts=5, planning_time=12.0):
            rospy.logerr("Replay bottleneck planning failed.")
            return False
    rospy.sleep(0.5)
    try:
        gripper.open()
        rospy.sleep(0.5)
    except Exception:
        pass

    remove_side_target_collision(
        target_collision_scene, target_collision_name,
        "Replay Step3: before demo interaction")
    if trajectory_recorder is not None and trajectory_recorder._thread is None:
        trajectory_recorder.start()
        rospy.loginfo("End-effector trajectory recording to: %s", trajectory_record_path)

    prefer_pose_replay = rospy.get_param(
        '/sawyer_auto_grasp/prefer_pose_replay', True)
    start_pose = move_group.get_current_pose().pose
    if prefer_pose_replay and len(poses) >= 2:
        rospy.loginfo(
            "Replay Step3: use recorded pose-relative trajectory from bottleneck")
        anchor_close_waypoint = rospy.get_param(
            '/sawyer_auto_grasp/top_replay_anchor_close_waypoint', False)
        if close_anchor_pose is not None and anchor_close_waypoint:
            rospy.loginfo(
                "Replay Step3: close waypoint XY-anchored at grasp center "
                "[%.3f, %.3f], replay keeps demo z profile",
                close_anchor_pose.position.x,
                close_anchor_pose.position.y)
        elif close_anchor_pose is not None:
            experiment_group = str(rospy.get_param(
                '/sawyer_auto_grasp/experiment_group', '')).strip().lower()
            if experiment_group == "top_grasp":
                rospy.loginfo(
                    "Replay Step3: close waypoint anchoring disabled; "
                    "using bottleneck-relative replay and final mouth-center correction")
            else:
                rospy.loginfo(
                    "Replay Step3: close waypoint anchoring disabled; "
                    "using bottleneck-relative replay; top mouth-center correction disabled "
                    "for experiment_group=%s", experiment_group or "unset")
        waypoints, close_index = make_replay_waypoints_from_poses(
            start_pose, target_orientation, poses, velocities,
            yaw_delta=replay_yaw_delta,
            close_anchor_pose=(
                close_anchor_pose if anchor_close_waypoint else None))
    else:
        rospy.loginfo("Replay Step3: integrate demo velocities into Cartesian path")
        waypoints, close_index = make_replay_waypoints(
            start_pose, target_orientation, velocities)
    close_index = max(1, min(len(waypoints) - 1, close_index or len(waypoints) // 2))
    rospy.loginfo("Replay gripper close event at integrated waypoint %d", close_index)

    before_close = waypoints[:close_index + 1]
    after_close = waypoints[close_index + 1:]
    use_segmented_replay = rospy.get_param(
        '/sawyer_auto_grasp/use_segmented_replay', False)
    close_on_replay_blocked = rospy.get_param(
        '/sawyer_auto_grasp/close_on_replay_blocked', False)
    close_on_blocked_min_progress = float(rospy.get_param(
        '/sawyer_auto_grasp/replay_close_on_blocked_min_progress', 0.35))
    replay_blocked_before_close = False
    replay_progress_before_close = 0.0
    if use_segmented_replay:
        replay_ok, replay_progress_before_close = execute_cartesian_waypoint_segmented(
            move_group, before_close, "before gripper close",
            return_progress=True)
    else:
        replay_ok = execute_cartesian_waypoint_segment(
            move_group, before_close, "before gripper close",
            allow_partial=False)
    if not replay_ok:
        if (close_on_replay_blocked and close_anchor_pose is not None and
                replay_progress_before_close >= close_on_blocked_min_progress):
            replay_blocked_before_close = True
            replay_ok = True
            rospy.set_param('/sawyer_auto_grasp/used_recovery_logic', True)
            rospy.set_param('/sawyer_auto_grasp/replay_recovery_progress',
                            float(replay_progress_before_close))
            rospy.set_param('/sawyer_auto_grasp/replay_recovery_stage',
                            "before_gripper_close_blocked")
            rospy.set_param('/sawyer_auto_grasp/execution_variant',
                            "partial_replay_with_terminal_correction")
            rospy.logwarn(
                "Replay before gripper close blocked after %.1f%% progress; "
                "object is considered aligned enough, so closing gripper and lifting "
                "(min_progress=%.1f%%).",
                replay_progress_before_close * 100.0,
                close_on_blocked_min_progress * 100.0)
        else:
            rospy.logerr("Replay before gripper close failed; aborting replay grasp.")
            if trajectory_recorder is not None:
                try:
                    trajectory_recorder.stop()
                    saved_path = trajectory_recorder.save(
                        trajectory_record_path, success=False)
                    rospy.loginfo(
                        "End-effector trajectory saved: %s samples=%d success=False",
                        saved_path, len(trajectory_recorder.samples))
                except Exception as exc:
                    rospy.logwarn("Failed to save failed replay trajectory: %s", exc)
            return False

    if close_anchor_pose is not None:
        desired_xy = [
            rospy.get_param('/sawyer_auto_grasp/object_base_x',
                            close_anchor_pose.position.x),
            rospy.get_param('/sawyer_auto_grasp/object_base_y',
                            close_anchor_pose.position.y),
        ]
        correct_top_mouth_xy_before_close(
            tf_listener, move_group, "Replay before gripper close", desired_xy)
        record_before_close_mouth_xy(
            tf_listener, move_group, "Replay before gripper close", desired_xy)

    rospy.loginfo("Replay gripper close")
    try:
        gripper.close()
        rospy.sleep(1.0)
    except Exception as exc:
        rospy.logwarn("Replay gripper close failed: %s", exc)

    skip_after_close_replay = rospy.get_param(
        '/sawyer_auto_grasp/skip_after_close_replay', True)
    if skip_after_close_replay and after_close:
        rospy.set_param('/sawyer_auto_grasp/execution_variant',
                        "top_grasp_replay_to_close_then_direct_lift")
        rospy.logwarn(
            "Skipping after-close replay by test variant; "
            "top grasp will close at the aligned pose and use direct vertical lift.")
    elif after_close and not replay_blocked_before_close:
        if use_segmented_replay:
            replay_ok = execute_cartesian_waypoint_segmented(
                move_group, after_close, "after gripper close") and replay_ok
        else:
            replay_ok = execute_cartesian_waypoint_segment(
                move_group, after_close, "after gripper close",
                allow_partial=False) and replay_ok
    elif replay_blocked_before_close:
        rospy.logwarn(
            "Skipping remaining after-close replay because before-close replay "
            "was blocked and gripper was closed at the aligned pose.")
    else:
        try:
            rospy.sleep(0.5)
        except Exception:
            pass

    lift_pose = move_group.get_current_pose().pose
    lift_pose.position.z = max(lift_pose.position.z + 0.06,
                               bottleneck_pose.position.z)
    rospy.loginfo("Replay Step4: lift and stop")
    move_group.set_pose_target(lift_pose)
    lift_ok = move_group.go(wait=True)
    rospy.sleep(0.8)

    if trajectory_recorder is not None:
        try:
            trajectory_recorder.stop()
            saved_path = trajectory_recorder.save(
                trajectory_record_path, success=replay_ok)
            rospy.loginfo(
                "End-effector trajectory saved: %s samples=%d success=%s",
                saved_path, len(trajectory_recorder.samples), replay_ok)
        except Exception as exc:
            rospy.logwarn("Failed to save replay trajectory: %s", exc)

    if replay_ok or lift_ok:
        rospy.set_param('/sawyer_auto_grasp/keep_gripper_closed_on_exit', True)

    return bool(replay_ok)


def wait_for_robot_move_group(timeout=60):
    rospy.loginfo(f"等待{ROS_NAMESPACE}/move_group节点启动")
    start_time = rospy.get_time()
    while rospy.get_time() - start_time < timeout:
        try:
            result = subprocess.check_output(['rosnode', 'list'], stderr=subprocess.STDOUT)
            if f'{ROS_NAMESPACE}/move_group' in result.decode('utf-8'):
                rospy.loginfo(f"{ROS_NAMESPACE}/move_group节点已启动")
                return True
        except subprocess.CalledProcessError as e:
            rospy.logwarn(f"查询节点失败：{e.output}")
        rospy.sleep(1)
    rospy.logerr(f"超时，未找到{ROS_NAMESPACE}/move_group节点，请先启动对应launch文件")
    return False

# 核心抓取主函数
def auto_grasp_with_moveit():
    # 初始化MoveIt和ROS节点
    moveit_commander.roscpp_initialize([])
    rospy.init_node('sawyer_auto_grasp', anonymous=True)
    gripper = None
    robot_enabled = False
    trajectory_recorder = None
    rospy.set_param('/sawyer_auto_grasp/keep_gripper_closed_on_exit', False)

    # 从ROS参数读取MT3算出的完整抓取位姿
    # --- MT3 直接输出的抓取目标（完整6-DoF） ---
    grasp_x = rospy.get_param('/sawyer_auto_grasp/grasp_x', 0.60)
    grasp_y = rospy.get_param('/sawyer_auto_grasp/grasp_y', 0.00)
    grasp_z = rospy.get_param('/sawyer_auto_grasp/grasp_z', -0.58)
    grasp_qx = rospy.get_param('/sawyer_auto_grasp/grasp_qx', 1.0)
    grasp_qy = rospy.get_param('/sawyer_auto_grasp/grasp_qy', 0.0)
    grasp_qz = rospy.get_param('/sawyer_auto_grasp/grasp_qz', 0.0)
    grasp_qw = rospy.get_param('/sawyer_auto_grasp/grasp_qw', 0.0)
    bottleneck_x = rospy.get_param('/sawyer_auto_grasp/bottleneck_x', grasp_x)
    bottleneck_y = rospy.get_param('/sawyer_auto_grasp/bottleneck_y', grasp_y)
    bottleneck_z = rospy.get_param('/sawyer_auto_grasp/bottleneck_z', grasp_z + 0.15)
    bottleneck_qx = rospy.get_param('/sawyer_auto_grasp/bottleneck_qx', grasp_qx)
    bottleneck_qy = rospy.get_param('/sawyer_auto_grasp/bottleneck_qy', grasp_qy)
    bottleneck_qz = rospy.get_param('/sawyer_auto_grasp/bottleneck_qz', grasp_qz)
    bottleneck_qw = rospy.get_param('/sawyer_auto_grasp/bottleneck_qw', grasp_qw)

    # --- 物体在base帧的位置（用于计算安全过渡点） ---
    obj_base_x = rospy.get_param('/sawyer_auto_grasp/obj_base_x', grasp_x)
    obj_base_y = rospy.get_param('/sawyer_auto_grasp/obj_base_y', grasp_y)
    obj_base_z = rospy.get_param('/sawyer_auto_grasp/obj_base_z', grasp_z)

    # --- 兼容旧参数 ---
    object_base_x = rospy.get_param(
        '/sawyer_auto_grasp/object_base_x',
        rospy.get_param('/sawyer_auto_grasp/object_x', grasp_x))
    object_base_y = rospy.get_param(
        '/sawyer_auto_grasp/object_base_y',
        rospy.get_param('/sawyer_auto_grasp/object_y', grasp_y))
    object_base_z = rospy.get_param(
        '/sawyer_auto_grasp/object_base_z',
        rospy.get_param('/sawyer_auto_grasp/object_z', grasp_z))

    object_size  = rospy.get_param('/sawyer_auto_grasp/object_size', [0.045, 0.045, 0.045])
    gripper_opening = rospy.get_param('/sawyer_auto_grasp/gripper_opening',
                                       max(object_size[0], object_size[1]) + 0.02)
    trajectory_record_path = rospy.get_param('/sawyer_auto_grasp/trajectory_record_path', "")
    trajectory_record_rate = rospy.get_param('/sawyer_auto_grasp/trajectory_record_rate_hz', 10.0)
    use_demo_replay = rospy.get_param('/sawyer_auto_grasp/use_demo_replay', False)
    use_top_grasp_replay = rospy.get_param(
        '/sawyer_auto_grasp/use_top_grasp_replay', False)
    hold_on_success = rospy.get_param(
        '/sawyer_auto_grasp/hold_on_success', True)
    rospy.set_param('/sawyer_auto_grasp/replay_executed', False)
    rospy.set_param('/sawyer_auto_grasp/replay_type', "")
    rospy.set_param('/sawyer_auto_grasp/replay_fallback_used', False)
    grasp_mode = str(rospy.get_param('/sawyer_auto_grasp/grasp_mode', 'top')).strip().lower()
    demo_replay_trajectory_path = rospy.get_param(
        '/sawyer_auto_grasp/demo_replay_trajectory_path', "")
    # MT3 grasp_z is the semantic/contact grasp height. MoveIt controls the
    # right_hand flange, not the fingertips. In the current Sawyer Gazebo setup,
    # the visually correct contact height corresponds to a flange pose above
    # MT3 grasp_z. Commanding the raw MT3 z causes repeated CONTROL_FAILED
    # downward pushes; too large an offset produces shallow edge grasps.
    FLANGE_GRASP_Z_OFFSET = rospy.get_param('/sawyer_auto_grasp/flange_grasp_z_offset', 0.040)
    if grasp_mode == "side":
        side_z_offset = rospy.get_param('/sawyer_auto_grasp/side_flange_grasp_z_offset', 0.0)
        execution_grasp_z = grasp_z + side_z_offset
    else:
        side_z_offset = 0.0
        execution_grasp_z = grasp_z + FLANGE_GRASP_Z_OFFSET
    TCP_X_OFFSET = rospy.get_param('/sawyer_auto_grasp/tcp_x_offset', 0.0)
    TCP_Y_OFFSET = rospy.get_param('/sawyer_auto_grasp/tcp_y_offset', 0.0)
    flange_grasp_x = grasp_x + TCP_X_OFFSET
    flange_grasp_y = grasp_y + TCP_Y_OFFSET
    side_tcp_forward_offset = rospy.get_param(
        '/sawyer_auto_grasp/side_tcp_forward_offset', 0.0)
    side_approach_sign = 1.0
    if grasp_mode == "side":
        try:
            side_approach_sign = 1.0 if float(grasp_x) >= float(bottleneck_x) else -1.0
        except Exception:
            side_approach_sign = 1.0
        flange_grasp_x = (
            grasp_x - side_approach_sign * side_tcp_forward_offset +
            TCP_X_OFFSET
        )

    EDGE_Y_THRESHOLD = rospy.get_param('/sawyer_auto_grasp/edge_y_threshold', 0.08)
    edge_mode = abs(obj_base_y) >= EDGE_Y_THRESHOLD
    PREGRASP_CLEARANCE = rospy.get_param('/sawyer_auto_grasp/pregrasp_clearance', 0.025)
    EDGE_PREGRASP_EXTRA = rospy.get_param('/sawyer_auto_grasp/edge_pregrasp_extra', 0.015)
    if edge_mode:
        PREGRASP_CLEARANCE += EDGE_PREGRASP_EXTRA
    FINAL_DESCENT_STEP = rospy.get_param(
        '/sawyer_auto_grasp/final_descent_step',
        0.002 if edge_mode else 0.003)
    if grasp_mode == "side":
        rospy.loginfo(
            "Side grasp target: mt3=[%.3f, %.3f, %.3f] "
            "flange=[%.3f, %.3f, %.3f] tcp_forward=%.3f approach_sign=%+.0f",
            grasp_x, grasp_y, execution_grasp_z,
            flange_grasp_x, flange_grasp_y, execution_grasp_z,
            side_tcp_forward_offset, side_approach_sign)

    rospy.loginfo(f"MT3抓取位姿: x={grasp_x:.3f} y={grasp_y:.3f} z={grasp_z:.3f}")
    rospy.loginfo(f"MoveIt法兰执行Z: {execution_grasp_z:.3f} = MT3_Z + {FLANGE_GRASP_Z_OFFSET:.3f}")
    rospy.loginfo(
        f"MoveIt法兰XY目标: x={flange_grasp_x:.3f} y={flange_grasp_y:.3f} "
        f"tcp_offset=[{TCP_X_OFFSET:.3f},{TCP_Y_OFFSET:.3f}] edge_mode={edge_mode}"
    )
    rospy.loginfo(f"MT3抓取朝向: qx={grasp_qx:.3f} qy={grasp_qy:.3f} qz={grasp_qz:.3f} qw={grasp_qw:.3f}")
    rospy.loginfo(f"物体base位置: x={obj_base_x:.3f} y={obj_base_y:.3f} z={obj_base_z:.3f}")
    rospy.loginfo(f"物体尺寸={object_size} 夹爪开度={gripper_opening:.3f}")

    # --- 安全过渡点：从MT3抓取位姿推导（不再写死+1cm） ---
    # 过渡点 = 抓取位姿正上方15cm，用于安全接近
    SAFE_APPROACH_HEIGHT = 0.15  # 从抓取位姿往上15cm作为过渡
    transition_x = 0.50            # 前方过渡点x（远离桌面，避免碰撞）
    transition_z = grasp_z + 0.30  # 抓取位姿上方30cm过渡
    overhead_z = grasp_z + SAFE_APPROACH_HEIGHT  # 抓取位姿上方15cm = 准备下降
    grasp_target_z = execution_grasp_z

    # ====================== 1. 初始化连接与校验 ======================
    rospy.loginfo("初始化MoveGroup连接")
    if not wait_for_robot_move_group(timeout=60):
        return gripper, robot_enabled
    
    # 等待规划场景服务就绪
    service_name = f'{ROS_NAMESPACE}/get_planning_scene'
    rospy.loginfo(f"等待服务{service_name}就绪")
    try:
        rospy.wait_for_service(service_name, timeout=60)
        rospy.loginfo(f"服务{service_name}已就绪")
    except rospy.ROSException:
        rospy.logerr(f"超时，未找到服务{service_name}")
        return gripper, robot_enabled
    
    # 校验SRDF参数（适配命名空间）
    semantic_param = f"{ROS_NAMESPACE}/robot_description_semantic"
    if not rospy.has_param(semantic_param):
        rospy.logwarn(f"未找到{semantic_param}，仿真环境可忽略，继续执行")
    else:
        rospy.loginfo("SRDF参数加载成功")

    # 机器人使能（仿真环境可跳过，兼容实机）
    try:
        rs = RobotEnable()
        if not rs.state().enabled:
            rs.enable()
        robot_enabled = True
        rospy.loginfo("Robot enabled")
    except Exception as e:
        rospy.logwarn(f"RobotEnable警告: {e}，仿真环境可忽略")

    # ====================== 2. MoveGroup核心配置（Noetic专属适配） ======================
    # 初始化机器人对象（显式指定robot_description，适配命名空间）【完全未修改】
    robot = moveit_commander.RobotCommander(
        robot_description=f"{ROS_NAMESPACE}/robot_description",
        ns=ROS_NAMESPACE
    )
    # 验证可用规划组【完全未修改】
    available_groups = robot.get_group_names()
    rospy.loginfo(f"MoveIt可用规划组：{available_groups}")
    if PLANNING_GROUP not in available_groups:
        rospy.logerr(f"规划组{PLANNING_GROUP}不存在，可用组：{available_groups}")
        return gripper, robot_enabled
    
    # 初始化规划组【完全未修改】
    move_group = moveit_commander.MoveGroupCommander(
        PLANNING_GROUP,
        robot_description=f"{ROS_NAMESPACE}/robot_description",
        ns=ROS_NAMESPACE
    )
    move_group.set_end_effector_link(END_EFFECTOR_LINK)
    install_moveit_timing(move_group)
    move_group.set_pose_reference_frame("base")  # 全程base坐标系
    # 打印当前配置（调试用）【完全未修改】
    rospy.loginfo(f"当前规划帧：{move_group.get_planning_frame()}")
    rospy.loginfo(f"当前位姿参考帧：{move_group.get_pose_reference_frame()}")
    rospy.loginfo(f"末端执行器link：{move_group.get_end_effector_link()}")

    # 【优化】规划核心参数：从保守→高效
    move_group.allow_replanning(False)  # 【关键】关闭重规划，固定点位不需要
    move_group.set_planning_time(5.0)    # 从30.0→5.0，大幅减少规划等待
    move_group.set_num_planning_attempts(2) # 从15→2，最多试2次
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
    # 目标容差保持放宽后的配置
    move_group.set_goal_position_tolerance(0.005)
    move_group.set_goal_orientation_tolerance(0.02)
    move_group.set_workspace([0.1, -0.45, -0.8, 1.2, 0.45, 0.5])
    planning_scene = moveit_commander.PlanningSceneInterface(ns=ROS_NAMESPACE)
    tf_listener = tf.TransformListener()

    # 获取运动关节列表【完全未修改】
    joint_names = [jn for jn in robot.get_joint_names(PLANNING_GROUP) if jn in JOINT_LIMITS.keys()]
    rospy.loginfo(f"运动关节：{joint_names}")

    # ====================== 3. 初始姿态修正（避碰撞+留抓取空间） ======================【完全未修改】
    rospy.loginfo("修正初始姿态")
    current_joints = move_group.get_current_joint_values()
    min_len = min(len(current_joints), len(joint_names))
    safe_joints = dict(zip(joint_names[:min_len], current_joints[:min_len]))
    # 初始姿态优化：J1向下偏，为低位抓取预留空间
    safe_joints['right_j0'] = clamp_joint_value('right_j0', 0.0)
    safe_joints['right_j1'] = clamp_joint_value('right_j1', -0.8)
    safe_joints['right_j2'] = clamp_joint_value('right_j2', 0.0)
    safe_joints['right_j3'] = clamp_joint_value('right_j3', 1.8)
    safe_joints['right_j4'] = clamp_joint_value('right_j4', 0.0)
    safe_joints['right_j5'] = clamp_joint_value('right_j5', 0.0)
    safe_joints['right_j6'] = clamp_joint_value('right_j6', 0.0)
    
    move_group.set_joint_value_target(safe_joints)
    plan_success = False
    retry_count = 0
    while not plan_success and retry_count < 2: # 【优化】重试从3→2
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        if not plan_success:
            rospy.logwarn(f"初始姿态规划重试{retry_count+1}/2")
            safe_joints['right_j1'] += 0.1
            move_group.set_joint_value_target(safe_joints)
            retry_count += 1
    if plan_success:
        move_group.execute(plan, wait=True)
        rospy.sleep(0.5) # 【优化】从2→0.5
        rospy.loginfo("初始化已完成")
    else:
        rospy.logwarn("初始姿态修正失败，使用备用安全态")
        backup_safe_joints = {
            'right_j0':0.2, 'right_j1':-0.6, 'right_j2':0.3,
            'right_j3':1.5, 'right_j4':0.1, 'right_j5':0.2, 'right_j6':0.0
        }
        for jn in backup_safe_joints.keys():
            backup_safe_joints[jn] = clamp_joint_value(jn, backup_safe_joints[jn])
        move_group.set_joint_value_target(backup_safe_joints)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        if plan_success:
            move_group.execute(plan, wait=True)
            rospy.sleep(0.5) # 【优化】从2→0.5
            rospy.loginfo("初始化已完成")
        else:
            rospy.logerr("安全态规划失败，程序退出")
            return gripper, robot_enabled

    # ====================== 4. 夹爪初始化（朝下姿态+校准） ======================
    try:
        gripper = Gripper('right_gripper')
        if not gripper.is_calibrated():
            gripper.calibrate()
            rospy.sleep(1) # 【优化】从2→1
        gripper.set_cmd_velocity(0.1) # 【优化】从0.03→0.1，夹爪也快点
        gripper.open()
        if trajectory_record_path:
            trajectory_recorder = EndEffectorTrajectoryRecorder(
                move_group, gripper=gripper, rate_hz=trajectory_record_rate)
            trajectory_recorder.start()
            rospy.loginfo(f"End-effector trajectory recording to: {trajectory_record_path}")
        rospy.loginfo("抓夹已打开")
        rospy.sleep(0.5) # 【优化】从1→0.5
    except Exception as e:
        rospy.logerr(f"夹爪初始化失败：{e}")
        return gripper, robot_enabled

    # ====================== 5. 使用MT3算出的抓取朝向 ======================
    target_pose = geometry_msgs.msg.Pose()
    # 默认用夹爪朝下（兼容demo未提供朝向的情况）
    target_pose.orientation.x = grasp_qx
    target_pose.orientation.y = grasp_qy
    target_pose.orientation.z = grasp_qz
    target_pose.orientation.w = grasp_qw
    target_pose.position.x = flange_grasp_x
    target_pose.position.y = flange_grasp_y
    target_pose.position.z = execution_grasp_z
    rospy.loginfo(f"抓取朝向使用MT3结果: [{grasp_qx:.3f}, {grasp_qy:.3f}, "
                  f"{grasp_qz:.3f}, {grasp_qw:.3f}]")

    # 轨迹显示发布器（RVIZ可视化）【完全未修改】
    display_pub = rospy.Publisher(
        '/move_group/display_planned_path', 
        moveit_msgs.msg.DisplayTrajectory, 
        queue_size=10
    )
    display_traj = moveit_msgs.msg.DisplayTrajectory()
    display_traj.trajectory_start = robot.get_current_state()

    target_collision_scene = None
    target_collision_name = None
    if (grasp_mode == "side" and use_demo_replay and
            demo_replay_trajectory_path and os.path.exists(demo_replay_trajectory_path)):
        target_collision_scene = planning_scene
        target_collision_name = add_side_target_collision(target_collision_scene)

    if use_demo_replay and demo_replay_trajectory_path and os.path.exists(demo_replay_trajectory_path):
        bottleneck_pose = geometry_msgs.msg.Pose()
        bottleneck_pose.position.x = bottleneck_x
        bottleneck_pose.position.y = bottleneck_y
        replay_bottleneck_z_offset = rospy.get_param(
            '/sawyer_auto_grasp/replay_bottleneck_z_offset', 0.0)
        bottleneck_pose.position.z = bottleneck_z + replay_bottleneck_z_offset
        bottleneck_pose.orientation.x = bottleneck_qx
        bottleneck_pose.orientation.y = bottleneck_qy
        bottleneck_pose.orientation.z = bottleneck_qz
        bottleneck_pose.orientation.w = bottleneck_qw
        rospy.loginfo(
            "MT3 replay bottleneck MoveIt Z: %.3f = bottleneck_z %.3f + replay_offset %.3f",
            bottleneck_pose.position.z, bottleneck_z, replay_bottleneck_z_offset)
        use_side_staged_replay = rospy.get_param(
            '/sawyer_auto_grasp/use_side_staged_replay', False)
        should_run_replay = (grasp_mode == "side" or bool(use_top_grasp_replay))
        if should_run_replay:
            if grasp_mode == "side" and use_side_staged_replay:
                bottleneck_pose = side_contact_pose_to_flange_pose(
                    bottleneck_pose, side_approach_sign,
                    side_tcp_forward_offset, "Side bottleneck")
                replay_success = execute_side_grasp_staged(
                    move_group, gripper, bottleneck_pose, target_pose,
                    trajectory_recorder=trajectory_recorder,
                    trajectory_record_path=trajectory_record_path,
                    target_collision_scene=target_collision_scene,
                    target_collision_name=target_collision_name)
            else:
                if grasp_mode == "side":
                    rospy.loginfo(
                        "Side grasp uses direct demo replay "
                        "(set /sawyer_auto_grasp/use_side_staged_replay:=true "
                        "to use the experimental staged executor).")
                else:
                    rospy.loginfo(
                        "Top/rotated top grasp demo replay is enabled; "
                        "scripted top grasp remains the fallback.")
                replay_success = execute_demo_replay(
                    move_group, gripper, bottleneck_pose, target_pose.orientation,
                    demo_replay_trajectory_path,
                    trajectory_recorder=trajectory_recorder,
                    trajectory_record_path=trajectory_record_path,
                    target_collision_scene=target_collision_scene,
                    target_collision_name=target_collision_name,
                    close_anchor_pose=(None if grasp_mode == "side" else target_pose))
            if not replay_success:
                remove_side_target_collision(
                    target_collision_scene, target_collision_name,
                    "Replay failed cleanup")
                if grasp_mode == "side":
                    raise RuntimeError("MT3 demo replay failed; marking grasp execution failed")
                rospy.set_param('/sawyer_auto_grasp/replay_fallback_used', True)
                fallback_to_script = rospy.get_param(
                    '/sawyer_auto_grasp/fallback_to_scripted_grasp', False)
                if not fallback_to_script:
                    rospy.logerr(
                        "Top/rotated top grasp replay failed; scripted fallback disabled.")
                    raise RuntimeError(
                        "Top/rotated top grasp replay failed; scripted fallback disabled")
                rospy.logwarn(
                    "Top/rotated top grasp replay failed; falling back to scripted grasp.")
                if trajectory_recorder is not None:
                    try:
                        trajectory_recorder.stop()
                    except Exception:
                        pass
                    trajectory_recorder = EndEffectorTrajectoryRecorder(
                        move_group, gripper=gripper, rate_hz=trajectory_record_rate)
                    trajectory_recorder.start()
            else:
                rospy.set_param('/sawyer_auto_grasp/replay_executed', True)
                replay_type = (
                    "side_staged_replay"
                    if grasp_mode == "side" and use_side_staged_replay
                    else ("side_demo_replay" if grasp_mode == "side"
                          else "top_grasp_demo_replay")
                )
                rospy.set_param('/sawyer_auto_grasp/replay_type', replay_type)
                return gripper, robot_enabled
        else:
            rospy.loginfo(
                "Demo replay input is available, but top grasp replay is disabled; "
                "using scripted top grasp.")
    elif use_demo_replay:
        rospy.logwarn(
            "Demo replay requested but trajectory file is missing: %s; "
            "falling back to scripted grasp",
            demo_replay_trajectory_path)

    # ====================== Step1: 初始态 → 远距过渡点（避奇点） ======================
    rospy.loginfo(f"Step1 过渡点")
    transition_pose = copy.deepcopy(target_pose)
    transition_pose.position.x = transition_x
    transition_pose.position.y = flange_grasp_y
    transition_pose.position.z = transition_z
    move_group.set_pose_target(transition_pose)
    
    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    planning_time = plan_result[2]
    error_msg = plan_result[3]
    retry_count = 0
    while not plan_success and retry_count < 2: # 【优化】重试从3→2
        transition_pose.position.z += 0.05
        move_group.set_pose_target(transition_pose)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        planning_time = plan_result[2]
        error_msg = plan_result[3]
        retry_count += 1
    if not plan_success:
        rospy.logerr(f"过渡点规划失败: {error_msg}")
        return gripper, robot_enabled
    
    rospy.loginfo(f"过渡点规划成功，耗时{planning_time:.2f}s")
    display_traj.trajectory.append(plan)
    display_pub.publish(display_traj)
    move_group.execute(plan, wait=True)
    rospy.sleep(0.5) # 【优化】从(len(...) + 2)→0.5，大幅缩短

    # ====================== Step2: 移动到物块上方安全高度 ======================
    rospy.loginfo(f"Step2 移动至物块上方")
    overhead_pose = copy.deepcopy(transition_pose)
    overhead_pose.position.x = flange_grasp_x
    overhead_pose.position.y = flange_grasp_y
    overhead_pose.position.z = overhead_z
    move_group.set_pose_target(overhead_pose)
    
    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    planning_time = plan_result[2]
    error_msg = plan_result[3]
    retry_count = 0
    while not plan_success and retry_count < 3: # 【优化】重试从5→3
        rospy.logwarn(f"高度规划重试{retry_count+1}/3，抬高目标高度")
        overhead_pose.position.z += 0.05
        move_group.set_pose_target(overhead_pose)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        planning_time = plan_result[2]
        error_msg = plan_result[3]
        retry_count += 1
    
    if plan_success:
        display_traj.trajectory[0] = plan
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        rospy.sleep(0.5) # 【优化】从2→0.5
        rospy.loginfo("到达物块上方安全高度")
    else:
        rospy.logwarn(f"安全高度移动失败，直接水平对齐: {error_msg}")
        overhead_pose = transition_pose

    # ====================== Step3: XY精准对齐 ======================
    rospy.loginfo(f"Step3 目标x={flange_grasp_x} y={flange_grasp_y}")
    start_pose = move_group.get_current_pose().pose
    target_align_pose = copy.deepcopy(start_pose)
    target_align_pose.position.x = flange_grasp_x
    target_align_pose.position.y = flange_grasp_y
    target_align_pose.position.z = overhead_z
    target_align_pose.orientation = copy.deepcopy(target_pose.orientation)

    # 【优化】直接规划一次笛卡尔路径，砍掉闭环重试
    waypoints = [start_pose, target_align_pose]
    (plan, fraction) = move_group.compute_cartesian_path(
        waypoints,
        CART_STEP,
        True
    )
    
    final_align = None
    if fraction >= 0.9:
        rospy.loginfo(f"笛卡尔路径规划成功，成功率{fraction*100:.1f}%")
        display_traj.trajectory.append(plan)
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        rospy.sleep(0.5) # 【优化】缩短等待
        final_align = move_group.get_current_pose().pose
        current_x = final_align.position.x
        current_y = final_align.position.y
        x_error = abs(current_x - flange_grasp_x)
        y_error = abs(current_y - flange_grasp_y)
        rospy.loginfo(f"对齐完成 x={current_x:.3f} 误差{x_error:.4f}m y={current_y:.3f} 误差{y_error:.4f}m")
    else:
        rospy.logwarn(f"笛卡尔规划成功率{fraction*100:.1f}%，改用关节空间规划")
        move_group.set_pose_target(target_align_pose)
        plan_result = move_group.plan()
        if plan_result[0]:
            move_group.execute(plan_result[1], wait=True)
            rospy.sleep(0.5)
            final_align = move_group.get_current_pose().pose
            rospy.loginfo("关节空间对齐完成")
        else:
            rospy.logerr("对齐失败，程序退出")
            return gripper, robot_enabled

    # ====================== Step4: 下降到MT3抓取位姿（笛卡尔直线） ======================
    # Extra XY correction. The MT3 run can stop with 5-8mm X residual, which
    # turns a top grasp into an edge grasp. Keep this correction horizontal.
    for xy_retry in range(3):
        final_align = move_group.get_current_pose().pose
        x_error = abs(final_align.position.x - flange_grasp_x)
        y_error = abs(final_align.position.y - flange_grasp_y)
        if x_error <= ALLOWED_ERROR and y_error <= ALLOWED_ERROR:
            break

        rospy.logwarn(
            f"XY residual before descent: x={x_error:.4f}m y={y_error:.4f}m; "
            f"correction {xy_retry+1}/3"
        )
        correction_pose = copy.deepcopy(final_align)
        correction_pose.position.x = flange_grasp_x
        correction_pose.position.y = flange_grasp_y
        correction_pose.orientation = copy.deepcopy(target_pose.orientation)
        correction_plan, correction_fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(final_align), copy.deepcopy(correction_pose)],
            0.003,
            True
        )
        if correction_fraction >= 0.98 and len(correction_plan.joint_trajectory.points) > 0:
            move_group.execute(correction_plan, wait=True)
            rospy.sleep(0.3)
        else:
            rospy.logwarn(f"XY correction planning insufficient: fraction={correction_fraction*100:.1f}%")
            break

    final_align = move_group.get_current_pose().pose
    rospy.loginfo(
        f"XY final before descent x={final_align.position.x:.3f} "
        f"err={abs(final_align.position.x-flange_grasp_x):.4f}m "
        f"y={final_align.position.y:.3f} "
        f"err={abs(final_align.position.y-flange_grasp_y):.4f}m"
    )

    rospy.loginfo(f"Step4 下降至MT3抓取位姿 z={grasp_target_z:.3f}")

    # 构造MT3算出的完整抓取目标位姿
    grasp_pose = geometry_msgs.msg.Pose()
    grasp_pose.position.x = flange_grasp_x
    grasp_pose.position.y = flange_grasp_y
    grasp_pose.position.z = grasp_target_z
    grasp_pose.orientation.x = grasp_qx
    grasp_pose.orientation.y = grasp_qy
    grasp_pose.orientation.z = grasp_qz
    grasp_pose.orientation.w = grasp_qw

    # 下降阶段用更宽松的参数
    move_group.set_max_velocity_scaling_factor(DOWN_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(DOWN_ACC_SCALE)
    move_group.set_goal_position_tolerance(0.01)
    move_group.set_goal_orientation_tolerance(0.05)

    descent_success = False
    final_pose_after_descent = None

    # Move to a low pregrasp pose first, then re-center XY before the final
    # short descent. This reduces side-finger bumps at workspace edges without
    # changing the perceived object pose.
    pregrasp_pose = move_group.get_current_pose().pose
    pregrasp_goal = copy.deepcopy(pregrasp_pose)
    pregrasp_goal.position.x = flange_grasp_x
    pregrasp_goal.position.y = flange_grasp_y
    pregrasp_goal.position.z = grasp_target_z + PREGRASP_CLEARANCE
    pregrasp_goal.orientation = copy.deepcopy(grasp_pose.orientation)
    rospy.loginfo(
        f"  pregrasp recenter: z={pregrasp_goal.position.z:.3f} "
        f"clearance={PREGRASP_CLEARANCE:.3f} step={FINAL_DESCENT_STEP:.3f}"
    )
    pregrasp_plan, pregrasp_fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(pregrasp_pose), copy.deepcopy(pregrasp_goal)],
        0.003,
        True
    )
    if pregrasp_fraction >= 0.98 and len(pregrasp_plan.joint_trajectory.points) > 0:
        move_group.execute(pregrasp_plan, wait=True)
        rospy.sleep(0.3)
    else:
        rospy.logwarn(f"  pregrasp planning insufficient: fraction={pregrasp_fraction*100:.1f}%")

    for retry in range(3):
        descend_start = move_group.get_current_pose().pose
        descend_goal = copy.deepcopy(descend_start)
        descend_goal.position.x = flange_grasp_x
        descend_goal.position.y = flange_grasp_y
        descend_goal.position.z = grasp_target_z
        descend_goal.orientation = copy.deepcopy(grasp_pose.orientation)
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(descend_start), copy.deepcopy(descend_goal)],
            FINAL_DESCENT_STEP,
            True
        )
        if fraction >= 0.98 and len(plan.joint_trajectory.points) > 0:
            move_group.execute(plan, wait=True)
            rospy.sleep(0.3)
            final_pose_after_descent = move_group.get_current_pose().pose
            actual_z = final_pose_after_descent.position.z
            z_error = abs(actual_z - grasp_target_z)
            rospy.loginfo(f"  笛卡尔下降: 目标z={grasp_target_z:.3f} 实际z={actual_z:.3f} 误差={z_error*100:.1f}cm")
            if z_error < 0.05:  # 5cm内算成功
                descent_success = True
                rospy.loginfo(f"  下降成功！")
                break
            else:
                rospy.logwarn(f"  重试{retry+1}/3 z偏差{z_error*100:.0f}cm")
        else:
            rospy.logwarn(f"  重试{retry+1}/3 笛卡尔下降规划不足，成功率{fraction*100:.1f}%")
            grasp_pose.position.z += 0.02

    # 兜底：yaw 过的长方体抓取在边缘位置可能无法得到 98% 的笛卡尔下降，
    # 但录制 demo 时小步 pose-target 下降可以稳定到达夹取高度。
    if not descent_success:
        rospy.logwarn("  笛卡尔下降失败，尝试小步位姿下降")
        best_error = float("inf")
        stalled_steps = 0
        for step_idx in range(18):
            current_pose = move_group.get_current_pose().pose
            current_error = abs(current_pose.position.z - grasp_target_z)
            if current_error < 0.050:
                final_pose_after_descent = current_pose
                descent_success = True
                rospy.loginfo(
                    f"  小步下降成功: z={current_pose.position.z:.3f} "
                    f"误差={current_error*100:.1f}cm"
                )
                break

            remaining = grasp_target_z - current_pose.position.z
            dz = max(-0.010, min(0.010, remaining))
            step_goal = copy.deepcopy(grasp_pose)
            step_goal.position.x = flange_grasp_x
            step_goal.position.y = flange_grasp_y
            step_goal.position.z = current_pose.position.z + dz
            step_goal.orientation = copy.deepcopy(grasp_pose.orientation)

            move_group.set_pose_target(step_goal)
            ok = move_group.go(wait=True)
            rospy.sleep(0.25)

            actual_pose = move_group.get_current_pose().pose
            error = abs(actual_pose.position.z - grasp_target_z)
            rospy.loginfo(
                f"  小步下降{step_idx+1:02d}: ok={ok} "
                f"actual_z={actual_pose.position.z:.3f} "
                f"target_z={grasp_target_z:.3f} 误差={error*100:.1f}cm"
            )

            if error < best_error - 0.002:
                best_error = error
                stalled_steps = 0
            else:
                stalled_steps += 1

            if error < 0.050:
                final_pose_after_descent = actual_pose
                descent_success = True
                rospy.loginfo("  小步下降到达可夹取高度")
                break

            if stalled_steps >= 3 and best_error <= 0.060:
                final_pose_after_descent = actual_pose
                descent_success = True
                rospy.logwarn(
                    f"  小步下降高度接近且停滞，误差={best_error*100:.1f}cm；"
                    "接受当前位置避免继续扭腕"
                )
                break

    # 兜底：如果笛卡尔/小步下降都失败，用当前位姿尝试
    if not descent_success:
        rospy.logwarn("  笛卡尔下降全部失败，以当前高度尝试抓取")
        final_pose_after_descent = move_group.get_current_pose().pose

    # 恢复参数
    move_group.set_goal_position_tolerance(0.005)
    move_group.set_goal_orientation_tolerance(0.02)
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
    move_group.set_num_planning_attempts(2)
    move_group.set_planning_time(5.0)

    rospy.loginfo(f"  最终下降高度 z={final_pose_after_descent.position.z:.3f}")

    # ====================== Step5: 抓取（智能判定） ======================
    # Final descent gate: do not close the gripper if the flange stopped high.
    descent_z_error = abs(final_pose_after_descent.position.z - grasp_target_z)
    if descent_z_error > 0.070:
        rospy.logwarn(
            f"  Step4 final z error is {descent_z_error*100:.1f}cm; "
            f"trying one fine descent before grasp"
        )
        fine_tune_pose = copy.deepcopy(final_pose_after_descent)
        fine_tune_pose.position.x = flange_grasp_x
        fine_tune_pose.position.y = flange_grasp_y
        fine_tune_pose.position.z = grasp_target_z
        fine_tune_pose.orientation = copy.deepcopy(grasp_pose.orientation)
        fine_plan, fine_fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(final_pose_after_descent), copy.deepcopy(fine_tune_pose)],
            0.003,
            True
        )
        if fine_fraction >= 0.98 and len(fine_plan.joint_trajectory.points) > 0:
            move_group.execute(fine_plan, wait=True)
            rospy.sleep(0.5)
            final_pose_after_descent = move_group.get_current_pose().pose
            descent_z_error = abs(final_pose_after_descent.position.z - grasp_target_z)
            rospy.loginfo(
                f"  fine descent result: target_z={grasp_target_z:.3f}, "
                f"actual_z={final_pose_after_descent.position.z:.3f}, "
                f"error={descent_z_error*100:.1f}cm"
            )
        else:
            rospy.logwarn(f"  fine descent planning insufficient: fraction={fine_fraction*100:.1f}%")

    if descent_z_error > 0.070:
        rospy.logerr(
            f"  descent still too high ({descent_z_error*100:.1f}cm); "
            "abort grasp instead of edge-closing"
        )
        if trajectory_recorder is not None:
            try:
                trajectory_recorder.stop()
                saved_path = trajectory_recorder.save(
                    trajectory_record_path, success=False)
                rospy.loginfo(
                    f"End-effector trajectory saved: {saved_path} "
                    f"samples={len(trajectory_recorder.samples)} success=False"
                )
                trajectory_recorder = None
            except Exception as e:
                rospy.logwarn(f"Failed to save failed end-effector trajectory: {e}")
        move_group.set_goal_position_tolerance(0.005)
        move_group.set_goal_orientation_tolerance(0.02)
        move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
        move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
        safe_pose = copy.deepcopy(final_pose_after_descent)
        safe_pose.position.z += 0.10
        move_group.set_pose_target(safe_pose)
        move_group.go(wait=True)
        return gripper, robot_enabled

    rospy.loginfo("Step5 抓取")
    is_gripped = False
    obj_width = object_size[1]  # 物体宽度（夹爪方向）
    expected_closed = max(0.005, obj_width - 0.005)  # 夹住物体时的期望夹爪位置

    try:
        initial_gripper_pos = gripper.get_position()
        rospy.loginfo(f"  夹爪初始位置: {initial_gripper_pos:.3f}m "
                      f"期望夹住时位置: ~{expected_closed:.3f}m")

        correct_top_mouth_xy_before_close(
            tf_listener, move_group, "Scripted before gripper close",
            [obj_base_x, obj_base_y])
        gripper.close()
        rospy.sleep(2.0)
        current_gripper_pos = gripper.get_position()
        closure = initial_gripper_pos - current_gripper_pos

        # 判定: 夹爪闭合量>5mm 且 最终位置合理（没全闭到0）
        if closure > 0.005 and current_gripper_pos > 0.003:
            is_gripped = True
            rospy.loginfo(f"  抓取成功！{initial_gripper_pos:.3f}→{current_gripper_pos:.3f} "
                          f"(闭合{closure*100:.0f}mm)")
        elif current_gripper_pos < 0.005:
            rospy.logwarn(f"  夹爪全闭({current_gripper_pos:.3f}m)→夹空了")
            # 打开重试一次
            gripper.open()
            rospy.sleep(1.0)
            gripper.close()
            rospy.sleep(2.0)
            current_gripper_pos = gripper.get_position()
            closure = initial_gripper_pos - current_gripper_pos
            if closure > 0.005 and current_gripper_pos > 0.003:
                is_gripped = True
                rospy.loginfo(f"  重试抓取成功！")
            else:
                rospy.logwarn(f"  重试仍失败，继续执行")
        else:
            rospy.logwarn(f"  夹爪闭合过小({closure*100:.0f}mm)→未抓到物体")

    except Exception as e:
        rospy.logwarn(f"  夹爪SDK异常: {e}")

    # ====================== Step6: 垂直抬起+停留展示+放下 ======================
    rospy.loginfo("Step6 垂直抬起")
    lift_success = False
    final_pose = final_pose_after_descent  # 从 Step4 传下来
    lift_pose_final = copy.deepcopy(final_pose)
    # 抬起到抓取位姿上方15cm（安全高度）
    lift_pose_final.position.z = grasp_z + 0.15
    move_group.set_pose_target(lift_pose_final)
    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    
    if plan_success:
        display_traj.trajectory[0] = plan
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        rospy.sleep(0.5)
        lift_success = True
        if not is_gripped:
            rospy.logwarn(
                "  gripper encoder did not confirm closure, but lift motion "
                "completed; marking grasp as failed instead of success"
            )
        rospy.loginfo(f"抬起成功！当前高度Z={lift_pose_final.position.z:.3f}m")
        
        # 【新增】抓起来后停留展示3秒
        rospy.loginfo("------------------------------------------------")
        rospy.loginfo("Grasp completed; holding object after lift.")
        rospy.loginfo("------------------------------------------------")
        rospy.sleep(0.8)
    else:
        rospy.logerr("抬起规划失败")

    rospy.loginfo("================ 全部任务完成 ================")
    if trajectory_recorder is not None:
        try:
            trajectory_recorder.stop()
            saved_path = trajectory_recorder.save(
                trajectory_record_path, success=is_gripped)
            rospy.loginfo(
                f"End-effector trajectory saved: {saved_path} "
                f"samples={len(trajectory_recorder.samples)} "
                f"success={is_gripped}"
            )
        except Exception as e:
            rospy.logwarn(f"Failed to save end-effector trajectory: {e}")
    if is_gripped and hold_on_success:
        rospy.set_param('/sawyer_auto_grasp/keep_gripper_closed_on_exit', True)
    return gripper, robot_enabled
# ====================== 主程序入口 ======================【完全未修改】
if __name__ == '__main__':
    gripper = None
    exit_code = 0
    try:
        gripper, robot_enabled = auto_grasp_with_moveit()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户手动中断")
    except Exception as e:
        rospy.logerr(f"程序运行异常: {e}")
        exit_code = 1
    finally:
        rospy.loginfo("资源清理")
        if gripper:
            try:
                if rospy.get_param('/sawyer_auto_grasp/keep_gripper_closed_on_exit', False):
                    rospy.loginfo("Keeping gripper closed after successful grasp.")
                else:
                    gripper.open()
                    rospy.loginfo("夹爪已打开")
            except:
                pass
        moveit_commander.roscpp_shutdown()
        rospy.sleep(0.5) # 【优化】从1→0.5
        rospy.loginfo("清理完成")
        if exit_code:
            sys.exit(exit_code)
