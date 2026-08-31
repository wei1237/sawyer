#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute a simple MT3-style pick-and-place rollout in Sawyer Gazebo.

The MT3 pipeline writes grasp and place targets into /sawyer_auto_grasp/*.
This script intentionally keeps the first version conservative:
top grasp -> lift -> move above place target -> descend until the object is
on the table -> open gripper -> retreat upward.
"""

import json
import os
import sys
import threading

import geometry_msgs.msg
import moveit_commander
import rospy
from intera_interface import Gripper, RobotEnable


ROS_NAMESPACE = "/robot"
PLANNING_GROUP = "right_arm"
END_EFFECTOR_LINK = "right_hand"

ORI_VEL_SCALE = 0.25
ORI_ACC_SCALE = 0.25
DOWN_VEL_SCALE = 0.08
DOWN_ACC_SCALE = 0.08
CART_STEP = 0.006
TOP_FLANGE_Z_OFFSET = 0.050


class EndEffectorTrajectoryRecorder(object):
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

    def save(self, path, success):
        if not path:
            return None
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
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
        data = {
            "format": "sampled_pick_place_rollout_v1",
            "frame": "base",
            "sample_rate_hz": self.rate_hz,
            "num_waypoints": len(self.samples),
            "poses": self.samples,
            "velocities": velocities,
            "success": bool(success),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


def _make_pose(x, y, z, q):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def _go_pose(move_group, pose, label, velocity=ORI_VEL_SCALE,
             acceleration=ORI_ACC_SCALE, attempts=3, planning_time=8.0):
    rospy.loginfo(
        "%s target: [%.3f, %.3f, %.3f]",
        label, pose.position.x, pose.position.y, pose.position.z)
    move_group.set_max_velocity_scaling_factor(float(velocity))
    move_group.set_max_acceleration_scaling_factor(float(acceleration))
    move_group.set_planning_time(float(planning_time))
    for attempt in range(attempts):
        move_group.set_pose_target(pose)
        plan_result = move_group.plan()
        ok = bool(plan_result[0])
        if ok:
            move_group.execute(plan_result[1], wait=True)
            rospy.sleep(0.4)
            move_group.stop()
            move_group.clear_pose_targets()
            return True
        rospy.logwarn("%s planning retry %d/%d", label, attempt + 1, attempts)
    move_group.clear_pose_targets()
    rospy.logerr("%s failed", label)
    return False


def _cartesian_to(move_group, pose, label, min_fraction=0.90, eef_step=CART_STEP):
    start = move_group.get_current_pose().pose
    plan, fraction = move_group.compute_cartesian_path(
        [start, pose],
        float(eef_step),
        True,
    )
    rospy.loginfo("%s cartesian fraction: %.1f%%", label, fraction * 100.0)
    if fraction < min_fraction or not plan.joint_trajectory.points:
        rospy.logwarn("%s cartesian insufficient; using pose target fallback", label)
        return _go_pose(
            move_group, pose, label + " fallback",
            velocity=DOWN_VEL_SCALE, acceleration=DOWN_ACC_SCALE,
            attempts=2, planning_time=6.0)
    move_group.execute(plan, wait=True)
    rospy.sleep(0.4)
    return True


def _init_robot():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("mt3_sawyer_place", anonymous=True)

    robot_enable = RobotEnable()
    try:
        robot_enable.enable()
    except Exception as exc:
        rospy.logwarn("Robot enable skipped/failed: %s", exc)

    move_group = moveit_commander.MoveGroupCommander(
        PLANNING_GROUP,
        robot_description="%s/robot_description" % ROS_NAMESPACE,
        ns=ROS_NAMESPACE,
    )
    move_group.set_end_effector_link(END_EFFECTOR_LINK)
    move_group.set_pose_reference_frame("base")
    move_group.set_goal_position_tolerance(0.008)
    move_group.set_goal_orientation_tolerance(0.05)
    move_group.set_num_planning_attempts(3)

    gripper = Gripper("right_gripper")
    if not gripper.is_calibrated():
        gripper.calibrate()
        rospy.sleep(1.0)
    gripper.set_cmd_velocity(0.1)
    return move_group, gripper


def execute_pick_place():
    move_group, gripper = _init_robot()
    trajectory_record_path = rospy.get_param(
        "/sawyer_auto_grasp/trajectory_record_path", "")
    trajectory_rate = float(rospy.get_param(
        "/sawyer_auto_grasp/trajectory_record_rate_hz", 10.0))
    recorder = EndEffectorTrajectoryRecorder(
        move_group, gripper=gripper, rate_hz=trajectory_rate)
    recorder.start()

    success = False
    try:
        grasp_x = float(rospy.get_param("/sawyer_auto_grasp/grasp_x"))
        grasp_y = float(rospy.get_param("/sawyer_auto_grasp/grasp_y"))
        grasp_z = float(rospy.get_param("/sawyer_auto_grasp/grasp_z"))
        q = [
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qx", -1.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qy", 0.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qz", 0.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qw", 0.0)),
        ]
        object_size = rospy.get_param(
            "/sawyer_auto_grasp/object_size", [0.045, 0.045, 0.045])
        object_height = float(object_size[2]) if len(object_size) >= 3 else 0.045

        place_x = float(rospy.get_param("/sawyer_auto_grasp/place_x"))
        place_y = float(rospy.get_param("/sawyer_auto_grasp/place_y"))
        place_z = float(rospy.get_param(
            "/sawyer_auto_grasp/place_z",
            grasp_z + object_height + 0.03))
        place_direction = rospy.get_param(
            "/sawyer_auto_grasp/place_direction", "right")
        place_clearance = float(rospy.get_param(
            "/sawyer_auto_grasp/place_clearance", 0.030))
        lift_height = float(rospy.get_param(
            "/sawyer_auto_grasp/place_lift_height", 0.150))

        grasp_contact_z = grasp_z
        grasp_flange_z = grasp_contact_z + TOP_FLANGE_Z_OFFSET
        pregrasp_z = grasp_flange_z + 0.10
        lift_z = grasp_flange_z + lift_height
        place_above_z = max(place_z + lift_height, lift_z)
        place_release_z = place_z + TOP_FLANGE_Z_OFFSET + place_clearance

        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 pick-place execution")
        rospy.loginfo("  grasp: [%.3f, %.3f, %.3f]", grasp_x, grasp_y, grasp_z)
        rospy.loginfo(
            "  place: [%.3f, %.3f, %.3f] direction=%s",
            place_x, place_y, place_z, place_direction)
        rospy.loginfo(
            "  release_z=%.3f object_height=%.3f clearance=%.3f",
            place_release_z, object_height, place_clearance)
        rospy.loginfo("=" * 60)

        gripper.open()
        rospy.sleep(0.8)

        pregrasp = _make_pose(grasp_x, grasp_y, pregrasp_z, q)
        grasp_pose = _make_pose(grasp_x, grasp_y, grasp_flange_z, q)
        lift_pose = _make_pose(grasp_x, grasp_y, lift_z, q)
        place_above = _make_pose(place_x, place_y, place_above_z, q)
        place_release = _make_pose(place_x, place_y, place_release_z, q)
        retreat = _make_pose(place_x, place_y, place_above_z, q)

        if not _go_pose(move_group, pregrasp, "Step A: pregrasp"):
            return False
        if not _cartesian_to(move_group, grasp_pose, "Step B: descend to grasp"):
            return False

        rospy.loginfo("Step C: close gripper")
        initial_gripper = None
        try:
            initial_gripper = float(gripper.get_position())
        except Exception:
            pass
        gripper.close()
        rospy.sleep(1.5)
        try:
            current_gripper = float(gripper.get_position())
            rospy.loginfo(
                "  gripper position: %.3f -> %.3f",
                initial_gripper if initial_gripper is not None else -1.0,
                current_gripper)
        except Exception:
            pass

        if not _cartesian_to(move_group, lift_pose, "Step D: lift object"):
            return False
        if not _go_pose(move_group, place_above, "Step E: move above place"):
            return False

        rospy.loginfo("Step F: descend to table release height before opening")
        if not _cartesian_to(
                move_group, place_release, "Step F: descend to place",
                min_fraction=0.80, eef_step=0.004):
            return False

        rospy.sleep(0.5)
        rospy.loginfo("Step G: open gripper only after reaching place height")
        gripper.open()
        rospy.sleep(1.0)

        if not _cartesian_to(move_group, retreat, "Step H: retreat upward"):
            return False

        success = True
        rospy.loginfo("MT3 pick-place completed successfully")
        return True
    finally:
        if recorder is not None:
            recorder.stop()
            saved = recorder.save(trajectory_record_path, success=success)
            if saved:
                rospy.loginfo(
                    "Pick-place rollout saved: %s success=%s samples=%d",
                    saved, success, len(recorder.samples))
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    ok = False
    try:
        ok = execute_pick_place()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted")
    except Exception as exc:
        rospy.logerr("Pick-place execution failed: %s", exc)
    if not ok:
        sys.exit(1)
