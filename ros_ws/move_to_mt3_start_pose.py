#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move Sawyer to the MT3 simulation startup pose.

The project historically starts trials from the Gazebo near-zero joint pose.
This script explicitly sends that pose as a controller setpoint after the
robot is enabled, instead of relying on the controller to inherit spawn state.
If the arm has already folded down, it can optionally recover through a higher
safe pose and then return to the MT3 zero startup pose.
"""

import argparse
import sys

import rospy
import intera_interface


MT3_START_JOINTS = {
    "right_j0": 0.0,
    "right_j1": 0.0,
    "right_j2": 0.0,
    "right_j3": 0.0,
    "right_j4": 0.0,
    "right_j5": 0.0,
    "right_j6": 0.0,
}

MT3_SAFE_JOINTS = {
    "right_j0": 0.0,
    "right_j1": -0.8,
    "right_j2": 0.0,
    "right_j3": 1.8,
    "right_j4": 0.0,
    "right_j5": 0.0,
    "right_j6": 0.0,
}


def max_joint_error(current, target):
    errors = []
    for name, value in target.items():
        if name in current:
            errors.append(abs(float(current[name]) - float(value)))
    return max(errors) if errors else float("inf")


def move_and_check(limb, target, speed, timeout, tolerance, label):
    rospy.loginfo("Moving to %s", label)
    limb.set_joint_position_speed(float(speed))
    limb.move_to_joint_positions(target, timeout=float(timeout))
    current = limb.joint_angles()
    error = max_joint_error(current, target)
    rospy.loginfo("%s max joint error: %.4frad", label, error)
    return error <= float(tolerance)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--tolerance", type=float, default=0.035)
    parser.add_argument(
        "--recover-safe-first",
        action="store_true",
        help="Move through the higher MT3 safe pose before returning to zero.")
    parser.add_argument(
        "--fallback-safe",
        action="store_true",
        help="If direct zero startup move fails, recover safe then retry zero.")
    args = parser.parse_args(argv)

    rospy.init_node("move_to_mt3_start_pose", anonymous=True)
    rospy.wait_for_message("/robot/state", rospy.AnyMsg, timeout=30.0)
    limb = intera_interface.Limb("right")

    rospy.loginfo("Current joints: %s", limb.joint_angles())
    if args.recover_safe_first:
        if not move_and_check(
                limb, MT3_SAFE_JOINTS, args.speed, args.timeout,
                args.tolerance, "MT3 safe pose"):
            rospy.logerr("Failed to reach MT3 safe pose")
            return 1

    if move_and_check(
            limb, MT3_START_JOINTS, args.speed, args.timeout,
            args.tolerance, "MT3 zero startup pose"):
        rospy.loginfo("MT3 startup pose ready: %s", limb.joint_angles())
        return 0

    if not args.fallback_safe:
        rospy.logerr("Failed to reach MT3 zero startup pose")
        return 1

    rospy.logwarn("Direct startup pose failed; recovering through safe pose")
    if not move_and_check(
            limb, MT3_SAFE_JOINTS, args.speed, args.timeout,
            args.tolerance, "MT3 safe recovery pose"):
        rospy.logerr("Failed to reach MT3 safe recovery pose")
        return 1

    if not move_and_check(
            limb, MT3_START_JOINTS, args.speed, args.timeout,
            args.tolerance, "MT3 zero startup pose after recovery"):
        rospy.logerr("Failed to reach MT3 zero startup pose after recovery")
        return 1

    rospy.loginfo("MT3 startup pose ready: %s", limb.joint_angles())
    return 0


if __name__ == "__main__":
    sys.exit(main())
