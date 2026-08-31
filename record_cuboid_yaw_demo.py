#!/usr/bin/env python3
"""
Record a cuboid top-grasp demo with an explicit gripper yaw angle.

The bottleneck pose is allowed to use normal MoveIt planning. After that, the
recorded interaction phase uses Cartesian descent/lift so the demo does not
contain a wrist spin or a joint-space detour.
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
from tf.transformations import quaternion_from_euler
try:
    from intera_interface import Gripper
except Exception:
    Gripper = None

from record_demo import DemoRecorder, OUTPUT_DIR
from mt3_scene_package import save_scene_package


class CuboidYawDemoRecorder(DemoRecorder):
    def __init__(self, object_x, object_y, object_z, object_size, demo_name,
                 gripper_yaw_deg, mask_path="", demo_surface_margin=0.005,
                 flange_grasp_z_offset=0.040):
        super().__init__(object_x, object_y, object_z, object_size, demo_name)
        self.gripper_yaw = math.radians(float(gripper_yaw_deg))
        self.gripper_yaw_deg = float(gripper_yaw_deg)
        self.mask_path = str(mask_path or "")
        self.demo_surface_margin = float(demo_surface_margin)
        self.flange_grasp_z_offset = float(flange_grasp_z_offset)
        self.gripper = None

    def _top_down_orientation(self):
        """Top-down gripper orientation plus yaw around the base z axis."""
        q = quaternion_from_euler(math.pi, 0.0, self.gripper_yaw)
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

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
        try:
            rospy.loginfo("  Gripper opened: position=%.3fm",
                          float(self.gripper.get_position()))
        except Exception:
            rospy.loginfo("  Gripper opened")
        return True

    def _close_gripper(self):
        if not self._init_gripper():
            return False
        self.gripper_command_state = 1
        self.gripper.close()
        rospy.sleep(1.5)
        try:
            rospy.loginfo("  Gripper closed: position=%.3fm",
                          float(self.gripper.get_position()))
        except Exception:
            rospy.loginfo("  Gripper closed")
        return True

    def _execute_cartesian_to(self, target_pose, label, eef_step=0.003,
                              min_fraction=0.70, max_z_error=0.05):
        """Move in a near-straight Cartesian segment while keeping orientation."""
        start_pose = self.move_group.get_current_pose().pose
        waypoints = [copy.deepcopy(start_pose), copy.deepcopy(target_pose)]
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step,
            True,
        )
        rospy.loginfo("  %s cartesian fraction: %.1f%%", label, fraction * 100.0)
        if fraction < min_fraction or not plan.joint_trajectory.points:
            rospy.logerr(
                "  %s cartesian path failed: fraction %.1f%% < %.1f%%",
                label, fraction * 100.0, min_fraction * 100.0)
            return False
        ok = self.move_group.execute(plan, wait=True)
        rospy.sleep(0.5)
        actual_pose = self.move_group.get_current_pose().pose
        z_error = abs(float(actual_pose.position.z) - float(target_pose.position.z))
        rospy.loginfo(
            "  %s result z: target=%.4f actual=%.4f error=%.1fcm",
            label, target_pose.position.z, actual_pose.position.z, z_error * 100.0)
        if z_error > max_z_error:
            rospy.logerr(
                "  %s stopped too far from target: %.1fcm > %.1fcm",
                label, z_error * 100.0, max_z_error * 100.0)
            return False
        return bool(ok)

    def _execute_incremental_z_to(self, target_pose, label, step=0.010,
                                  max_z_error=0.050, max_steps=25,
                                  accept_stall_near_target=True):
        """Move mostly vertically in small pose-target steps.

        The Sawyer simulation controller can abort on one long Cartesian
        segment after a 90 degree wrist yaw. Small z steps avoid that timing
        spike while still recording a near-vertical interaction trajectory.
        """
        rospy.loginfo(
            "  %s incremental z move: target_z=%.4f step=%.1fcm",
            label, target_pose.position.z, step * 100.0)

        best_error = float("inf")
        stalled_steps = 0

        for i in range(max_steps):
            current = self.move_group.get_current_pose().pose
            remaining = float(target_pose.position.z) - float(current.position.z)
            if abs(remaining) <= max_z_error:
                rospy.loginfo(
                    "  %s reached: actual_z=%.4f error=%.1fcm",
                    label, current.position.z, abs(remaining) * 100.0)
                return True

            dz = max(-step, min(step, remaining))
            next_pose = copy.deepcopy(target_pose)
            next_pose.position.z = float(current.position.z) + dz

            self.move_group.set_pose_target(next_pose)
            ok = self.move_group.go(wait=True)
            rospy.sleep(0.25)

            actual = self.move_group.get_current_pose().pose
            error = abs(float(actual.position.z) - float(target_pose.position.z))
            rospy.loginfo(
                "  %s step %02d ok=%s actual_z=%.4f target_z=%.4f error=%.1fcm",
                label, i + 1, ok, actual.position.z,
                target_pose.position.z, error * 100.0)

            if error < best_error - 0.002:
                best_error = error
                stalled_steps = 0
            else:
                stalled_steps += 1

            if error <= max_z_error:
                rospy.loginfo(
                    "  %s accepted: actual_z=%.4f error=%.1fcm <= %.1fcm",
                    label, actual.position.z, error * 100.0,
                    max_z_error * 100.0)
                return True

            if accept_stall_near_target and stalled_steps >= 3 and best_error <= 0.060:
                rospy.logwarn(
                    "  %s z stalled near object: best_error=%.1fcm; accepting to avoid wrist twisting",
                    label, best_error * 100.0)
                return True

            if not ok and error > max_z_error:
                rospy.logwarn(
                    "  %s step %02d planner/controller did not finish; retrying smaller move",
                    label, i + 1)

        actual = self.move_group.get_current_pose().pose
        z_error = abs(float(actual.position.z) - float(target_pose.position.z))
        rospy.logerr(
            "  %s stopped too far from target after %d steps: %.1fcm > %.1fcm",
            label, max_steps, z_error * 100.0, max_z_error * 100.0)
        return False

    def execute_and_record(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 Cuboid Yaw Demo Recording: %s", self.demo_name)
        rospy.loginfo("Object at: %s size=%s", self.object_pos, self.object_size)
        rospy.loginfo("Gripper yaw: %.1f deg", self.gripper_yaw_deg)
        rospy.loginfo(
            "Demo-style grasp z offset: object_height + %.3fm",
            self.demo_surface_margin)
        rospy.loginfo(
            "MoveIt flange z offset: contact_z + %.3fm",
            self.flange_grasp_z_offset)
        rospy.loginfo("=" * 60)

        rospy.loginfo("[1/5] Moving to safe starting pose...")
        safe_joints = {
            "right_j0": 0.0, "right_j1": -0.8, "right_j2": 0.0,
            "right_j3": 1.8, "right_j4": 0.0, "right_j5": 0.0, "right_j6": 0.0
        }
        self.move_group.set_joint_value_target(safe_joints)
        self.move_group.go(wait=True)
        rospy.sleep(1.0)

        rospy.loginfo("[2/5] Moving to yawed bottleneck pose above cuboid...")
        bx, by, bz = self.object_pos
        half_h = self.object_size[2] / 2.0
        q = self._top_down_orientation()

        bottleneck_pose = geometry_msgs.msg.Pose()
        bottleneck_pose.position.x = bx
        bottleneck_pose.position.y = by
        bottleneck_pose.position.z = bz + half_h + 0.15
        bottleneck_pose.orientation.x = q[0]
        bottleneck_pose.orientation.y = q[1]
        bottleneck_pose.orientation.z = q[2]
        bottleneck_pose.orientation.w = q[3]

        self.move_group.set_pose_target(bottleneck_pose)
        success = self.move_group.go(wait=True)
        rospy.sleep(1.0)
        if not success:
            rospy.logerr("Failed to reach yawed bottleneck pose.")
            return False

        rospy.loginfo("  Opening gripper before descent...")
        self._open_gripper()

        rospy.loginfo("[3/5] Capturing bottleneck RGB-D observation...")
        self._capture_bottleneck(timeout=3.0)

        bottleneck_ee = self._get_end_effector_pose()
        if bottleneck_ee is None:
            rospy.logerr("Failed to get bottleneck end-effector pose.")
            return False

        rospy.loginfo("[4/5] Executing incremental yawed top grasp and recording...")
        self.recording = True
        self.recorded_poses = []
        first_pose = self._stamp_gripper_state(self._get_end_effector_pose())
        if first_pose:
            self.recorded_poses.append(first_pose)

        self.move_group.set_max_velocity_scaling_factor(0.12)
        self.move_group.set_max_acceleration_scaling_factor(0.12)

        record_thread = threading.Thread(
            target=self._continuous_record, args=(8.0,))
        record_thread.start()

        grasp_pose = copy.deepcopy(bottleneck_pose)
        contact_z = bz + self.object_size[2] + self.demo_surface_margin
        grasp_pose.position.z = contact_z + self.flange_grasp_z_offset
        rospy.loginfo(
            "  demo-style contact z: %.4f = object_z %.4f + height %.4f + margin %.4f",
            contact_z, bz, self.object_size[2], self.demo_surface_margin)
        rospy.loginfo(
            "  MoveIt flange target z: %.4f = contact_z %.4f + flange_offset %.4f",
            grasp_pose.position.z, contact_z, self.flange_grasp_z_offset)
        descent_ok = self._execute_incremental_z_to(
            grasp_pose, "yawed vertical descent")
        if not descent_ok:
            self.recording = False
            record_thread.join(timeout=2.0)
            rospy.logerr("Cuboid yaw demo aborted: descent did not reach target.")
            return False

        rospy.loginfo("  Closing gripper for recording...")
        self._close_gripper()

        lift_pose = copy.deepcopy(grasp_pose)
        lift_pose.position.z = bz + half_h + 0.15
        lift_ok = self._execute_incremental_z_to(
            lift_pose, "yawed vertical lift",
            max_z_error=0.015,
            max_steps=30,
            accept_stall_near_target=False)
        if not lift_ok:
            self.recording = False
            record_thread.join(timeout=2.0)
            rospy.logerr("Cuboid yaw demo aborted: lift did not reach target.")
            return False

        self.recording = False
        record_thread.join(timeout=2.0)
        rospy.loginfo("  Recording stopped. Total poses: %d", len(self.recorded_poses))

        self.move_group.set_max_velocity_scaling_factor(0.6)
        self.move_group.set_max_acceleration_scaling_factor(0.6)

        rospy.loginfo("[5/5] Saving cuboid yaw demo to recorded library...")
        self._save_demo(bottleneck_ee)
        self._patch_recorded_json()

        rospy.loginfo("=" * 60)
        rospy.loginfo("Cuboid yaw demo '%s' recorded successfully.", self.demo_name)
        rospy.loginfo("=" * 60)
        return True

    def _patch_recorded_json(self):
        json_path = os.path.join(OUTPUT_DIR, "%s.json" % self.demo_name)
        with open(json_path, "r", encoding="utf-8") as f:
            demo = json.load(f)

        demo["language_description"] = (
            "Pick up the green rectangular prism from above with yawed gripper")
        demo["language_tags"] = [
            "grasp",
            "pick up",
            "top-down grasp",
            "cuboid",
            "rectangular prism",
            "green cuboid",
            "green rectangular prism",
            "yaw grasp",
            "rotated gripper",
            "short-side grasp",
            "抓取",
            "长方体",
            "绿色长方体",
            "旋转夹爪",
            "夹短边",
        ]
        demo["object_info"]["category"] = "cuboid"
        demo["object_info"]["label"] = "green_cuboid"
        demo["object_info"]["color"] = "green"
        demo["grasp_strategy"] = "top_down_yaw_short_side"
        demo["gripper_yaw_deg"] = self.gripper_yaw_deg
        demo["notes"] = (
            "Recorded cuboid top grasp demo. The gripper is yawed so the "
            "fingers close across the cuboid short side. Descent/lift are "
            "recorded as Cartesian segments after the yawed bottleneck pose.")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(demo, f, indent=2, ensure_ascii=False)
        rospy.loginfo("  Patched cuboid metadata in %s", json_path)

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
                "  Using LangSAM mask for demo scene package: %s pixels=%d",
                self.mask_path, int(np.count_nonzero(mask)))
            return mask
        except Exception as exc:
            rospy.logwarn("  Failed to load LangSAM mask %s: %s", self.mask_path, exc)
            return None

    def _save_scene_package(self, bottleneck_ee):
        """Save demo scene package, preferring an external LangSAM mask."""
        if self.bottleneck_rgb is None or self.bottleneck_depth is None:
            rospy.logwarn("  Scene package skipped: missing bottleneck RGB-D")
            return

        import cv2
        rgb = cv2.cvtColor(self.bottleneck_rgb, cv2.COLOR_BGR2RGB)
        segmap = self._load_langsam_mask()
        if segmap is None:
            segmap = self._green_mask_from_bgr(self.bottleneck_rgb)
            rospy.logwarn(
                "  Falling back to HSV mask for cuboid demo scene package: pixels=%d",
                int(np.count_nonzero(segmap)) if segmap is not None else 0)

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
                "method": "recorded_bottleneck_pose",
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
                "object_shape": "cuboid",
                "object_label": "green_cuboid",
                "gripper_yaw_deg": self.gripper_yaw_deg,
                "mask_source": self.mask_path or "hsv_fallback",
            })
        rospy.loginfo(
            "  Scene package saved to %s (points=%d, mask_px=%d)",
            package["package_dir"],
            package["stats"]["pointcloud_points"],
            package["stats"]["segmap_pixels"])


if __name__ == "__main__":
    rospy.init_node("mt3_record_cuboid_yaw_demo", anonymous=True)

    obj_x = rospy.get_param("~object_x", 0.60)
    obj_y = rospy.get_param("~object_y", 0.00)
    obj_z = rospy.get_param("~object_z", -0.58)
    obj_size = rospy.get_param("~object_size", [0.04, 0.08, 0.035])
    demo_name = rospy.get_param(
        "~demo_name", "cuboid_green_top_yaw_grasp_v1")
    gripper_yaw_deg = rospy.get_param("~gripper_yaw_deg", 90.0)
    mask_path = rospy.get_param(
        "~mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")
    demo_surface_margin = rospy.get_param("~demo_surface_margin", 0.005)
    flange_grasp_z_offset = rospy.get_param("~flange_grasp_z_offset", 0.040)

    recorder = CuboidYawDemoRecorder(
        obj_x, obj_y, obj_z, obj_size, demo_name, gripper_yaw_deg,
        mask_path, demo_surface_margin, flange_grasp_z_offset)

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
