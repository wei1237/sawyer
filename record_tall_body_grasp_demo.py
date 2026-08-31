#!/usr/bin/env python3
"""
Record an MT3-style vertical body grasp demo for tall objects.

The default mode keeps the gripper horizontal, aligns x/y at a high z, then
descends straight down to the upper body of a tall object before closing and
lifting. This is a stable alternative when true horizontal side approach is
unreliable in Sawyer Gazebo.
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

from mt3_scene_package import save_scene_package
from record_demo import DemoRecorder, OUTPUT_DIR


class TallBodyGraspDemoRecorder(DemoRecorder):
    def __init__(self, object_x, object_y, object_z, object_size, demo_name,
                 body_grasp_height_fraction=0.70,
                 orientation_mode="horizontal",
                 gripper_roll_deg=-180.0,
                 gripper_pitch_deg=-90.0,
                 gripper_yaw_deg=90.0,
                 mask_path="", body_grasp_margin=0.000,
                 bottleneck_clearance=0.18, lift_height=0.12):
        super().__init__(object_x, object_y, object_z, object_size, demo_name)
        self.body_grasp_height_fraction = max(
            0.35, min(0.85, float(body_grasp_height_fraction)))
        self.orientation_mode = str(orientation_mode or "horizontal").lower()
        self.gripper_roll_deg = float(gripper_roll_deg)
        self.gripper_pitch_deg = float(gripper_pitch_deg)
        self.gripper_yaw = math.radians(float(gripper_yaw_deg))
        self.gripper_yaw_deg = float(gripper_yaw_deg)
        self.mask_path = str(mask_path or "")
        self.body_grasp_margin = float(body_grasp_margin)
        self.bottleneck_clearance = float(bottleneck_clearance)
        self.lift_height = float(lift_height)
        self.gripper = None

    def _body_grasp_orientation(self):
        if self.orientation_mode == "top_down":
            roll = math.pi
            pitch = 0.0
            yaw = self.gripper_yaw
        else:
            roll = math.radians(self.gripper_roll_deg)
            pitch = math.radians(self.gripper_pitch_deg)
            yaw = self.gripper_yaw
        q = quaternion_from_euler(roll, pitch, yaw)
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

    def _execute_incremental_z_to(self, target_pose, label, step=0.010,
                                  max_z_error=0.025, max_steps=35):
        rospy.loginfo(
            "  %s incremental z move: target_z=%.4f step=%.1fcm",
            label, target_pose.position.z, step * 100.0)

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
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            rospy.sleep(0.25)

            actual = self.move_group.get_current_pose().pose
            error = abs(float(actual.position.z) - float(target_pose.position.z))
            rospy.loginfo(
                "  %s step %02d ok=%s actual_z=%.4f target_z=%.4f error=%.1fcm",
                label, i + 1, ok, actual.position.z,
                target_pose.position.z, error * 100.0)

            if error <= max_z_error:
                return True

        actual = self.move_group.get_current_pose().pose
        z_error = abs(float(actual.position.z) - float(target_pose.position.z))
        rospy.logerr(
            "  %s stopped too far from target after %d steps: %.1fcm > %.1fcm",
            label, max_steps, z_error * 100.0, max_z_error * 100.0)
        return False

    def _lookup_base_xyz(self, frame):
        try:
            trans = self.tf_buffer.lookup_transform(
                "base", frame, rospy.Time(0), rospy.Duration(1.0))
            p = trans.transform.translation
            return np.array([p.x, p.y, p.z], dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("  TF lookup failed for %s: %s", frame, exc)
            return None

    def _get_gripper_mouth_state(self):
        left = self._lookup_base_xyz("right_gripper_l_finger_tip")
        right = self._lookup_base_xyz("right_gripper_r_finger_tip")
        hand = self._lookup_base_xyz("right_hand")
        if left is None or right is None or hand is None:
            rospy.logwarn(
                "  Gripper mouth TF unavailable; using right_hand center")
            return None
        center = 0.5 * (left + right)
        opening = float(np.linalg.norm(left - right))
        return {
            "left": left,
            "right": right,
            "hand": hand,
            "center": center,
            "offset": center - hand,
            "opening": opening,
        }

    def _pose_for_mouth_center(self, reference_pose, desired_mouth_center,
                               mouth_offset):
        pose = copy.deepcopy(reference_pose)
        command_xyz = (
            np.array(desired_mouth_center, dtype=np.float64) -
            np.array(mouth_offset, dtype=np.float64))
        pose.position.x = float(command_xyz[0])
        pose.position.y = float(command_xyz[1])
        pose.position.z = float(command_xyz[2])
        return pose

    def _log_mouth_alignment(self, label, desired_mouth_center):
        state = self._get_gripper_mouth_state()
        if state is None:
            return
        desired = np.array(desired_mouth_center, dtype=np.float64)
        err = state["center"] - desired
        rospy.loginfo(
            "  %s mouth center: actual=[%.3f, %.3f, %.3f] "
            "desired=[%.3f, %.3f, %.3f] error=[%.1f, %.1f, %.1f]cm "
            "opening=%.1fcm",
            label,
            state["center"][0], state["center"][1], state["center"][2],
            desired[0], desired[1], desired[2],
            err[0] * 100.0, err[1] * 100.0, err[2] * 100.0,
            state["opening"] * 100.0)

    def execute_and_record(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 Tall Body Grasp Demo Recording: %s", self.demo_name)
        rospy.loginfo("Object at: %s size=%s", self.object_pos, self.object_size)
        rospy.loginfo(
            "Body grasp height fraction: %.2f",
            self.body_grasp_height_fraction)
        rospy.loginfo(
            "Gripper orientation mode=%s rpy=[%.1f, %.1f, %.1f] deg",
            self.orientation_mode,
            self.gripper_roll_deg,
            self.gripper_pitch_deg,
            self.gripper_yaw_deg)
        rospy.loginfo("=" * 60)

        rospy.loginfo("[1/5] Moving to safe starting pose...")
        safe_joints = {
            "right_j0": 0.0, "right_j1": -0.8, "right_j2": 0.0,
            "right_j3": 1.8, "right_j4": 0.0, "right_j5": 0.0, "right_j6": 0.0
        }
        self.move_group.set_joint_value_target(safe_joints)
        self.move_group.go(wait=True)
        self.move_group.stop()
        rospy.sleep(1.0)

        bx, by, bz = [float(v) for v in self.object_pos]
        sx, sy, sz = [float(v) for v in self.object_size]
        q = self._body_grasp_orientation()
        object_top_z = bz + sz
        body_grasp_z = bz + sz * self.body_grasp_height_fraction + self.body_grasp_margin

        bottleneck_pose = geometry_msgs.msg.Pose()
        bottleneck_pose.position.x = bx
        bottleneck_pose.position.y = by
        bottleneck_pose.position.z = object_top_z + self.bottleneck_clearance
        bottleneck_pose.orientation.x = q[0]
        bottleneck_pose.orientation.y = q[1]
        bottleneck_pose.orientation.z = q[2]
        bottleneck_pose.orientation.w = q[3]

        rospy.loginfo("[2/5] Moving to high-z x/y-aligned bottleneck above tall object...")
        rospy.loginfo(
            "  bottleneck=[%.3f, %.3f, %.3f], body_grasp_z=%.3f",
            bx, by, bottleneck_pose.position.z, body_grasp_z)
        self.move_group.set_max_velocity_scaling_factor(0.15)
        self.move_group.set_max_acceleration_scaling_factor(0.15)
        self.move_group.set_pose_target(bottleneck_pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        rospy.sleep(1.0)
        if not success:
            rospy.logerr("Failed to reach tall-body bottleneck pose.")
            return False

        rospy.loginfo("  Opening gripper before descent...")
        self._open_gripper()

        desired_bottleneck_mouth = np.array(
            [bx, by, object_top_z + self.bottleneck_clearance],
            dtype=np.float64)
        mouth_state = self._get_gripper_mouth_state()
        mouth_offset = None
        if mouth_state is not None:
            mouth_offset = mouth_state["offset"]
            rospy.loginfo(
                "  Gripper mouth offset from right_hand: [%.3f, %.3f, %.3f] "
                "opening=%.1fcm",
                mouth_offset[0], mouth_offset[1], mouth_offset[2],
                mouth_state["opening"] * 100.0)
            corrected_bottleneck = self._pose_for_mouth_center(
                bottleneck_pose, desired_bottleneck_mouth, mouth_offset)
            rospy.loginfo(
                "  Correcting high-z bottleneck so finger-mouth center, not "
                "right_hand, aligns with object x/y")
            self.move_group.set_pose_target(corrected_bottleneck)
            corrected_ok = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            rospy.sleep(0.5)
            if not corrected_ok:
                rospy.logerr("Failed to correct bottleneck by mouth center.")
                return False
            bottleneck_pose = corrected_bottleneck
            self._log_mouth_alignment(
                "high-z xy aligned", desired_bottleneck_mouth)

        rospy.loginfo("[3/5] Capturing bottleneck RGB-D observation...")
        self._capture_bottleneck(timeout=3.0)
        bottleneck_ee = self._get_end_effector_pose()
        if bottleneck_ee is None:
            rospy.logerr("Failed to get bottleneck end-effector pose.")
            return False

        rospy.loginfo("[4/5] Executing horizontal-gripper vertical body grasp and recording...")
        self.recording = True
        self.recorded_poses = []
        first_pose = self._stamp_gripper_state(self._get_end_effector_pose())
        if first_pose:
            self.recorded_poses.append(first_pose)

        record_thread = threading.Thread(
            target=self._continuous_record, args=(8.0,))
        record_thread.start()

        grasp_pose = copy.deepcopy(bottleneck_pose)
        if mouth_offset is not None:
            desired_grasp_mouth = np.array([bx, by, body_grasp_z],
                                           dtype=np.float64)
            grasp_pose = self._pose_for_mouth_center(
                bottleneck_pose, desired_grasp_mouth, mouth_offset)
            rospy.loginfo(
                "  Descending with mouth-center target: [%.3f, %.3f, %.3f]",
                desired_grasp_mouth[0], desired_grasp_mouth[1],
                desired_grasp_mouth[2])
        else:
            desired_grasp_mouth = None
            grasp_pose.position.z = body_grasp_z
        if not self._execute_incremental_z_to(
                grasp_pose, "horizontal body descent",
                step=0.010, max_z_error=0.045, max_steps=40):
            self.recording = False
            record_thread.join(timeout=2.0)
            rospy.logerr("Tall body grasp demo aborted: descent failed.")
            return False
        if desired_grasp_mouth is not None:
            self._log_mouth_alignment("before gripper close", desired_grasp_mouth)

        rospy.loginfo("  Closing gripper on upper body...")
        self._close_gripper()

        lift_pose = copy.deepcopy(grasp_pose)
        if mouth_offset is not None:
            desired_lift_mouth = np.array(
                [bx, by, min(desired_bottleneck_mouth[2],
                             body_grasp_z + self.lift_height)],
                dtype=np.float64)
            lift_pose = self._pose_for_mouth_center(
                grasp_pose, desired_lift_mouth, mouth_offset)
        else:
            lift_pose.position.z = min(
                bottleneck_pose.position.z,
                grasp_pose.position.z + self.lift_height)
        if not self._execute_incremental_z_to(
                lift_pose, "horizontal body lift",
                step=0.012, max_z_error=0.025, max_steps=30):
            self.recording = False
            record_thread.join(timeout=2.0)
            rospy.logerr("Tall body grasp demo aborted: lift failed.")
            return False

        self.recording = False
        record_thread.join(timeout=2.0)
        rospy.loginfo("  Recording stopped. Total poses: %d", len(self.recorded_poses))

        self.move_group.set_max_velocity_scaling_factor(0.6)
        self.move_group.set_max_acceleration_scaling_factor(0.6)

        rospy.loginfo("[5/5] Saving tall body grasp demo to recorded library...")
        self._save_demo(bottleneck_ee)
        self._patch_recorded_json()

        rospy.loginfo("=" * 60)
        rospy.loginfo("Tall body grasp demo '%s' recorded successfully.", self.demo_name)
        rospy.loginfo("=" * 60)
        return True

    def _patch_recorded_json(self):
        json_path = os.path.join(OUTPUT_DIR, "%s.json" % self.demo_name)
        with open(json_path, "r", encoding="utf-8") as f:
            demo = json.load(f)

        shape = str(rospy.get_param("~object_shape", "cylinder"))
        label = str(rospy.get_param("~object_label", "green_tall_cylinder"))
        demo["language_description"] = (
            "Pick up the green tall cylinder with a horizontal gripper vertical body grasp")
        demo["language_tags"] = [
            "grasp",
            "pick up",
            "horizontal gripper",
            "body grasp",
            "horizontal body grasp",
            "vertical descent",
            "vertical body grasp",
            "tall object",
            "tall cylinder",
            "green cylinder",
            "green tall cylinder",
            "upper body grasp",
            "high object grasp",
            "side-like grasp",
            "yawed gripper",
            "body side wall grasp",
        ]
        demo["object_info"]["category"] = shape
        demo["object_info"]["label"] = label
        demo["object_info"]["color"] = "green"
        demo["grasp_strategy"] = "horizontal_gripper_vertical_body_grasp"
        demo["body_grasp_height_fraction"] = float(
            self.body_grasp_height_fraction)
        demo["orientation_mode"] = self.orientation_mode
        demo["gripper_roll_deg"] = float(self.gripper_roll_deg)
        demo["gripper_pitch_deg"] = float(self.gripper_pitch_deg)
        demo["gripper_yaw_deg"] = float(self.gripper_yaw_deg)
        demo["notes"] = (
            "Recorded horizontal-gripper vertical body grasp demo for tall "
            "objects. The gripper is already horizontal at a high z pose, x/y "
            "are aligned before descent, then the robot descends straight down "
            "to close around the upper body and lift.")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(demo, f, indent=2, ensure_ascii=False)
        rospy.loginfo("  Patched tall body metadata in %s", json_path)

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
                "  Using LangSAM mask for tall body demo scene package: %s pixels=%d",
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
                "  Falling back to HSV mask for tall body demo scene package: pixels=%d",
                int(np.count_nonzero(segmap)) if segmap is not None else 0)

        shape = str(rospy.get_param("~object_shape", "cylinder"))
        label = str(rospy.get_param("~object_label", "green_tall_cylinder"))
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
                "method": "recorded_tall_body_bottleneck_pose",
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
                "grasp_strategy": "horizontal_gripper_vertical_body_grasp",
                "body_grasp_height_fraction": self.body_grasp_height_fraction,
                "orientation_mode": self.orientation_mode,
                "gripper_roll_deg": self.gripper_roll_deg,
                "gripper_pitch_deg": self.gripper_pitch_deg,
                "gripper_yaw_deg": self.gripper_yaw_deg,
                "mask_source": self.mask_path or "hsv_fallback",
            })
        rospy.loginfo(
            "  Scene package saved to %s (points=%d, mask_px=%d)",
            package["package_dir"],
            package["stats"]["pointcloud_points"],
            package["stats"]["segmap_pixels"])


if __name__ == "__main__":
    rospy.init_node("mt3_record_tall_body_grasp_demo", anonymous=True)

    obj_x = rospy.get_param("~object_x", 0.60)
    obj_y = rospy.get_param("~object_y", 0.00)
    obj_z = rospy.get_param("~object_z", -0.58)
    obj_size = rospy.get_param("~object_size", [0.045, 0.045, 0.160])
    demo_name = rospy.get_param(
        "~demo_name", "tall_cylinder_horizontal_body_grasp")
    body_grasp_height_fraction = rospy.get_param(
        "~body_grasp_height_fraction", 0.70)
    orientation_mode = rospy.get_param("~orientation_mode", "horizontal")
    gripper_roll_deg = rospy.get_param("~gripper_roll_deg", -180.0)
    gripper_pitch_deg = rospy.get_param("~gripper_pitch_deg", -90.0)
    gripper_yaw_deg = rospy.get_param("~gripper_yaw_deg", 90.0)
    mask_path = rospy.get_param(
        "~mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")
    body_grasp_margin = rospy.get_param("~body_grasp_margin", 0.0)
    bottleneck_clearance = rospy.get_param("~bottleneck_clearance", 0.18)
    lift_height = rospy.get_param("~lift_height", 0.12)

    recorder = TallBodyGraspDemoRecorder(
        obj_x, obj_y, obj_z, obj_size, demo_name,
        body_grasp_height_fraction=body_grasp_height_fraction,
        orientation_mode=orientation_mode,
        gripper_roll_deg=gripper_roll_deg,
        gripper_pitch_deg=gripper_pitch_deg,
        gripper_yaw_deg=gripper_yaw_deg,
        mask_path=mask_path,
        body_grasp_margin=body_grasp_margin,
        bottleneck_clearance=bottleneck_clearance,
        lift_height=lift_height)

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
