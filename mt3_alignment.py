#!/usr/bin/env python3
"""
MT3 Alignment Phase

After retrieving the best-matching demonstration from the library and detecting
the object in the scene, this module aligns the demo's grasp trajectory to the
current object pose.

The MT3 paper approach:
  1. Retrieve demo = (object_pose_demo, grasp_pose_demo)
  2. Compute relative transform: T_grasp_in_object = inv(T_object_demo) * T_grasp_demo
  3. Detect current object: T_object_current (from perception)
  4. Aligned grasp: T_grasp_aligned = T_object_current * T_grasp_in_object

We implement this using pure Python 3D transformations (no numpy dependency).
"""
import math
import rospy
import numpy as np


# ============================================================
# 3D Transform utilities (pure Python)
# ============================================================
def quat_multiply(q1, q2):
    """Multiply two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def quat_conjugate(q):
    """Conjugate of quaternion [x,y,z,w]."""
    return [-q[0], -q[1], -q[2], q[3]]


def quat_rotate(q, v):
    """Rotate vector v by quaternion q. q = [x,y,z,w], v = [x,y,z]."""
    qv = [v[0], v[1], v[2], 0.0]
    q_inv = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, qv), q_inv)
    return [result[0], result[1], result[2]]


def quat_inverse(q):
    """Inverse of a unit quaternion (same as conjugate for unit quat)."""
    return quat_conjugate(q)


def pose_inverse(position, orientation):
    """
    Compute inverse of a pose (position, orientation_xyzw).
    Returns (inv_position, inv_orientation).
    """
    q_inv = quat_inverse(orientation)
    # inv_position = -R_inv * position
    neg_pos = [-position[0], -position[1], -position[2]]
    inv_pos = quat_rotate(q_inv, neg_pos)
    return inv_pos, q_inv


def pose_compose(pos_a, ori_a, pos_b, ori_b):
    """
    Compose two poses: T_a * T_b
    Returns (composed_pos, composed_ori).
    """
    # Rotate b's position by a's orientation, then add a's position
    rotated_b_pos = quat_rotate(ori_a, pos_b)
    composed_pos = [pos_a[0] + rotated_b_pos[0],
                    pos_a[1] + rotated_b_pos[1],
                    pos_a[2] + rotated_b_pos[2]]
    composed_ori = quat_multiply(ori_a, ori_b)
    return composed_pos, composed_ori


# ============================================================
# Camera extrinsics (fallback estimates, TF is preferred)
# ============================================================
# Sawyer head_camera on the head pan/tilt unit.
# Computed from URDF kinematic chain (calc_camera_pose.py):
#   base -> right_l0 (j0=-0.27,z) -> head (head_pan=0,z) -> head_camera
# This is the camera LINK frame (not optical) — alignment.py handles REP 103 internally.
HEAD_CAMERA_IN_BASE = {
    "position": [0.0228, 0.0000, 0.5931],
    "orientation": [0.0000, -0.2756, -0.0000, -0.9613],
}

# Empirical correction for the current Sawyer Gazebo head-camera setup.
# This is a perception/extrinsic calibration layer: it corrects the object pose
# after camera->base projection, before MT3 computes the demo-relative grasp.
# Measured on 2026-06-29 from known Gazebo cube placements.
BASE_POSE_CALIBRATION_OFFSET = [0.006, 0.0, -0.015]
BASE_POSE_CALIBRATION_Y_POSITIVE = -0.021
BASE_POSE_CALIBRATION_Y_NEGATIVE = 0.015

# ROS optical→link frame conversion (standard REP 103)
# optical: z-forward, x-right, y-down
# link:    x-forward, y-left,  z-up
# R maps:  optical_z→link_x, optical_x→-link_y, optical_y→-link_z
OPTICAL_TO_LINK_QUAT = [-0.5, 0.5, -0.5, 0.5]  # [x,y,z,w]

WRIST_CAMERA_IN_BASE = {
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0],
}


def apply_base_pose_calibration(pos_base):
    """
    Correct systematic perception/extrinsic bias in base-frame object position.

    X/Z showed a stable translational bias across tested cube placements. Y was
    asymmetric around the camera center in the current Gazebo/head-camera setup,
    so use the sign of the raw projected Y for the smallest measured correction.
    """
    y_offset = (BASE_POSE_CALIBRATION_Y_POSITIVE
                if pos_base[1] >= 0.0
                else BASE_POSE_CALIBRATION_Y_NEGATIVE)
    return [
        pos_base[0] + BASE_POSE_CALIBRATION_OFFSET[0],
        pos_base[1] + y_offset,
        pos_base[2] + BASE_POSE_CALIBRATION_OFFSET[2],
    ], [BASE_POSE_CALIBRATION_OFFSET[0], y_offset, BASE_POSE_CALIBRATION_OFFSET[2]]


# ============================================================
# Alignment class
# ============================================================
class TrajectoryAligner:
    """
    Aligns a demonstration grasp to a detected object pose.

    The demo library stores:
      - grasp_pose_base_frame: absolute grasp in base frame (for the demo scene)
      - object_pose_in_demo: the object's position during the demo

    The alignment computes:
      T_grasp_in_obj = inv(T_obj_demo) * T_grasp_demo
      T_grasp_new = T_obj_detected * T_grasp_in_obj
    """

    def __init__(self, head_camera_extrinsics=None):
        if head_camera_extrinsics is None:
            head_camera_extrinsics = HEAD_CAMERA_IN_BASE
        self.head_camera_pose = head_camera_extrinsics
        self._tf_buffer = None
        self._tf_listener = None
        self._tf_source = "hardcoded_estimate"
        self.pointcloud_auto_z_correction = rospy.get_param(
            "~pointcloud_auto_z_correction", True)
        self.pointcloud_visible_z_fraction = float(rospy.get_param(
            "~pointcloud_visible_z_fraction", 0.42))
        self.pointcloud_min_z_offset = float(rospy.get_param(
            "~pointcloud_min_z_offset", -0.025))
        self.pointcloud_max_z_offset = float(rospy.get_param(
            "~pointcloud_max_z_offset", -0.006))
        self.pointcloud_base_z_offset = None
        if rospy.has_param("~pointcloud_base_z_offset"):
            self.pointcloud_base_z_offset = float(rospy.get_param(
                "~pointcloud_base_z_offset"))
        try:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        except ImportError:
            pass

    def _lookup_camera_pose_tf(self, camera_frame="head_camera", timeout_s=2.0):
        """Try to get camera→base transform from TF tree. Returns pose dict or None."""
        if self._tf_buffer is None:
            return None
        try:
            import rospy
            optical_frame = camera_frame + "_rgb_optical_frame"
            # Try all possible frame names — prefer optical frames first
            for src_frame in [camera_frame + "_optical", optical_frame,
                              camera_frame, "head_camera_link",
                              "head_camera_rgb_optical_frame"]:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        "base", src_frame, rospy.Time(0), rospy.Duration(timeout_s))
                    pos = tf.transform.translation
                    ori = tf.transform.rotation
                    rospy.loginfo(f"[TF] base→{src_frame} pos=[{pos.x:.4f},{pos.y:.4f},{pos.z:.4f}] "
                                  f"ori=[{ori.x:.4f},{ori.y:.4f},{ori.z:.4f},{ori.w:.4f}]")
                    return {
                        "position": [pos.x, pos.y, pos.z],
                        "orientation": [ori.x, ori.y, ori.z, ori.w],
                        "frame": src_frame
                    }
                except Exception:
                    continue
        except Exception:
            pass
        rospy.logwarn("[TF] No camera→base transform found in TF tree")
        return None

    def _apply_optical_to_link(self, pos_optical):
        """Convert position from camera optical frame to camera link frame."""
        return [pos_optical[2], -pos_optical[0], -pos_optical[1]]

    def _percentile(self, values, pct):
        if not values:
            return 0.0
        vals = sorted(values)
        idx = int(round((len(vals) - 1) * pct / 100.0))
        idx = max(0, min(len(vals) - 1, idx))
        return vals[idx]

    def _estimate_pointcloud_z_offset(self, pose_in_source, tf_pos, tf_ori):
        """Estimate center correction from the masked point cloud itself."""
        if self.pointcloud_base_z_offset is not None:
            return self.pointcloud_base_z_offset, abs(self.pointcloud_base_z_offset) / 0.31

        points = pose_in_source.get("object_points", [])
        if not self.pointcloud_auto_z_correction or len(points) < 10:
            return -0.014, 0.045

        z_values = []
        for p in points:
            if len(p) < 3:
                continue
            pos_base, _ = pose_compose(
                tf_pos, tf_ori, [float(p[0]), float(p[1]), float(p[2])],
                [0.0, 0.0, 0.0, 1.0])
            z_values.append(pos_base[2])

        if len(z_values) < 10:
            return -0.014, 0.045

        z10 = self._percentile(z_values, 10)
        z90 = self._percentile(z_values, 90)
        visible_z_span = max(0.0, z90 - z10)
        z_offset = -visible_z_span * self.pointcloud_visible_z_fraction
        z_offset = max(self.pointcloud_min_z_offset,
                       min(self.pointcloud_max_z_offset, z_offset))
        # This is only the visible object thickness in base z, not the full
        # physical height. Keep it as a conservative measurement instead of
        # inflating it; object size is estimated separately from point clouds.
        estimated_height = visible_z_span
        return z_offset, estimated_height

    def _estimate_pointcloud_size_base(self, pose_in_source, tf_pos, tf_ori):
        """Estimate masked point-cloud extents after transforming points to base."""
        points = pose_in_source.get("object_points", [])
        if len(points) < 10:
            return None

        base_points = []
        identity = [0.0, 0.0, 0.0, 1.0]
        for p in points:
            if len(p) < 3:
                continue
            try:
                pos_base, _ = pose_compose(
                    tf_pos, tf_ori,
                    [float(p[0]), float(p[1]), float(p[2])],
                    identity)
                base_points.append(pos_base)
            except Exception:
                continue

        if len(base_points) < 10:
            return None
        arr = np.asarray(base_points, dtype=np.float64)
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if len(arr) < 10:
            return None
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 90, axis=0)
        size = np.maximum(hi - lo, 0.0)
        if not np.all(np.isfinite(size)) or np.max(size) <= 0:
            return None
        return [float(v) for v in size.tolist()]

    def _transform_source_frame_to_base(self, pose_in_source):
        """Transform a pose already expressed in a named TF source frame to base."""
        source_frame = pose_in_source.get("source_frame")
        if not source_frame or self._tf_buffer is None:
            return None
        try:
            import rospy
            tf = self._tf_buffer.lookup_transform(
                "base", source_frame, rospy.Time(0), rospy.Duration(2.0))
            trans = tf.transform.translation
            rot = tf.transform.rotation
            tf_pos = [trans.x, trans.y, trans.z]
            tf_ori = [rot.x, rot.y, rot.z, rot.w]
            pos_base, ori_base = pose_compose(
                tf_pos,
                tf_ori,
                pose_in_source["position"],
                pose_in_source.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            )
            if "depth_pointcloud" in pose_in_source.get("method", ""):
                raw_z = pos_base[2]
                z_offset, estimated_height = self._estimate_pointcloud_z_offset(
                    pose_in_source, tf_pos, tf_ori)
                pos_base[2] += z_offset
                rospy.loginfo(
                    f"  pointcloud base z correction: "
                    f"{raw_z:.4f} -> {pos_base[2]:.4f} "
                    f"(offset={z_offset:.4f}, "
                    f"estimated_height={estimated_height:.4f})"
                )
                base_size = self._estimate_pointcloud_size_base(
                    pose_in_source, tf_pos, tf_ori)
                if base_size is not None:
                    rospy.loginfo(
                        "  pointcloud base size estimate: "
                        "[%.4f, %.4f, %.4f]",
                        base_size[0], base_size[1], base_size[2])
            self._tf_source = "tf_tree_direct"
            rospy.loginfo(
                f"  direct TF {source_frame}->base: "
                f"object_in_base={['%.4f' % v for v in pos_base]}"
            )
            result = {
                "position": pos_base,
                "orientation": ori_base,
                "tf_source": self._tf_source,
                "estimated_object_height": estimated_height if "depth_pointcloud" in pose_in_source.get("method", "") else None,
            }
            if "depth_pointcloud" in pose_in_source.get("method", "") and base_size is not None:
                result["estimated_object_size"] = base_size
                result["estimated_object_size_source"] = "base_pointcloud_p10_p90"
            elif pose_in_source.get("estimated_object_size") is not None:
                result["estimated_object_size"] = pose_in_source.get("estimated_object_size")
            return result
        except Exception as exc:
            try:
                import rospy
                rospy.logwarn(f"  direct TF {source_frame}->base failed: {exc}")
            except Exception:
                pass
            return None

    def transform_camera_to_base(self, pose_in_camera):
        """
        Transform a pose from camera optical frame to robot base frame.

        Chain: optical → link (REP 103) → base (TF link frame)
        """
        direct = self._transform_source_frame_to_base(pose_in_camera)
        if direct is not None:
            return direct

        pos_optical = pose_in_camera["position"]
        ori_cam = pose_in_camera["orientation"]

        # Step 1: optical → link (REP 103 conversion)
        pos_link = self._apply_optical_to_link(pos_optical)
        ori_link = quat_multiply(ori_cam, OPTICAL_TO_LINK_QUAT)

        # Step 2: get camera link frame in base from TF (or hardcoded fallback)
        camera_pose = self._lookup_camera_pose_tf()
        if camera_pose is not None:
            self.head_camera_pose = camera_pose
            self._tf_source = "tf_tree"
        else:
            camera_pose = self.head_camera_pose
            self._tf_source = "hardcoded"

        cam_pos = camera_pose["position"]
        cam_ori = camera_pose["orientation"]

        rospy.loginfo(f"  camera_link_in_base: pos={[f'{v:.4f}' for v in cam_pos]} "
                      f"src={self._tf_source} frame={camera_pose.get('frame','?')}")
        rospy.loginfo(f"  object_in_link: pos={[f'{v:.4f}' for v in pos_link]}")

        # Compose: object_in_base = camera_link_in_base * object_in_link
        pos_base_raw, ori_base = pose_compose(cam_pos, cam_ori, pos_link, ori_link)
        pos_base, calibration_offset = apply_base_pose_calibration(pos_base_raw)
        rospy.loginfo(f"  object_in_base_raw: pos={[f'{v:.4f}' for v in pos_base_raw]}")
        rospy.loginfo(
            f"  applied base calibration offset: "
            f"{[f'{v:.4f}' for v in calibration_offset]}"
        )

        rospy.loginfo(f"  object_in_base: pos={[f'{v:.4f}' for v in pos_base]} "
                      f"z={pos_base[2]:.4f}")

        return {"position": pos_base, "orientation": ori_base, "tf_source": self._tf_source}

    def compute_aligned_grasp(self, demo_grasp_pose, demo_object_position, detected_object_pose,
                              rotate_relative=True):
        """
        Compute aligned grasp pose for the current scene.

        Args:
            demo_grasp_pose: {"position": [x,y,z], "orientation": [x,y,z,w]} in base frame
            demo_object_position: [x,y,z] of object during demo (in base frame)
            detected_object_pose: {"position": [x,y,z], "orientation": [x,y,z,w]} in base frame

        Returns:
            aligned_grasp: {"position": [x,y,z], "orientation": [x,y,z,w]} in base frame
        """
        # Object orientation in demo (assume upright)
        demo_obj_ori = [0.0, 0.0, 0.0, 1.0]

        # Step 1: T_grasp_in_object = inv(T_obj_demo) * T_grasp_demo
        inv_obj_pos, inv_obj_ori = pose_inverse(demo_object_position, demo_obj_ori)
        grasp_rel_pos, grasp_rel_ori = pose_compose(
            inv_obj_pos, inv_obj_ori,
            demo_grasp_pose["position"], demo_grasp_pose["orientation"]
        )

        # Step 2: T_grasp_aligned = T_obj_current * T_grasp_in_object
        det_obj_pos = detected_object_pose["position"]
        det_obj_ori = detected_object_pose.get("orientation", [0.0, 0.0, 0.0, 1.0])

        if rotate_relative:
            aligned_pos, aligned_ori = pose_compose(
                det_obj_pos, det_obj_ori,
                grasp_rel_pos, grasp_rel_ori
            )
        else:
            # For upright tabletop top-down grasps, the visual pose estimate can
            # include noisy cube rotations. Keep the demonstrated base-frame
            # offset vertical instead of rotating it below/sideways.
            aligned_pos = [
                det_obj_pos[0] + grasp_rel_pos[0],
                det_obj_pos[1] + grasp_rel_pos[1],
                det_obj_pos[2] + grasp_rel_pos[2],
            ]
            aligned_ori = demo_grasp_pose["orientation"]

        return {
            "position": aligned_pos,
            "orientation": aligned_ori,
            "relative_grasp": {
                "position": grasp_rel_pos,
                "orientation": grasp_rel_ori,
            }
        }

    def align(self, demo, detected_object_pose_in_camera):
        """
        Full alignment pipeline:
          1. Transform detected pose from camera to base frame
          2. Get demo grasp and demo object position
          3. Compute aligned grasp

        Args:
            demo: demo entry from DemoLibrary
            detected_object_pose_in_camera: pose from PoseEstimator

        Returns:
            aligned_grasp_pose in base frame, ready for grasp execution
        """
        # Transform to base frame
        obj_pose_base = self.transform_camera_to_base(detected_object_pose_in_camera)

        # Get demo data
        demo_grasp = {
            "position": [
                demo["grasp_pose_base_frame"]["position_m"]["x"],
                demo["grasp_pose_base_frame"]["position_m"]["y"],
                demo["grasp_pose_base_frame"]["position_m"]["z"],
            ],
            "orientation": [
                demo["grasp_pose_base_frame"]["orientation_xyzw"]["x"],
                demo["grasp_pose_base_frame"]["orientation_xyzw"]["y"],
                demo["grasp_pose_base_frame"]["orientation_xyzw"]["z"],
                demo["grasp_pose_base_frame"]["orientation_xyzw"]["w"],
            ]
        }
        obj_frame = demo.get("object_pose_base_frame", {}).get("position_m", None)
        if obj_frame is not None:
            demo_obj_pos = [obj_frame["x"], obj_frame["y"], obj_frame["z"]]
        else:
            rel = demo.get("grasp_relative_to_object", {}).get("delta_position_m", [0.0, 0.0, 0.0])
            demo_obj_pos = [
                demo_grasp["position"][0] - rel[0],
                demo_grasp["position"][1] - rel[1],
                demo_grasp["position"][2] - rel[2],
            ]

        # Log: compare computed vs demo z for diagnostics
        obj_z = obj_pose_base.get("position", [0, 0, 0])[2]
        rospy.loginfo(f"  object_z in base: {obj_z:.4f} (demo_z={demo_obj_pos[2]:.4f}, "
                      f"diff={abs(obj_z-demo_obj_pos[2]):.4f}m, src={self._tf_source})")

        # Get approach direction from demo
        approach = demo.get("approach_direction", [0.0, 0.0, -1.0])
        retract = demo.get("retract_direction", [0.0, 0.0, 1.0])
        rotate_relative = not (
            abs(approach[0]) < 1e-6 and abs(approach[1]) < 1e-6 and approach[2] < -0.5
        )

        # Compute aligned grasp — use computed pose directly, no fallback
        aligned = self.compute_aligned_grasp(
            demo_grasp, demo_obj_pos, obj_pose_base,
            rotate_relative=rotate_relative
        )
        rospy.loginfo(f"  relative grasp rotation: {'enabled' if rotate_relative else 'disabled for top-down'}")

        return {
            "grasp_pose": aligned,
            "object_pose_base": obj_pose_base,
            "approach_direction": approach,
            "retract_direction": retract,
            "gripper_opening": demo.get("gripper_opening_m", 0.07),
            "tf_source": self._tf_source,
        }


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    aligner = TrajectoryAligner()

    # Simulate a demo and detected object
    demo = {
        "grasp_pose_base_frame": {
            "position_m": {"x": 0.6, "y": 0.0, "z": -0.53},
            "orientation_xyzw": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
        },
        "object_pose_base_frame": {
            "position_m": {"x": 0.6, "y": 0.0, "z": -0.58}
        }
    }

    # Detected object pose in camera frame (simulated: cube at 0.5m in front of camera)
    detected = {
        "position": [0.0, 0.0, 0.5],
        "orientation": [0.0, 0.0, 0.0, 1.0]
    }

    result = aligner.align(demo, detected)
    pos = result["grasp_pose"]["position"]
    ori = result["grasp_pose"]["orientation"]
    obj_pos = result["object_pose_base"]["position"]

    print("=== MT3 Alignment Test ===")
    print(f"Detected object (camera): pos={detected['position']}")
    print(f"Object in base frame: pos=[{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"Aligned grasp pose: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
          f"ori=[{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
    print(f"Approach: {result['approach_direction']}")
    print(f"Gripper opening: {result['gripper_opening']}m")
