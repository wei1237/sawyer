#!/usr/bin/env python3
"""
MT3 Perception Module

Object detection and 6-DoF pose estimation for the 4.5cm cube.
Uses two cameras (MT3 paper approach):
  - head_camera: wide-angle global scene view for initial detection
  - right_hand_camera: wrist-mounted for fine alignment when near object

Detection pipeline:
  1. Color-based segmentation (HSV thresholding for red)
  2. Contour extraction + polygon filtering
  3. Bounding box fitting
  4. PnP pose estimation using known cube geometry + camera intrinsics
"""
import rospy
import math
import os
import struct

import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge


PERCEPTION_BUILD_MARKER = "2026-08-20_mt3_perception_strict_pointcloud_v24"


# ============================================================
# Pure-Python geometry helpers
# ============================================================
def _quaternion_from_rotation_matrix(R):
    """Convert 3x3 rotation matrix to [x,y,z,w] quaternion."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return [x, y, z, w]


# ============================================================
# Cube detection using OpenCV
# ============================================================
class CubeDetector:
    """Detect green cube in camera images using color thresholding."""

    def __init__(self, cube_size_m=0.045, debug=False):
        self.cube_size = cube_size_m
        self.half_size = cube_size_m / 2.0
        self.debug = debug
        # Green HSV range — tightened S/V lower bounds for tighter contour
        # (loose bounds make the bbox too large → PnP depth underestimates)
        self.green_lower = (40, 55, 55)
        self.green_upper = (80, 255, 255)
        self._hsv_diag_printed = False
        self.bridge = CvBridge()
        self._cv2 = None
        self._np = None
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
        except ImportError:
            rospy.logwarn("[CubeDetector] OpenCV not available")

    def detect(self, ros_image_msg):
        """
        Detect cube in ROS Image message.
        Returns (detections, bgr_image, hsv_mask) tuple.
        detections: list of {bbox_2d, center_2d, size_2d, angle_2d, area, extent, confidence}
        """
        if self._cv2 is None:
            rospy.logwarn_throttle(10, "[CubeDetector] OpenCV required for detection")
            return [], None, None

        cv_img = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="bgr8")
        hsv = self._cv2.cvtColor(cv_img, self._cv2.COLOR_BGR2HSV)

        # One-time diagnostic: print full-image and masked HSV stats
        if self.debug and not self._hsv_diag_printed:
            self._hsv_diag_printed = True
            h, w = cv_img.shape[:2]
            rospy.loginfo("[CubeDetector] === HSV DIAGNOSTIC ===")
            rospy.loginfo(f"  Image size: {w}x{h}")
            rospy.loginfo(f"  Full image H: [{hsv[:,:,0].min()}, {hsv[:,:,0].max()}]"
                          f"  S: [{hsv[:,:,1].min()}, {hsv[:,:,1].max()}]"
                          f"  V: [{hsv[:,:,2].min()}, {hsv[:,:,2].max()}]")
            # Scan image in 4x4 grid to locate green regions
            rospy.loginfo("  4x4 grid H-mean scan (green=60±30):")
            for row in range(4):
                parts = []
                for col in range(4):
                    y1, y2 = row * h // 4, (row + 1) * h // 4
                    x1, x2 = col * w // 4, (col + 1) * w // 4
                    cell_h = hsv[y1:y2, x1:x2, 0]
                    cell_s = hsv[y1:y2, x1:x2, 1]
                    cell_v = hsv[y1:y2, x1:x2, 2]
                    # Count green-ish pixels in cell
                    gm = (cell_h >= 35) & (cell_h <= 90) & (cell_s >= 30) & (cell_v >= 30)
                    gp = self._np.count_nonzero(gm)
                    h_mean = int(cell_h.mean())
                    parts.append(f"g{gp:5d}/h{h_mean:3d}")
                rospy.loginfo(f"    row{row}: " + " | ".join(parts))
            # BGR values near the image centre
            cy, cx = h // 2, w // 2
            patch = hsv[cy-40:cy+40, cx-40:cx+40, :]
            rospy.loginfo(f"  Centre patch H: [{patch[:,:,0].min()}, {patch[:,:,0].max()}]"
                          f"  S: [{patch[:,:,1].min()}, {patch[:,:,1].max()}]"
                          f"  V: [{patch[:,:,2].min()}, {patch[:,:,2].max()}]")
            bgr_patch = cv_img[cy-40:cy+40, cx-40:cx+40, :]
            rospy.loginfo(f"  Centre BGR B:[{bgr_patch[:,:,0].min()},{bgr_patch[:,:,0].max()}]"
                          f" G:[{bgr_patch[:,:,1].min()},{bgr_patch[:,:,1].max()}]"
                          f" R:[{bgr_patch[:,:,2].min()},{bgr_patch[:,:,2].max()}]")
            rospy.loginfo("[CubeDetector] ======================")

        mask = self._cv2.inRange(hsv, self.green_lower, self.green_upper)
        raw_green_px = self._cv2.countNonZero(mask)

        kernel = self._cv2.getStructuringElement(self._cv2.MORPH_ELLIPSE, (5, 5))
        mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_OPEN, kernel)
        mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_CLOSE, kernel)
        clean_green_px = self._cv2.countNonZero(mask)

        contours, _ = self._cv2.findContours(mask, self._cv2.RETR_EXTERNAL,
                                              self._cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        img_area = cv_img.shape[0] * cv_img.shape[1]
        min_area = img_area * 0.0002
        max_area = img_area * 0.5

        # Diagnostic: report green pixel counts and contour filtering
        if self.debug and not self._hsv_diag_printed:
            # _hsv_diag_printed already True from above, but we want per-frame diag for this
            pass

        # Per-call diagnostic (always prints when debug=True, not just once)
        if self.debug:
            rejected = {"area_small": 0, "area_large": 0, "dim_tiny": 0,
                        "aspect": 0, "extent": 0}
            for cnt in contours:
                area = self._cv2.contourArea(cnt)
                if area < min_area:
                    rejected["area_small"] += 1; continue
                if area > max_area:
                    rejected["area_large"] += 1; continue
                rect = self._cv2.minAreaRect(cnt)
                (rw, rh) = rect[1]
                if min(rw, rh) < 5:
                    rejected["dim_tiny"] += 1; continue
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                if aspect > 2.5:
                    rejected["aspect"] += 1; continue
                rect_area = rw * rh
                extent = area / max(rect_area, 1)
                if extent < 0.5:
                    rejected["extent"] += 1; continue
            if raw_green_px > 0 or len(contours) > 0:
                rospy.loginfo(f"[CubeDetector] Green px: raw={raw_green_px} → morph={clean_green_px}"
                              f" | contours={len(contours)}"
                              f" | rejected: {rejected}"
                              f" | limits: area=[{min_area:.0f}, {max_area:.0f}]")

        for cnt in contours:
            area = self._cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            rect = self._cv2.minAreaRect(cnt)
            box = self._cv2.boxPoints(rect)
            box = self._np.int0(box)

            (rw, rh) = rect[1]
            if min(rw, rh) < 5:
                continue
            aspect = max(rw, rh) / max(min(rw, rh), 1)
            if aspect > 2.5:
                continue

            rect_area = rw * rh
            extent = area / max(rect_area, 1)
            if extent < 0.5:
                continue

            detections.append({
                "bbox_2d": box.tolist(),
                "center_2d": list(rect[0]),
                "size_2d": list(rect[1]),
                "angle_2d": rect[2],
                "area": float(area),
                "extent": float(extent),
                "confidence": float(min(extent, 1.0) * min(1.0 / max(aspect - 0.8, 0.1), 1.0))
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        if self.debug and detections:
            rospy.loginfo(f"[CubeDetector] Found {len(detections)} cube candidates, "
                          f"best confidence={detections[0]['confidence']:.2f}")

        return detections, cv_img, mask


# ============================================================
# Pose estimation using PnP
# ============================================================
class PoseEstimator:
    """Estimate 6-DoF pose of cube from 2D detections using PnP."""

    def __init__(self, cube_size_m=0.045):
        self.cube_size = cube_size_m
        self.half = cube_size_m / 2.0
        self.camera_matrix = None
        self.dist_coeffs = None
        self._cv2 = None
        self._np = None
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
        except ImportError:
            pass

    def set_camera_info(self, camera_info_msg):
        """Set camera intrinsics from ROS CameraInfo message."""
        K = camera_info_msg.K
        if self._np:
            self.camera_matrix = self._np.array([
                [K[0], K[1], K[2]],
                [K[3], K[4], K[5]],
                [K[6], K[7], K[8]]
            ], dtype=self._np.float64)
        else:
            self.camera_matrix = [
                [K[0], K[1], K[2]],
                [K[3], K[4], K[5]],
                [K[6], K[7], K[8]]
            ]
        if camera_info_msg.D:
            if self._np:
                self.dist_coeffs = self._np.array(camera_info_msg.D, dtype=self._np.float64)
            else:
                self.dist_coeffs = list(camera_info_msg.D)
        else:
            self.dist_coeffs = None

    def set_camera_intrinsics(self, fx, fy, cx, cy):
        """Set camera intrinsics directly (no ROS message needed)."""
        if self._np:
            self.camera_matrix = self._np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=self._np.float64)
        else:
            self.camera_matrix = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ]
        self.dist_coeffs = None

    def get_object_points(self):
        """Return 8 cube corner points in object frame (for PnP and visualization)."""
        h = self.half
        return [
            [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
            [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
        ]

    def estimate_pose(self, detection):
        """
        Estimate 6-DoF pose using PnP.
        Returns: {position: [x,y,z], orientation: [x,y,z,w], confidence, method}
                 in camera optical frame, plus object_points for visualization.
        """
        if self._cv2 is None or self.camera_matrix is None:
            return self._estimate_pose_simple(detection)

        h = self.half
        object_points = self._np.array(self.get_object_points(), dtype=self._np.float64)

        bbox = detection["bbox_2d"]
        if len(bbox) < 4:
            return None

        image_points = self._np.array(bbox[:4], dtype=self._np.float64)
        center = self._np.mean(image_points, axis=0)

        obj_top_face = object_points[:4]
        obj_center = self._np.mean(obj_top_face, axis=0)
        obj_angles = self._np.arctan2(obj_top_face[:, 1] - obj_center[1],
                                       obj_top_face[:, 0] - obj_center[0])
        obj_order = self._np.argsort(obj_angles)
        obj_top_ordered = obj_top_face[obj_order]

        img_angles = self._np.arctan2(image_points[:, 1] - center[1],
                                       image_points[:, 0] - center[0])
        img_order = self._np.argsort(img_angles)
        img_ordered = image_points[img_order]

        try:
            success, rvec, tvec = self._cv2.solvePnP(
                obj_top_ordered, img_ordered,
                self.camera_matrix, self.dist_coeffs,
                flags=self._cv2.SOLVEPNP_IPPE
            )
            if not success:
                return self._estimate_pose_simple(detection)

            R, _ = self._cv2.Rodrigues(rvec)
            quat = _quaternion_from_rotation_matrix(R.tolist())
            pos = tvec.flatten().tolist()

            # Transform object points to camera frame for visualization
            obj_pts_cam = (R @ object_points.T + tvec).T.tolist()

            return {
                "position": pos,
                "orientation": quat,
                "object_points_cam": obj_pts_cam,
                "confidence": detection.get("confidence", 0.5),
                "method": "PnP"
            }
        except Exception as e:
            rospy.logwarn(f"[PoseEstimator] PnP failed: {e}")
            return self._estimate_pose_simple(detection)

    def _estimate_pose_simple(self, detection):
        """Fallback: estimate depth from bounding box size."""
        if self.camera_matrix is None:
            rospy.logwarn("[PoseEstimator] No camera intrinsics, cannot estimate pose")
            return None

        if isinstance(self.camera_matrix, list):
            fx = self.camera_matrix[0][0]
            fy = self.camera_matrix[1][1]
            cx = self.camera_matrix[0][2]
            cy = self.camera_matrix[1][2]
        else:
            fx = float(self.camera_matrix[0][0])
            fy = float(self.camera_matrix[1][1])
            cx = float(self.camera_matrix[0][2])
            cy = float(self.camera_matrix[1][2])

        size_2d = detection.get("size_2d", [50, 50])
        center_2d = detection.get("center_2d", [cx, cy])
        avg_pixel_size = (size_2d[0] + size_2d[1]) / 2.0
        f_avg = (fx + fy) / 2.0
        depth = f_avg * self.cube_size / max(avg_pixel_size, 1.0)
        x = (center_2d[0] - cx) * depth / fx
        y = (center_2d[1] - cy) * depth / fy
        z = depth

        # Simple object points in camera frame
        h = self.half
        pts = [[x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h],
               [x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h]]

        return {
            "position": [x, y, z],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "object_points_cam": pts,
            "confidence": detection.get("confidence", 0.3) * 0.7,
            "method": "simple_projection"
        }


# ============================================================
# Perception node
# ============================================================
class PerceptionNode:
    """
    ROS node that subscribes to head camera and runs detection + pose estimation.
    Provides raw data for external visualization modules.
    """

    def __init__(self, cube_size_m=0.045):
        self.detector = CubeDetector(cube_size_m=cube_size_m, debug=True)
        self.pose_estimator = PoseEstimator(cube_size_m=cube_size_m)
        # Set known intrinsics immediately (from sawyer_base.gazebo.xacro)
        # fx=fy=407.391526, cx=640.5, cy=400.5, 1280x800
        self.pose_estimator.set_camera_intrinsics(407.391526, 407.391526, 640.5, 400.5)

        self.build_marker = PERCEPTION_BUILD_MARKER
        self.use_pointcloud_pose = rospy.get_param("~use_pointcloud_pose", False)
        self.strict_pointcloud_pose = rospy.get_param(
            "~strict_pointcloud_pose", False)
        self.langsam_mask_path = rospy.get_param(
            "~langsam_mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")
        self.pointcloud_topic = rospy.get_param(
            "~pointcloud_topic", "/io/internal_camera/head_camera/depth/points")
        self.pointcloud_center_depth_offset = float(rospy.get_param(
            "~pointcloud_center_depth_offset", 0.0))

        # Latest data
        self.head_image = None
        self.head_depth = None
        self.head_points = None
        self.head_camera_info = None
        self.wrist_image = None
        self.wrist_camera_info = None
        self.latest_detection = None
        self.latest_pose = None
        self.latest_bgr = None
        self.latest_mask = None
        self.latest_clean_mask = None

        # Subscribers
        self.head_img_sub = rospy.Subscriber(
            "/head_camera/image_raw", Image, self._head_image_cb, queue_size=5)
        self.head_depth_sub = rospy.Subscriber(
            "/head_camera/depth/image_raw", Image, self._head_depth_cb, queue_size=5)
        self.head_info_sub = rospy.Subscriber(
            "/io/internal_camera/head_camera/camera_info", CameraInfo, self._head_info_cb, queue_size=5)
        self.head_points_sub = rospy.Subscriber(
            self.pointcloud_topic, PointCloud2, self._head_points_cb, queue_size=2)
        self.wrist_img_sub = rospy.Subscriber(
            "/right_hand_camera/image_raw", Image, self._wrist_image_cb, queue_size=5)
        self.wrist_info_sub = rospy.Subscriber(
            "/right_hand_camera/camera_info", CameraInfo, self._wrist_info_cb, queue_size=5)

        rospy.loginfo("[PerceptionNode] Subscribed to head_camera (RGB+Depth+PointCloud) and right_hand_camera")
        if self.use_pointcloud_pose:
            rospy.loginfo(
                "[PerceptionNode] Using LangSAM mask + PointCloud2 pose path "
                "(strict=%s, topic=%s)",
                self.strict_pointcloud_pose, self.pointcloud_topic)

    def _head_image_cb(self, msg):
        self.head_image = msg

    def _head_depth_cb(self, msg):
        self.head_depth = msg

    def _head_points_cb(self, msg):
        self.head_points = msg

    def _head_info_cb(self, msg):
        self.head_camera_info = msg
        self.pose_estimator.set_camera_info(msg)

    def _wrist_image_cb(self, msg):
        self.wrist_image = msg

    def _wrist_info_cb(self, msg):
        self.wrist_camera_info = msg

    def detect_with_head_camera(self):
        """Run detection on latest head camera image."""
        if self.head_image is None:
            rospy.logwarn_throttle(5, "[PerceptionNode] No head camera image yet")
            return None

        detections, bgr, mask = self.detector.detect(self.head_image)
        self.latest_bgr = bgr
        self.latest_mask = mask

        if not detections:
            self.latest_clean_mask = None
            return None

        self.latest_detection = detections[0]

        # Create a CLEAN mask using only the detection bbox region.
        # This filters out red robot-arm pixels that the raw HSV mask picks up.
        bbox = self.latest_detection.get("bbox_2d")
        if bbox is not None and bgr is not None:
            clean = self.detector._np.zeros(bgr.shape[:2], dtype=self.detector._np.uint8)
            pts = self.detector._np.array(bbox, dtype=self.detector._np.int32)
            self.detector._cv2.fillPoly(clean, [pts], 255)
            # Intersect with original mask: keep only red pixels inside bbox
            if mask is not None:
                clean = self.detector._cv2.bitwise_and(clean, clean, mask=mask)
            self.latest_clean_mask = clean
        else:
            self.latest_clean_mask = mask

        return self.latest_detection

    def estimate_pose_with_head(self):
        """Estimate pose using head camera detection + depth sensor for accurate Z."""
        if self.use_pointcloud_pose:
            pose = self.estimate_pose_with_pointcloud_mask()
            if pose is not None:
                self.latest_pose = pose
                return pose
            if self.strict_pointcloud_pose:
                rospy.logerr(
                    "[PerceptionNode] STRICT PointCloud2 pose failed; "
                    "refusing HSV/PnP/depth fallback")
                self.latest_pose = None
                return None
            rospy.logwarn(
                "[PerceptionNode] PointCloud2 pose failed; "
                "falling back to HSV+PnP+depth")

        if self.latest_detection is None:
            if not self.detect_with_head_camera():
                return None

        pose = self.pose_estimator.estimate_pose(self.latest_detection)
        if pose is not None:
            pose = self._correct_depth_from_sensor(pose)
        self.latest_pose = pose
        return pose

    def _resize_mask_nearest(self, mask, target_height, target_width):
        if mask.shape == (target_height, target_width):
            return mask
        y_idx = np.floor(np.arange(target_height) * mask.shape[0] / target_height).astype(int)
        x_idx = np.floor(np.arange(target_width) * mask.shape[1] / target_width).astype(int)
        y_idx = np.clip(y_idx, 0, mask.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, mask.shape[1] - 1)
        return mask[y_idx[:, None], x_idx[None, :]]

    def _point_field_offsets(self, cloud_msg):
        offsets = {field.name: field.offset for field in cloud_msg.fields}
        for name in ("x", "y", "z"):
            if name not in offsets:
                raise RuntimeError("PointCloud2 missing '{}' field".format(name))
        return offsets["x"], offsets["y"], offsets["z"]

    def _read_xyz_at_indices(self, cloud_msg, indices):
        x_off, y_off, z_off = self._point_field_offsets(cloud_msg)
        endian = ">" if cloud_msg.is_bigendian else "<"
        data = cloud_msg.data
        step = cloud_msg.point_step
        points = []
        for idx in indices:
            base = int(idx) * step
            if base + max(x_off, y_off, z_off) + 4 > len(data):
                continue
            x = struct.unpack_from(endian + "f", data, base + x_off)[0]
            y = struct.unpack_from(endian + "f", data, base + y_off)[0]
            z = struct.unpack_from(endian + "f", data, base + z_off)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append((x, y, z))
        return points

    def _cloud_size_for_mask(self, cloud_msg, mask_width):
        if cloud_msg.height > 1:
            return cloud_msg.width, cloud_msg.height
        if cloud_msg.width % mask_width != 0:
            raise RuntimeError(
                "Flattened PointCloud2 width {} does not match mask width {}".format(
                    cloud_msg.width, mask_width))
        return mask_width, cloud_msg.width // mask_width

    def _detection_from_mask(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        area = float(len(xs))
        extent = area / float(width * height)
        return {
            "bbox_2d": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "center_2d": [float(xs.mean()), float(ys.mean())],
            "size_2d": [float(width), float(height)],
            "angle_2d": 0.0,
            "area": area,
            "extent": float(extent),
            "confidence": 0.97,
        }

    def estimate_pose_with_pointcloud_mask(self):
        """Estimate object position from an external LangSAM mask and PointCloud2."""
        if self.head_points is None:
            rospy.logwarn_throttle(3, "[PointCloud2] No point cloud message yet")
            return None
        if not os.path.exists(self.langsam_mask_path):
            rospy.logwarn("[PointCloud2] LangSAM mask file not found: %s", self.langsam_mask_path)
            return None

        try:
            mask = np.load(self.langsam_mask_path).astype(bool)
            cloud_width, cloud_height = self._cloud_size_for_mask(self.head_points, mask.shape[1])
            mask = self._resize_mask_nearest(mask, cloud_height, cloud_width)
            ys, xs = np.where(mask)
            if len(xs) < 5:
                rospy.logwarn("[PointCloud2] Mask too small: %d px", len(xs))
                return None

            indices = ys.astype(np.int64) * cloud_width + xs.astype(np.int64)
            points = self._read_xyz_at_indices(self.head_points, indices)
            if len(points) < 5:
                rospy.logwarn("[PointCloud2] Too few valid points in mask: %d", len(points))
                return None

            points = np.asarray(points, dtype=np.float64)
            surface_center = np.median(points, axis=0)
            center = surface_center.copy()
            center[2] += self.pointcloud_center_depth_offset
            spread = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)

            clean_mask = (mask.astype(np.uint8) * 255)
            self.latest_clean_mask = clean_mask
            self.latest_mask = clean_mask
            self.latest_detection = self._detection_from_mask(mask)
            mask_bbox_2d = None
            mask_center_2d = None
            mask_pixels_2d = None
            max_mask_pixels = int(rospy.get_param(
                "~mask_pixels_2d_max", 6000))
            if len(xs) > 0 and max_mask_pixels > 0:
                stride = max(1, int(math.ceil(float(len(xs)) / max_mask_pixels)))
                mask_pixels_2d = np.column_stack(
                    [xs[::stride], ys[::stride]]).astype(float).tolist()
            if self.latest_detection is not None:
                mask_bbox_2d = self.latest_detection.get("bbox_2d")
                bbox = mask_bbox_2d or []
                if len(bbox) >= 4:
                    mask_center_2d = [
                        0.25 * sum(float(p[0]) for p in bbox[:4]),
                        0.25 * sum(float(p[1]) for p in bbox[:4]),
                    ]
            if self.head_image is not None and self.latest_bgr is None:
                self.latest_bgr = self.detector.bridge.imgmsg_to_cv2(
                    self.head_image, desired_encoding="bgr8")

            rospy.loginfo(
                "[PointCloud2] mask points=%d surface_cam=[%.4f, %.4f, %.4f] "
                "center_cam=[%.4f, %.4f, %.4f] offset_z=%.3f "
                "spread=[%.4f, %.4f, %.4f]",
                len(points), surface_center[0], surface_center[1], surface_center[2],
                center[0], center[1], center[2], self.pointcloud_center_depth_offset,
                spread[0], spread[1], spread[2])

            return {
                "position": center.tolist(),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "confidence": 0.97,
                "method": "LangSAM+depth_pointcloud",
                "estimated_object_size": spread.tolist(),
                "estimated_object_size_source": "pointcloud_spread_source_frame_p10_p90",
                "object_points": points.tolist(),
                "mask_bbox_2d": mask_bbox_2d,
                "mask_center_2d": mask_center_2d,
                "mask_pixels_2d": mask_pixels_2d,
                "cloud_size": [int(cloud_width), int(cloud_height)],
                "source_frame": self.head_points.header.frame_id,
            }
        except Exception as e:
            rospy.logwarn("[PointCloud2] Failed: %s", e)
            return None

    def _correct_depth_from_sensor(self, pose):
        """Replace PnP-estimated depth with real depth from depth camera.
        Uses median depth over the cube's detection mask region for robustness."""
        if self.head_depth is None or self.latest_detection is None:
            return pose
        try:
            depth_img = self.detector.bridge.imgmsg_to_cv2(
                self.head_depth, desired_encoding="passthrough")
            # Use clean_mask (bbox ∩ green mask) to sample depth only on the cube
            mask = self.latest_clean_mask
            if mask is not None and self.detector._np is not None:
                # Get depth values only where the cube mask is active
                depth_vals = depth_img[mask > 0]
            else:
                # Fallback: sample a small patch around the detection center
                center = self.latest_detection.get("center_2d")
                cx, cy = int(center[0]), int(center[1])
                r = 5
                h, w = depth_img.shape[:2]
                patch = depth_img[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
                depth_vals = patch.flatten()

            # Filter: non-nan, non-zero, in reasonable range
            np_dep = self.detector._np
            total = len(depth_vals)
            n_nan = int(np_dep.sum(np_dep.isnan(depth_vals))) if total > 0 else 0
            valid = depth_vals[~np_dep.isnan(depth_vals)]
            rospy.loginfo(f"  [Depth diag] total={total} nan={n_nan} "
                          f"min={np_dep.min(valid) if len(valid) else '?'} "
                          f"max={np_dep.max(valid) if len(valid) else '?'} "
                          f"enc={self.head_depth.encoding}")
            valid = valid[(valid > 0.1) & (valid < 10.0)]
            if len(valid) < 3:
                rospy.logwarn(f"  [Depth] Not enough valid depth pixels ({len(valid)}/{total})")
                return pose

            real_depth = float(self.detector._np.median(valid))
            pnp_pos = pose["position"]
            pnp_depth = pnp_pos[2]
            if abs(pnp_depth) < 0.01:
                return pose

            scale = real_depth / pnp_depth
            new_pos = [pnp_pos[0] * scale, pnp_pos[1] * scale, real_depth]
            rospy.loginfo(f"  [Depth] PnP z={pnp_depth:.3f} -> sensor z={real_depth:.3f} "
                          f"(scale={scale:.3f}, {len(valid)} valid px)")
            pose["position"] = new_pos
            pose["method"] = pose.get("method", "PnP") + "+depth"
        except Exception as e:
            rospy.logwarn(f"  [Depth] Failed: {e}")
        return pose

    def get_object_pose(self):
        """
        Main interface: detect object and return its 6-DoF pose in camera frame.
        Returns None if no object detected.
        """
        return self.estimate_pose_with_head()

    def get_debug_data(self):
        """Return raw detection data for external visualization."""
        return {
            "bgr_image": self.latest_bgr,
            "hsv_mask": self.latest_clean_mask,  # clean mask (bbox-filtered)
            "detection": self.latest_detection,
            "pose": self.latest_pose,
            "camera_info": self.head_camera_info,
        }

    def get_detected_features(self):
        """Extract geometric features from detection for demo library matching."""
        if self.latest_detection is None:
            return None

        det = self.latest_detection
        return {
            "shape": "box",
            "dimensions_m": [self.detector.cube_size] * 3,
            "aspect_ratio": [1.0, 1.0, 1.0],
            "color_rgb": [0.0, 1.0, 0.0],
            "bounding_box_area": float(det.get("area", 0)),
            "extent": float(det.get("extent", 0)),
            "confidence": float(det.get("confidence", 0)),
        }


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    rospy.init_node("mt3_perception_test")
    node = PerceptionNode()
    rospy.loginfo("MT3 Perception node running. Waiting for images...")

    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        pose = node.get_object_pose()
        if pose:
            pos = pose["position"]
            rospy.loginfo(f"Object pose: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                          f"method={pose['method']} conf={pose['confidence']:.2f}")
        else:
            rospy.loginfo_throttle(10, "No object detected")
        rate.sleep()
