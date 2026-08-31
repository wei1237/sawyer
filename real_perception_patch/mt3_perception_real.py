#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-camera perception for MT3 on Sawyer + ASC60C/HP60C.

This module is intentionally separate from mt3_perception.py so the existing
Gazebo/head-camera path remains untouched.

Primary real-robot path:
    external LangSAM bool mask
      + registered ASC60C depth image
      + dynamic CameraInfo
      -> camera-frame point cloud / object position

The ASC60C PointCloud2 topic is NOT used for mask pixel indexing because the
real camera publishes a sparse/unorganized cloud.  A compatibility method named
``estimate_pose_with_pointcloud_mask`` is kept, but on the real robot it routes
to the registered-depth implementation below.
"""

import math
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from mt3_perception import CubeDetector, PoseEstimator
from mt3_scene_package import pointcloud_from_depth_mask


DEFAULT_RGB_TOPIC = "/ascamera_hp60c/rgb0/image"
DEFAULT_DEPTH_TOPIC = "/ascamera_hp60c/depth0/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/ascamera_hp60c/rgb0/camera_info"
DEFAULT_POINTCLOUD_TOPIC = "/ascamera_hp60c/depth0/points"
DEFAULT_CAMERA_FRAME = "ascamera_hp60c_color_0"
DEFAULT_MASK_PATH = "/mnt/hgfs2/ascamera_data/current_mask.npy"


def _global_real_param_name(name):
    return "/sawyer_auto_grasp/%s" % str(name).lstrip("~/")


def _param(name, default=None):
    """Private node param first, then shared real-robot YAML namespace."""
    private = "~%s" % str(name).lstrip("~/")
    if rospy.has_param(private):
        return rospy.get_param(private)
    return rospy.get_param(_global_real_param_name(name), default)


def _param_bool(name, default=False):
    value = _param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _depth_to_meters(depth):
    """Normalize ROS depth arrays to float64 meters.

    ASC60C depth is currently uint16 millimeters.  Float depth images in meters
    are also accepted so the code stays usable if the driver configuration
    changes later.
    """
    arr = np.asarray(depth)
    if arr.size == 0:
        return arr.astype(np.float64)
    out = arr.astype(np.float64)
    finite = out[np.isfinite(out)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return out
    if np.issubdtype(arr.dtype, np.integer) or float(np.median(finite)) > 20.0:
        out *= 0.001
    return out


class PerceptionNode(object):
    """ASC60C real-robot perception with the old PerceptionNode surface API."""

    def __init__(self, cube_size_m=0.045):
        self.bridge = CvBridge()
        self.detector = CubeDetector(
            cube_size_m=cube_size_m,
            debug=_param_bool("perception_hsv_debug", False),
        )
        # Old code accesses self.perception.detector.bridge.
        self.detector.bridge = self.bridge
        self.pose_estimator = PoseEstimator(cube_size_m=cube_size_m)

        self.execution_environment = str(_param("execution_environment", "real"))
        self.rgb_topic = str(_param("rgb_topic", DEFAULT_RGB_TOPIC))
        self.depth_topic = str(_param("depth_topic", DEFAULT_DEPTH_TOPIC))
        self.camera_info_topic = str(
            _param("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC))
        self.pointcloud_topic = str(
            _param("pointcloud_topic", DEFAULT_POINTCLOUD_TOPIC))
        self.camera_frame = str(_param("camera_frame", DEFAULT_CAMERA_FRAME))
        self.langsam_mask_path = str(
            _param("langsam_mask_path", DEFAULT_MASK_PATH))

        self.use_registered_depth_mask = _param_bool(
            "use_registered_depth_mask", True)
        # Kept only as a compatibility flag.  It no longer means mask-indexing
        # the ASC60C PointCloud2.
        self.use_pointcloud_pose = _param_bool("use_pointcloud_pose", False)
        self.subscribe_pointcloud_diagnostic = _param_bool(
            "subscribe_pointcloud_diagnostic", False)
        self.strict_camera_frame = _param_bool("strict_camera_frame", True)
        self.max_mask_points = int(_param("perception_max_points", 12000))
        self.min_mask_points = int(_param("perception_min_points", 20))
        self.mask_erode_iterations = int(_param(
            "perception_mask_erode_iterations", 2))
        self.pointcloud_center_depth_offset = float(
            _param("pointcloud_center_depth_offset", 0.0))

        # Compatibility names used by the existing perception/pipeline code.
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

        self._warned_pointcloud_alias = False
        self._frame_mismatch_warned = False

        self.head_img_sub = rospy.Subscriber(
            self.rgb_topic, Image, self._head_image_cb, queue_size=2,
            buff_size=2 ** 24)
        self.head_depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self._head_depth_cb, queue_size=2,
            buff_size=2 ** 24)
        self.head_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._head_info_cb, queue_size=2)

        self.head_points_sub = None
        if self.subscribe_pointcloud_diagnostic:
            self.head_points_sub = rospy.Subscriber(
                self.pointcloud_topic, PointCloud2,
                self._head_points_cb, queue_size=1)

        rospy.loginfo(
            "[PerceptionReal] ASC60C registered-depth path enabled\n"
            "  rgb=%s\n  depth=%s\n  info=%s\n  camera_frame=%s",
            self.rgb_topic, self.depth_topic,
            self.camera_info_topic, self.camera_frame)
        rospy.loginfo(
            "[PerceptionReal] mask erosion iterations before depth projection: %d",
            self.mask_erode_iterations)
        if self.subscribe_pointcloud_diagnostic:
            rospy.loginfo(
                "[PerceptionReal] PointCloud2 is diagnostic only: %s",
                self.pointcloud_topic)

    def _head_image_cb(self, msg):
        self.head_image = msg

    def _head_depth_cb(self, msg):
        self.head_depth = msg

    def _head_points_cb(self, msg):
        self.head_points = msg

    def _head_info_cb(self, msg):
        self.head_camera_info = msg
        self.pose_estimator.set_camera_info(msg)
        actual = str(msg.header.frame_id or "")
        if (self.strict_camera_frame and actual and self.camera_frame
                and actual != self.camera_frame and not self._frame_mismatch_warned):
            self._frame_mismatch_warned = True
            rospy.logerr(
                "[PerceptionReal] CameraInfo frame mismatch: configured=%s actual=%s. "
                "Use the actual calibrated frame before autonomous execution.",
                self.camera_frame, actual)

    def wait_for_registered_rgbd(self, timeout_s=8.0):
        deadline = rospy.Time.now().to_sec() + float(timeout_s)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            if self.head_image is not None and self.head_depth is not None \
                    and self.head_camera_info is not None:
                return True
            rate.sleep()
        rospy.logwarn(
            "[PerceptionReal] Timed out waiting for RGB + registered depth + CameraInfo")
        return False

    def _resize_mask_nearest(self, mask, target_height, target_width):
        if mask.shape == (target_height, target_width):
            return mask
        y_idx = np.floor(
            np.arange(target_height) * mask.shape[0] / float(target_height)).astype(int)
        x_idx = np.floor(
            np.arange(target_width) * mask.shape[1] / float(target_width)).astype(int)
        y_idx = np.clip(y_idx, 0, mask.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, mask.shape[1] - 1)
        return mask[y_idx[:, None], x_idx[None, :]]

    def _erode_mask_for_depth_projection(self, mask):
        iterations = max(0, int(self.mask_erode_iterations))
        if iterations <= 0:
            return mask.astype(bool), 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(
            mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
        if int(np.count_nonzero(eroded)) < self.min_mask_points:
            rospy.logwarn(
                "[PerceptionReal] Eroded mask too small (%d px); "
                "falling back to raw mask (%d px)",
                int(np.count_nonzero(eroded)),
                int(np.count_nonzero(mask)))
            return mask.astype(bool), 0
        return eroded, iterations

    def _camera_matrix(self):
        info = self.head_camera_info
        if info is None or len(info.K) < 9 or float(info.K[0]) <= 0.0:
            return None
        K = info.K
        return np.asarray([
            [K[0], K[1], K[2]],
            [K[3], K[4], K[5]],
            [K[6], K[7], K[8]],
        ], dtype=np.float64)

    def get_camera_intrinsics_tuple(self):
        K = self._camera_matrix()
        if K is None:
            return None
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    def _source_frame(self):
        info_frame = ""
        depth_frame = ""
        if self.head_camera_info is not None:
            info_frame = str(self.head_camera_info.header.frame_id or "")
        if self.head_depth is not None:
            depth_frame = str(self.head_depth.header.frame_id or "")
        source = info_frame or depth_frame or self.camera_frame
        if self.strict_camera_frame and source != self.camera_frame:
            rospy.logerr_throttle(
                5.0,
                "[PerceptionReal] Refusing mismatched source frame: %s != configured %s",
                source, self.camera_frame)
            return None
        return source

    def _load_registered_depth(self):
        if self.head_depth is None:
            return None
        try:
            depth = self.bridge.imgmsg_to_cv2(
                self.head_depth, desired_encoding="passthrough")
            return np.asarray(depth)
        except Exception as exc:
            rospy.logwarn("[PerceptionReal] Depth conversion failed: %s", exc)
            return None

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

    def _mask_metadata(self, mask):
        ys, xs = np.where(mask)
        detection = self._detection_from_mask(mask)
        bbox = detection.get("bbox_2d") if detection else None
        center = detection.get("center_2d") if detection else None
        pixels = None
        max_mask_pixels = int(_param("mask_pixels_2d_max", 6000))
        if len(xs) and max_mask_pixels > 0:
            stride = max(1, int(math.ceil(float(len(xs)) / max_mask_pixels)))
            pixels = np.column_stack([xs[::stride], ys[::stride]]).astype(float).tolist()
        return detection, bbox, center, pixels

    def estimate_pose_with_depth_mask(self, mask_path=None):
        """Back-project a LangSAM mask with ASC60C registered depth + CameraInfo."""
        mask_path = str(mask_path or self.langsam_mask_path)
        if self.head_depth is None or self.head_camera_info is None:
            rospy.logwarn_throttle(
                3.0, "[PerceptionReal] registered depth / CameraInfo not ready")
            return None
        if not os.path.exists(mask_path):
            rospy.logwarn("[PerceptionReal] LangSAM mask file not found: %s", mask_path)
            return None

        source_frame = self._source_frame()
        if not source_frame:
            return None
        K = self._camera_matrix()
        depth = self._load_registered_depth()
        if K is None or depth is None:
            return None

        try:
            mask = np.load(mask_path).astype(bool)
        except Exception as exc:
            rospy.logwarn("[PerceptionReal] Failed loading mask %s: %s", mask_path, exc)
            return None

        target_h, target_w = depth.shape[:2]
        if mask.shape != (target_h, target_w):
            rospy.logwarn(
                "[PerceptionReal] Resizing mask %s -> registered depth %s",
                str(mask.shape), str((target_h, target_w)))
            mask = self._resize_mask_nearest(mask, target_h, target_w)

        raw_mask_pixels = int(np.count_nonzero(mask))
        mask, applied_erode_iterations = self._erode_mask_for_depth_projection(mask)
        eroded_mask_pixels = int(np.count_nonzero(mask))
        rospy.loginfo(
            "[PerceptionReal] mask erosion for depth projection: "
            "raw=%d eroded=%d iterations=%d",
            raw_mask_pixels, eroded_mask_pixels, applied_erode_iterations)

        if int(np.count_nonzero(mask)) < self.min_mask_points:
            rospy.logwarn(
                "[PerceptionReal] Mask too small: %d pixels",
                int(np.count_nonzero(mask)))
            return None

        points = pointcloud_from_depth_mask(
            depth, mask, K, max_points=self.max_mask_points)
        if points is None or len(points) < self.min_mask_points:
            rospy.logwarn(
                "[PerceptionReal] Too few valid registered-depth points: %s",
                0 if points is None else len(points))
            return None

        points = np.asarray(points, dtype=np.float64)
        surface_center = np.median(points, axis=0)
        center = surface_center.copy()
        center[2] += self.pointcloud_center_depth_offset
        spread = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)

        detection, bbox, mask_center, mask_pixels = self._mask_metadata(mask)
        clean_mask = (mask.astype(np.uint8) * 255)
        self.latest_clean_mask = clean_mask
        self.latest_mask = clean_mask
        self.latest_detection = detection
        if self.head_image is not None and self.latest_bgr is None:
            try:
                self.latest_bgr = self.bridge.imgmsg_to_cv2(
                    self.head_image, desired_encoding="bgr8")
            except Exception:
                pass

        pose = {
            "position": center.tolist(),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "confidence": 0.97,
            "method": "LangSAM+registered_depth_camera_info",
            "object_points": points.tolist(),
            "mask_bbox_2d": bbox,
            "mask_center_2d": mask_center,
            "mask_pixels_2d": mask_pixels,
            "cloud_size": [int(target_w), int(target_h)],
            "source_frame": source_frame,
            "depth_encoding": str(self.head_depth.encoding),
            "valid_depth_points": int(len(points)),
            "mask_erode_iterations": int(applied_erode_iterations),
            "raw_mask_pixels_2d": int(raw_mask_pixels),
            "eroded_mask_pixels_2d": int(eroded_mask_pixels),
            "object_size_m": spread.tolist(),
            "visible_spread_camera_m": spread.tolist(),
        }
        self.latest_pose = pose
        rospy.loginfo(
            "[PerceptionReal] mask points=%d center_cam=[%.4f %.4f %.4f] "
            "spread=[%.4f %.4f %.4f] frame=%s",
            len(points), center[0], center[1], center[2],
            spread[0], spread[1], spread[2], source_frame)
        return pose

    def estimate_pose_with_pointcloud_mask(self):
        """Compatibility alias; real ASC60C uses registered depth, not sparse cloud."""
        if not self._warned_pointcloud_alias:
            self._warned_pointcloud_alias = True
            rospy.loginfo(
                "[PerceptionReal] estimate_pose_with_pointcloud_mask() redirected "
                "to registered depth + CameraInfo")
        return self.estimate_pose_with_depth_mask()

    def camera_points_from_depth_patch(self, center_uv, radius_px=4):
        """Return camera-frame xyz samples around one registered RGB/depth pixel."""
        if center_uv is None:
            return []
        depth = self._load_registered_depth()
        K = self._camera_matrix()
        if depth is None or K is None:
            return []
        depth_m = _depth_to_meters(depth)
        h, w = depth_m.shape[:2]
        cx_px = int(round(float(center_uv[0])))
        cy_px = int(round(float(center_uv[1])))
        r = max(0, int(radius_px))
        x1, x2 = max(0, cx_px - r), min(w, cx_px + r + 1)
        y1, y2 = max(0, cy_px - r), min(h, cy_px + r + 1)
        if x2 <= x1 or y2 <= y1:
            return []
        yy, xx = np.mgrid[y1:y2, x1:x2]
        z = depth_m[y1:y2, x1:x2]
        valid = np.isfinite(z) & (z > 0.01) & (z < 10.0)
        if not np.any(valid):
            return []
        xs = xx[valid].astype(np.float64)
        ys = yy[valid].astype(np.float64)
        zs = z[valid].astype(np.float64)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        X = (xs - cx) * zs / fx
        Y = (ys - cy) * zs / fy
        return np.stack([X, Y, zs], axis=1).tolist()

    def detect_with_head_camera(self):
        """Compatibility HSV detector on ASC60C RGB."""
        if self.head_image is None:
            rospy.logwarn_throttle(5.0, "[PerceptionReal] No RGB image yet")
            return None
        detections, bgr, mask = self.detector.detect(self.head_image)
        self.latest_bgr = bgr
        self.latest_mask = mask
        if not detections:
            self.latest_clean_mask = None
            return None
        self.latest_detection = detections[0]
        self.latest_clean_mask = mask
        return self.latest_detection

    def _correct_depth_from_sensor(self, pose):
        """Depth-correct the old HSV/PnP fallback with registered depth in meters."""
        if pose is None or self.head_depth is None or self.latest_detection is None:
            return pose
        depth = self._load_registered_depth()
        if depth is None:
            return pose
        depth_m = _depth_to_meters(depth)
        mask = self.latest_clean_mask
        if mask is not None and mask.shape == depth_m.shape[:2]:
            values = depth_m[mask > 0]
        else:
            center = self.latest_detection.get("center_2d")
            if center is None:
                return pose
            u, v = int(round(center[0])), int(round(center[1]))
            r = 5
            h, w = depth_m.shape[:2]
            values = depth_m[max(0, v-r):min(h, v+r+1),
                             max(0, u-r):min(w, u+r+1)].reshape(-1)
        valid = values[np.isfinite(values)]
        valid = valid[(valid > 0.01) & (valid < 10.0)]
        if len(valid) < 3:
            return pose
        real_depth = float(np.median(valid))
        pnp = list(pose.get("position", [0.0, 0.0, 0.0]))
        if abs(float(pnp[2])) < 1e-6:
            return pose
        scale = real_depth / float(pnp[2])
        pose["position"] = [float(pnp[0]) * scale, float(pnp[1]) * scale, real_depth]
        pose["method"] = str(pose.get("method", "PnP")) + "+registered_depth"
        pose["source_frame"] = self._source_frame()
        return pose

    def estimate_pose_with_head(self):
        # Real robot: external segmentation + registered depth is the primary path.
        if self.use_registered_depth_mask or self.use_pointcloud_pose:
            pose = self.estimate_pose_with_depth_mask()
            if pose is not None:
                return pose
            rospy.logwarn(
                "[PerceptionReal] Mask+registered-depth pose failed; trying HSV/PnP fallback")
        if self.latest_detection is None and not self.detect_with_head_camera():
            return None
        pose = self.pose_estimator.estimate_pose(self.latest_detection)
        pose = self._correct_depth_from_sensor(pose)
        if pose is not None and not pose.get("source_frame"):
            pose["source_frame"] = self._source_frame()
        self.latest_pose = pose
        return pose

    def estimate_pose(self):
        """Compatibility convenience method used by some pipeline revisions."""
        return self.estimate_pose_with_head()

    def get_object_pose(self):
        return self.estimate_pose_with_head()

    def get_debug_data(self):
        return {
            "bgr_image": self.latest_bgr,
            "hsv_mask": self.latest_clean_mask,
            "detection": self.latest_detection,
            "pose": self.latest_pose,
            "camera_info": self.head_camera_info,
        }

    def get_detected_features(self):
        if self.latest_detection is None:
            return None
        det = self.latest_detection
        return {
            "shape": "box",
            "dimensions_m": [self.detector.cube_size] * 3,
            "aspect_ratio": [1.0, 1.0, 1.0],
            "color_rgb": [0.0, 1.0, 0.0],
            "bounding_box_area": float(det.get("area", 0.0)),
            "extent": float(det.get("extent", 0.0)),
            "confidence": float(det.get("confidence", 0.0)),
        }


if __name__ == "__main__":
    rospy.init_node("mt3_perception_real_test")
    node = PerceptionNode()
    rospy.loginfo("Real MT3 perception waiting for ASC60C and LangSAM mask...")
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        pose = node.get_object_pose()
        if pose:
            p = pose["position"]
            rospy.loginfo(
                "Object camera pose: [%.4f %.4f %.4f] method=%s frame=%s",
                p[0], p[1], p[2], pose.get("method"), pose.get("source_frame"))
        rate.sleep()
