#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-object perception for anchored pick-and-place.

This module intentionally matches the current project setup: one frozen
head-camera RGB-D observation, external LangSAM masks saved as .npy files, and
PointCloud2 for 3D positions.  It detects:

  - target object mask, e.g. green cube
  - anchor object mask, e.g. blue tray

No obstacle avoidance or closed-loop re-detection is done here.
"""

import ast
import os
import time

import numpy as np
import rospy

from mt3_alignment import TrajectoryAligner, pose_compose, quat_rotate
from mt3_perception import PerceptionNode


DEFAULT_TARGET_MASK = "/mnt/hgfs2/tmp_vision/current_mask.npy"
DEFAULT_ANCHOR_MASK = "/mnt/hgfs2/tmp_vision/current_anchor_mask.npy"


def _param_bool(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _param_float_list(name, default=None):
    value = rospy.get_param(name, default if default is not None else [])
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed]
        except Exception:
            pass
        text = text.strip("[]")
        if not text:
            return []
        return [float(v.strip()) for v in text.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return []


def _estimate_size_from_points(points):
    if points is None or len(points) < 5:
        return None
    arr = np.asarray(points, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) < 5:
        return None
    spread = np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)
    return [float(max(0.0, v)) for v in spread.tolist()]


class DualMaskAnchorPerception(object):
    """Detect target and anchor from two external masks plus PointCloud2."""

    def __init__(self, target_mask_path=None, anchor_mask_path=None,
                 target_size=None, anchor_size=None):
        self.target_mask_path = target_mask_path or rospy.get_param(
            "~target_mask_path", DEFAULT_TARGET_MASK)
        self.anchor_mask_path = anchor_mask_path or rospy.get_param(
            "~anchor_mask_path", DEFAULT_ANCHOR_MASK)
        self.target_size = target_size
        self.anchor_size = anchor_size
        self.use_geometry_center_correction = rospy.get_param(
            "~use_geometry_center_correction", True)
        self.perception = PerceptionNode()
        self.aligner = TrajectoryAligner()

    def wait_for_pointcloud(self, timeout_s=8.0):
        deadline = time.time() + float(timeout_s)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.perception.head_points is not None:
                return True
            rate.sleep()
        rospy.logwarn("[AnchorPerception] no PointCloud2 received within %.1fs",
                      timeout_s)
        return False

    def _detect_one(self, label, mask_path):
        if not os.path.exists(mask_path):
            rospy.logwarn("[AnchorPerception] %s mask missing: %s", label, mask_path)
            return None
        old_path = self.perception.langsam_mask_path
        try:
            self.perception.langsam_mask_path = mask_path
            pose_source = self.perception.estimate_pose_with_pointcloud_mask()
        finally:
            self.perception.langsam_mask_path = old_path
        if pose_source is None:
            rospy.logwarn("[AnchorPerception] %s detection failed", label)
            return None

        pose_source["mask_path"] = mask_path
        pose_source["estimated_object_size"] = _estimate_size_from_points(
            pose_source.get("object_points"))
        pose_base = self.aligner.transform_camera_to_base(pose_source)
        if pose_base is None:
            rospy.logwarn("[AnchorPerception] %s TF to base failed", label)
            return None
        if pose_source.get("estimated_object_size") is not None:
            pose_base["estimated_object_size"] = pose_source.get(
                "estimated_object_size")
        pose_base = self._correct_geometry_center(label, pose_source, pose_base)
        scene_arrays = self._current_scene_arrays()

        return {
            "label": label,
            "mask_path": mask_path,
            "pose_source": pose_source,
            "pose_base": pose_base,
            "position_base": pose_base.get("position"),
            "orientation_base": pose_base.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            "estimated_size": pose_base.get("estimated_object_size"),
            "rgb": scene_arrays.get("rgb"),
            "depth": scene_arrays.get("depth"),
            "intrinsics": scene_arrays.get("intrinsics"),
            "method": "dual_LangSAM_masks+PointCloud2+TF",
        }

    def _current_scene_arrays(self):
        rgb = None
        depth = None
        intrinsics = None

        try:
            bgr = self.perception.latest_bgr
            if bgr is None and self.perception.head_image is not None:
                bgr = self.perception.detector.bridge.imgmsg_to_cv2(
                    self.perception.head_image, desired_encoding="bgr8")
                self.perception.latest_bgr = bgr
            if bgr is not None:
                rgb = np.asarray(bgr)[:, :, ::-1].copy()
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] RGB snapshot unavailable: %s", exc)

        try:
            if self.perception.head_depth is not None:
                depth = self.perception.detector.bridge.imgmsg_to_cv2(
                    self.perception.head_depth, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] depth snapshot unavailable: %s", exc)

        try:
            info = self.perception.head_camera_info
            if info is not None:
                K = info.K
                intrinsics = np.asarray([
                    [K[0], K[1], K[2]],
                    [K[3], K[4], K[5]],
                    [K[6], K[7], K[8]],
                ], dtype=np.float64)
            else:
                K = self.perception.pose_estimator.camera_matrix
                if K is not None:
                    intrinsics = np.asarray(K, dtype=np.float64)
        except Exception as exc:
            rospy.logwarn(
                "[AnchorPerception] camera intrinsics unavailable: %s", exc)

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
        }

    def _points_to_base(self, pose_source):
        points = pose_source.get("object_points") or []
        source_frame = pose_source.get("source_frame")
        if not points or not source_frame or self.aligner._tf_buffer is None:
            return None
        try:
            tf = self.aligner._tf_buffer.lookup_transform(
                "base", source_frame, rospy.Time(0), rospy.Duration(2.0))
            trans = tf.transform.translation
            rot = tf.transform.rotation
            tf_pos = [trans.x, trans.y, trans.z]
            tf_ori = [rot.x, rot.y, rot.z, rot.w]
            base_points = []
            for p in points:
                if len(p) < 3:
                    continue
                pos_base, _ = pose_compose(
                    tf_pos, tf_ori,
                    [float(p[0]), float(p[1]), float(p[2])],
                    [0.0, 0.0, 0.0, 1.0])
                if all(np.isfinite(pos_base)):
                    base_points.append(pos_base)
            if len(base_points) < 10:
                return None
            return np.asarray(base_points, dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] geometry correction TF failed: %s",
                          exc)
            return None

    def _camera_points_to_base(self, points, source_frame):
        if not points or not source_frame or self.aligner._tf_buffer is None:
            return None
        try:
            tf = self.aligner._tf_buffer.lookup_transform(
                "base", source_frame, rospy.Time(0), rospy.Duration(2.0))
            trans = tf.transform.translation
            rot = tf.transform.rotation
            tf_pos = [trans.x, trans.y, trans.z]
            tf_ori = [rot.x, rot.y, rot.z, rot.w]
            base_points = []
            for p in points:
                if len(p) < 3:
                    continue
                pos_base, _ = pose_compose(
                    tf_pos, tf_ori,
                    [float(p[0]), float(p[1]), float(p[2])],
                    [0.0, 0.0, 0.0, 1.0])
                if all(np.isfinite(pos_base)):
                    base_points.append(pos_base)
            if not base_points:
                return None
            return np.asarray(base_points, dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] camera point TF failed: %s", exc)
            return None

    def _anchor_hole_center_from_mask_center(self, pose_source):
        center = pose_source.get("mask_center_2d")
        cloud_size = pose_source.get("cloud_size") or []
        source_frame = pose_source.get("source_frame")
        if center is None or len(cloud_size) < 2:
            return None
        try:
            cloud_width, cloud_height = int(cloud_size[0]), int(cloud_size[1])
            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))
            radius = int(rospy.get_param("~anchor_center_depth_patch_px", 4))
            indices = []
            for y in range(max(0, cy - radius), min(cloud_height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(cloud_width, cx + radius + 1)):
                    indices.append(y * cloud_width + x)
            points = self.perception._read_xyz_at_indices(
                self.perception.head_points, np.asarray(indices, dtype=np.int64))
            base_points = self._camera_points_to_base(points, source_frame)
            if base_points is None or len(base_points) < 3:
                return None
            center_base = np.median(base_points, axis=0)
            return [float(v) for v in center_base.tolist()]
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] anchor mask-center depth failed: %s",
                          exc)
            return None

    def _anchor_ring_center_from_points(self, points_base, box_center):
        """Estimate circular socket center from masked blue ring points.

        The blue socket is an annulus.  Fitting a circle in base XY is usually
        more stable than sampling depth through the hole, because the hole
        patch may land on the table or an occluded pixel.
        """
        if points_base is None or len(points_base) < 20:
            return None
        pts = np.asarray(points_base[:, :2], dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 20:
            return None

        low = np.percentile(pts, 2, axis=0)
        high = np.percentile(pts, 98, axis=0)
        keep = np.all((pts >= low) & (pts <= high), axis=1)
        pts = pts[keep]
        if len(pts) < 20:
            return None

        x = pts[:, 0]
        y = pts[:, 1]
        a = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        try:
            sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        except Exception:
            return None
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        radius = float(np.sqrt(radius_sq))

        anchor_size = self.anchor_size or []
        expected_outer = None
        if len(anchor_size) >= 2:
            expected_outer = 0.5 * max(float(anchor_size[0]), float(anchor_size[1]))
        if expected_outer is not None:
            if radius < expected_outer * 0.25 or radius > expected_outer * 1.35:
                return None
        elif radius < 0.015 or radius > 0.090:
            return None

        center = np.asarray([cx, cy], dtype=np.float64)
        box_xy = np.asarray(box_center[:2], dtype=np.float64)
        max_delta = float(rospy.get_param(
            "~anchor_circle_fit_max_delta_m",
            max(0.030, (expected_outer or 0.050) * 0.65)))
        if np.linalg.norm(center - box_xy) > max_delta:
            return None

        radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        residual = float(np.median(np.abs(radial - radius)))
        return {
            "center_xy": [cx, cy],
            "radius": radius,
            "residual": residual,
            "num_points": int(len(pts)),
        }

    def _anchor_top_band_center_from_points(self, points_base, box_center,
                                            z_plane):
        """Fit the socket axis from blue points close to the socket top plane."""
        if points_base is None or len(points_base) < 20 or z_plane is None:
            return None
        pts3 = np.asarray(points_base, dtype=np.float64)
        pts3 = pts3[np.all(np.isfinite(pts3), axis=1)]
        if len(pts3) < 20:
            return None

        band = float(rospy.get_param("~anchor_top_band_z_m", 0.012))
        keep = np.abs(pts3[:, 2] - float(z_plane)) <= band
        pts = pts3[keep, :2]
        min_pts = int(rospy.get_param("~anchor_top_band_min_points", 20))
        if len(pts) < min_pts:
            return None

        low = np.percentile(pts, 2, axis=0)
        high = np.percentile(pts, 98, axis=0)
        pts = pts[np.all((pts >= low) & (pts <= high), axis=1)]
        if len(pts) < min_pts:
            return None

        anchor_size = self.anchor_size or []
        expected_outer = None
        if len(anchor_size) >= 2:
            expected_outer = 0.5 * max(float(anchor_size[0]),
                                       float(anchor_size[1]))
        if expected_outer is None:
            return None

        fixed = self._fit_fixed_radius_center(
            pts, expected_outer, np.asarray(box_center[:2], dtype=np.float64))
        if fixed is None:
            return None
        center, residual = fixed
        max_residual = float(rospy.get_param(
            "~anchor_top_band_circle_max_residual_m",
            max(0.004, expected_outer * 0.15)))
        if residual > max_residual:
            return None
        return {
            "center_xy": [float(center[0]), float(center[1])],
            "radius": float(expected_outer),
            "residual": float(residual),
            "num_points": int(len(pts)),
            "z_band_m": float(band),
            "fit_method": "base_pointcloud_top_band_circle_fit",
        }

    def _camera_intrinsics(self):
        info = self.perception.head_camera_info
        if info is not None and len(info.K) >= 9 and info.K[0] > 0.0:
            return float(info.K[0]), float(info.K[4]), float(info.K[2]), float(info.K[5])
        # Gazebo Sawyer head camera default used by mt3_perception.py.
        return 407.391526, 407.391526, 640.5, 400.5

    def _source_frame_tf(self, source_frame):
        if not source_frame or self.aligner._tf_buffer is None:
            return None
        try:
            tf = self.aligner._tf_buffer.lookup_transform(
                "base", source_frame, rospy.Time(0), rospy.Duration(2.0))
            trans = tf.transform.translation
            rot = tf.transform.rotation
            return (
                [float(trans.x), float(trans.y), float(trans.z)],
                [float(rot.x), float(rot.y), float(rot.z), float(rot.w)],
            )
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] source-frame TF failed: %s", exc)
            return None

    def _anchor_center_from_mask_ray_plane(self, pose_source, z_plane):
        """Project the 2D socket mask center onto a known base-z plane.

        Pointcloud circle fitting is biased when the camera sees only a partial
        annulus.  The 2D mask center remains stable for the visible blue ring,
        so project that image center through the camera model to the socket
        plane and use the resulting XY as the insertion axis.
        """
        center = pose_source.get("mask_center_2d")
        source_frame = pose_source.get("source_frame")
        if center is None or source_frame is None or z_plane is None:
            return None
        tf_pose = self._source_frame_tf(source_frame)
        if tf_pose is None:
            return None
        try:
            u = float(center[0])
            v = float(center[1])
            fx, fy, cx, cy = self._camera_intrinsics()
            ray_source = [(u - cx) / fx, (v - cy) / fy, 1.0]
            origin_base, source_q = tf_pose
            ray_base = quat_rotate(source_q, ray_source)
            if abs(ray_base[2]) < 1e-6:
                return None
            scale = (float(z_plane) - origin_base[2]) / ray_base[2]
            if scale <= 0.0:
                return None
            projected = [
                origin_base[0] + scale * ray_base[0],
                origin_base[1] + scale * ray_base[1],
                float(z_plane),
            ]
            if not all(np.isfinite(projected)):
                return None
            return projected
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] mask ray-plane center failed: %s",
                          exc)
            return None

    def _project_image_pixel_to_plane(self, center, source_frame, z_plane):
        if center is None or source_frame is None or z_plane is None:
            return None
        tf_pose = self._source_frame_tf(source_frame)
        if tf_pose is None:
            return None
        try:
            u = float(center[0])
            v = float(center[1])
            fx, fy, cx, cy = self._camera_intrinsics()
            ray_source = [(u - cx) / fx, (v - cy) / fy, 1.0]
            origin_base, source_q = tf_pose
            ray_base = quat_rotate(source_q, ray_source)
            if abs(ray_base[2]) < 1e-6:
                return None
            scale = (float(z_plane) - origin_base[2]) / ray_base[2]
            if scale <= 0.0:
                return None
            projected = [
                origin_base[0] + scale * ray_base[0],
                origin_base[1] + scale * ray_base[1],
                float(z_plane),
            ]
            if not all(np.isfinite(projected)):
                return None
            return projected
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] image ray-plane projection failed: %s",
                          exc)
            return None

    def _latest_bgr_image(self):
        try:
            bgr = self.perception.latest_bgr
            if bgr is None and self.perception.head_image is not None:
                bgr = self.perception.detector.bridge.imgmsg_to_cv2(
                    self.perception.head_image, desired_encoding="bgr8")
                self.perception.latest_bgr = bgr
            if bgr is None:
                return None
            return np.asarray(bgr)
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] BGR image unavailable: %s", exc)
            return None

    def _anchor_dark_hole_center_from_rgb(self, pose_source, z_plane):
        """Find the black visual hole marker inside the blue socket ROI."""
        mask = self._load_pose_mask(pose_source)
        bgr = self._latest_bgr_image()
        source_frame = pose_source.get("source_frame")
        if mask is None or bgr is None or source_frame is None:
            return None
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return None

        img_h, img_w = bgr.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        sx = float(img_w) / float(mask_w)
        sy = float(img_h) / float(mask_h)
        margin = int(rospy.get_param("~anchor_dark_hole_roi_margin_px", 6))
        x1 = max(0, int(np.floor(xs.min() * sx)) - margin)
        x2 = min(img_w, int(np.ceil((xs.max() + 1) * sx)) + margin)
        y1 = max(0, int(np.floor(ys.min() * sy)) - margin)
        y2 = min(img_h, int(np.ceil((ys.max() + 1) * sy)) + margin)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = bgr[y1:y2, x1:x2, :].astype(np.float32)
        blue = roi[:, :, 0]
        green = roi[:, :, 1]
        red = roi[:, :, 2]
        max_channel = np.maximum(np.maximum(blue, green), red)
        mean_channel = (blue + green + red) / 3.0
        dark_max = float(rospy.get_param("~anchor_dark_hole_max_channel", 65.0))
        dark_mean = float(rospy.get_param("~anchor_dark_hole_mean_channel", 45.0))
        dark = (max_channel <= dark_max) & (mean_channel <= dark_mean)

        # Restrict the search toward the inside of the blue socket bbox so a
        # dark robot/table edge just outside the ROI is not selected.
        roi_h, roi_w = dark.shape
        border = int(rospy.get_param("~anchor_dark_hole_ignore_border_px", 2))
        if border > 0 and roi_h > 2 * border and roi_w > 2 * border:
            dark[:border, :] = False
            dark[-border:, :] = False
            dark[:, :border] = False
            dark[:, -border:] = False

        min_area = int(rospy.get_param("~anchor_dark_hole_min_pixels", 12))
        max_area = int(rospy.get_param(
            "~anchor_dark_hole_max_pixels",
            max(30, int(0.65 * roi_h * roi_w))))
        visited = np.zeros(dark.shape, dtype=bool)
        yy, xx = np.where(dark)
        if len(xx) < min_area:
            return None

        roi_center = np.asarray([0.5 * (roi_w - 1), 0.5 * (roi_h - 1)],
                                dtype=np.float64)
        best = None
        best_score = None
        for sy0, sx0 in zip(yy.tolist(), xx.tolist()):
            if visited[sy0, sx0] or not dark[sy0, sx0]:
                continue
            stack = [(sy0, sx0)]
            visited[sy0, sx0] = True
            comp = []
            while stack:
                cy0, cx0 = stack.pop()
                comp.append((cy0, cx0))
                for ny, nx in ((cy0 - 1, cx0), (cy0 + 1, cx0),
                               (cy0, cx0 - 1), (cy0, cx0 + 1)):
                    if (0 <= ny < roi_h and 0 <= nx < roi_w and
                            dark[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(comp)
            if area < min_area or area > max_area:
                continue
            arr = np.asarray(comp, dtype=np.float64)
            cyx = arr.mean(axis=0)
            center_xy = np.asarray([cyx[1], cyx[0]], dtype=np.float64)
            distance = float(np.linalg.norm(center_xy - roi_center))
            score = distance - 0.03 * float(area)
            if best is None or score < best_score:
                best = {
                    "area": int(area),
                    "center_pixel": [
                        float(x1 + center_xy[0]),
                        float(y1 + center_xy[1]),
                    ],
                    "roi_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "roi_center_pixel": [
                        float(x1 + roi_center[0]),
                        float(y1 + roi_center[1]),
                    ],
                    "component_center_roi": [
                        float(center_xy[0]), float(center_xy[1])],
                    "distance_to_roi_center_px": distance,
                }
                best_score = score

        if best is None:
            return None
        projected = self._project_image_pixel_to_plane(
            best["center_pixel"], source_frame, z_plane)
        if projected is None:
            return None
        best["center_xyz"] = [float(v) for v in projected]
        best["method"] = "dark_hole_rgb_ray_plane"
        return best

    def _load_pose_mask(self, pose_source):
        mask_path = pose_source.get("mask_path")
        cloud_size = pose_source.get("cloud_size") or []
        if not mask_path or not os.path.exists(mask_path) or len(cloud_size) < 2:
            return None
        try:
            mask = np.load(mask_path).astype(bool)
            cloud_width, cloud_height = int(cloud_size[0]), int(cloud_size[1])
            if mask.shape != (cloud_height, cloud_width):
                mask = self.perception._resize_mask_nearest(
                    mask, cloud_height, cloud_width)
            return mask.astype(bool)
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] load anchor mask failed: %s", exc)
            return None

    def _largest_component(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        visited = np.zeros(mask.shape, dtype=bool)
        best = []
        height, width = mask.shape
        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < height and 0 <= nx < width and
                            mask[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > len(best):
                best = component
        if not best:
            return None
        out = np.zeros(mask.shape, dtype=bool)
        for y, x in best:
            out[y, x] = True
        return out

    def _inner_boundary_pixels_from_mask(self, pose_source):
        """Return blue mask pixels adjacent to the socket hole."""
        mask = self._load_pose_mask(pose_source)
        if mask is None:
            return None
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return None

        margin = int(rospy.get_param("~anchor_inner_boundary_roi_margin_px", 8))
        y1 = max(0, int(ys.min()) - margin)
        y2 = min(mask.shape[0], int(ys.max()) + margin + 1)
        x1 = max(0, int(xs.min()) - margin)
        x2 = min(mask.shape[1], int(xs.max()) + margin + 1)
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        background = ~roi
        outside = np.zeros(roi.shape, dtype=bool)
        stack = []
        height, width = roi.shape
        for x in range(width):
            if background[0, x]:
                stack.append((0, x))
            if background[height - 1, x]:
                stack.append((height - 1, x))
        for y in range(height):
            if background[y, 0]:
                stack.append((y, 0))
            if background[y, width - 1]:
                stack.append((y, width - 1))
        while stack:
            y, x = stack.pop()
            if outside[y, x] or not background[y, x]:
                continue
            outside[y, x] = True
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    if background[ny, nx] and not outside[ny, nx]:
                        stack.append((ny, nx))

        holes = background & (~outside)
        hole = self._largest_component(holes)
        if hole is None:
            return None
        min_hole_px = int(rospy.get_param("~anchor_inner_hole_min_pixels", 20))
        if int(np.count_nonzero(hole)) < min_hole_px:
            return None

        neighbor_hole = np.zeros(roi.shape, dtype=bool)
        neighbor_hole[1:, :] |= hole[:-1, :]
        neighbor_hole[:-1, :] |= hole[1:, :]
        neighbor_hole[:, 1:] |= hole[:, :-1]
        neighbor_hole[:, :-1] |= hole[:, 1:]
        inner_boundary = roi & neighbor_hole
        by, bx = np.where(inner_boundary)
        if len(bx) < 12:
            return None
        return np.column_stack([bx + x1, by + y1]).astype(float).tolist()

    def _outer_boundary_pixels_from_mask(self, pose_source, top_only=False):
        """Return blue mask silhouette boundary pixels.

        LangSAM often segments the blue insert socket as a solid visible
        cylinder without the hole.  In that case the inner-hole extraction
        cannot work.  The upper silhouette arc is still tied to the top outer
        rim, so fit that arc on the known socket plane with the known radius.
        """
        mask = self._load_pose_mask(pose_source)
        if mask is None:
            return None
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return None

        neighbor_bg = np.zeros(mask.shape, dtype=bool)
        neighbor_bg[1:, :] |= ~mask[:-1, :]
        neighbor_bg[:-1, :] |= ~mask[1:, :]
        neighbor_bg[:, 1:] |= ~mask[:, :-1]
        neighbor_bg[:, :-1] |= ~mask[:, 1:]
        boundary = mask & neighbor_bg
        by, bx = np.where(boundary)
        if len(bx) < 12:
            return None

        if top_only:
            y_min = float(ys.min())
            y_max = float(ys.max())
            top_fraction = float(rospy.get_param(
                "~anchor_outer_silhouette_top_fraction", 0.58))
            y_cut = y_min + max(0.10, min(0.90, top_fraction)) * (
                y_max - y_min + 1.0)
            keep = by.astype(float) <= y_cut
            bx = bx[keep]
            by = by[keep]
            if len(bx) < 12:
                return None
        return np.column_stack([bx, by]).astype(float).tolist()

    def _fit_fixed_radius_center(self, pts, radius, initial_center):
        if pts is None or len(pts) < 12 or radius is None:
            return None
        pts = np.asarray(pts, dtype=np.float64)
        center = np.asarray(initial_center, dtype=np.float64)
        if center.shape[0] < 2 or not np.all(np.isfinite(center[:2])):
            center = np.median(pts[:, :2], axis=0)
        center = center[:2].astype(np.float64)
        radius = float(radius)
        for _ in range(12):
            diff = center[None, :] - pts[:, :2]
            dist = np.linalg.norm(diff, axis=1)
            valid = dist > 1e-6
            if np.count_nonzero(valid) < 8:
                return None
            residual = dist[valid] - radius
            jac = diff[valid] / dist[valid][:, None]
            try:
                step, _, _, _ = np.linalg.lstsq(jac, -residual, rcond=None)
            except Exception:
                return None
            if not np.all(np.isfinite(step)):
                return None
            max_step = float(rospy.get_param(
                "~anchor_fixed_radius_fit_max_step_m", 0.010))
            norm = float(np.linalg.norm(step))
            if norm > max_step:
                step = step * (max_step / max(norm, 1e-9))
            center = center + step
            if np.linalg.norm(step) < 1e-5:
                break
        radial = np.linalg.norm(pts[:, :2] - center[None, :], axis=1)
        residual = float(np.median(np.abs(radial - radius)))
        return center, residual

    def _fit_mask_plane_circle(self, pixels, pose_source, z_plane,
                               expected_radius=None, method="mask_plane_circle_fit"):
        if len(pixels) < 12:
            return None
        source_frame = pose_source.get("source_frame")
        if source_frame is None or z_plane is None:
            return None
        tf_pose = self._source_frame_tf(source_frame)
        if tf_pose is None:
            return None
        try:
            fx, fy, cx, cy = self._camera_intrinsics()
            origin_base, source_q = tf_pose
            max_pixels = int(rospy.get_param(
                "~anchor_mask_plane_fit_max_pixels", 4000))
            stride = max(1, int(np.ceil(float(len(pixels)) / max_pixels)))
            projected = []
            for px in pixels[::stride]:
                if len(px) < 2:
                    continue
                u = float(px[0])
                v = float(px[1])
                ray_source = [(u - cx) / fx, (v - cy) / fy, 1.0]
                ray_base = quat_rotate(source_q, ray_source)
                if abs(ray_base[2]) < 1e-6:
                    continue
                scale = (float(z_plane) - origin_base[2]) / ray_base[2]
                if scale <= 0.0:
                    continue
                p = [
                    origin_base[0] + scale * ray_base[0],
                    origin_base[1] + scale * ray_base[1],
                ]
                if all(np.isfinite(p)):
                    projected.append(p)
            if len(projected) < 12:
                return None

            pts = np.asarray(projected, dtype=np.float64)
            low = np.percentile(pts, 1, axis=0)
            high = np.percentile(pts, 99, axis=0)
            keep = np.all((pts >= low) & (pts <= high), axis=1)
            pts = pts[keep]
            if len(pts) < 12:
                return None

            bbox_center = (np.percentile(pts, 2, axis=0) +
                           np.percentile(pts, 98, axis=0)) * 0.5

            x = pts[:, 0]
            y = pts[:, 1]
            a = np.column_stack([x, y, np.ones_like(x)])
            b = -(x * x + y * y)
            sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
            fit_x = -0.5 * float(sol[0])
            fit_y = -0.5 * float(sol[1])
            radius_sq = max(0.0, fit_x * fit_x + fit_y * fit_y - float(sol[2]))
            radius = float(np.sqrt(radius_sq))
            radial = np.sqrt((x - fit_x) ** 2 + (y - fit_y) ** 2)
            residual = float(np.median(np.abs(radial - radius)))
            fixed_radius_used = False

            if expected_radius is not None:
                tol = float(rospy.get_param(
                    "~anchor_mask_plane_radius_tolerance_m", 0.012))
                fixed = self._fit_fixed_radius_center(
                    pts, expected_radius, bbox_center)
                if fixed is not None:
                    fixed_center, fixed_residual = fixed
                    if (abs(radius - expected_radius) > tol or
                            fixed_residual <= residual * 1.25 or
                            _param_bool(
                                "~anchor_mask_plane_force_fixed_radius",
                                True)):
                        fit_x = float(fixed_center[0])
                        fit_y = float(fixed_center[1])
                        radius = float(expected_radius)
                        residual = float(fixed_residual)
                        fixed_radius_used = True
                if abs(radius - expected_radius) > tol:
                    return None
                default_residual = max(0.0035, expected_radius * 0.18)
            else:
                if radius < 0.010 or radius > 0.090:
                    return None
                default_residual = 0.008
            max_residual = float(rospy.get_param(
                "~anchor_mask_plane_circle_max_residual_m", default_residual))
            if residual > max_residual:
                return None

            return {
                "center_xy": [fit_x, fit_y],
                "z": float(z_plane),
                "radius": radius,
                "expected_radius": expected_radius,
                "residual": residual,
                "num_points": int(len(pts)),
                "bbox_center_xy": [float(v) for v in bbox_center.tolist()],
                "fit_method": method,
                "fixed_radius": bool(fixed_radius_used),
            }
        except Exception as exc:
            rospy.logwarn("[AnchorPerception] %s failed: %s", method, exc)
            return None

    def _anchor_ring_center_from_mask_plane(self, pose_source, z_plane):
        """Project socket mask pixels to the socket plane and fit the ring axis.

        A 2D mask bounding-box center is biased by perspective.  For a socket
        lying on a known horizontal plane, first prefer the inner-hole boundary
        because the insertion target is the hole axis, not the area centroid of
        the blue annulus.
        """
        opening = _param_float_list("~socket_opening", [])
        expected_inner = None
        if len(opening) >= 2:
            expected_inner = 0.5 * max(float(opening[0]), float(opening[1]))
        anchor_size = self.anchor_size or []
        expected_outer = None
        if len(anchor_size) >= 2:
            expected_outer = 0.5 * max(float(anchor_size[0]),
                                       float(anchor_size[1]))

        inner_pixels = self._inner_boundary_pixels_from_mask(pose_source)
        if (inner_pixels and
                _param_bool("~anchor_prefer_inner_boundary_circle_fit", True)):
            fit = self._fit_mask_plane_circle(
                inner_pixels, pose_source, z_plane,
                expected_radius=expected_inner,
                method="mask_plane_inner_circle_fit")
            if fit is not None:
                return fit

        outer_pixels = self._outer_boundary_pixels_from_mask(
            pose_source, top_only=True)
        if (outer_pixels and expected_outer is not None and
                _param_bool("~anchor_prefer_outer_silhouette_circle_fit", True)):
            fit = self._fit_mask_plane_circle(
                outer_pixels, pose_source, z_plane,
                expected_radius=expected_outer,
                method="mask_plane_outer_silhouette_circle_fit")
            if fit is not None:
                return fit

        pixels = pose_source.get("mask_pixels_2d") or []
        if len(pixels) < 20:
            return None
        fit = self._fit_mask_plane_circle(
            pixels, pose_source, z_plane,
            expected_radius=None,
            method="mask_plane_annulus_circle_fit")
        if fit is None:
            return None

        if expected_outer is not None:
            min_radius = ((expected_inner * 0.65)
                          if expected_inner is not None
                          else expected_outer * 0.35)
            max_radius = expected_outer * 1.20
            if fit["radius"] < min_radius or fit["radius"] > max_radius:
                return None
        return fit

    def _correct_geometry_center(self, label, pose_source, pose_base):
        """Replace visible-surface median xy with a geometry-aware xy center.

        The raw pointcloud pose is the median of visible masked points.  For
        tall cylinders and circular sockets, the camera sees one side more than
        the other, so that median is a surface center rather than the physical
        axis/hole center.  A percentile bounding-box center is much closer to
        the geometric center while remaining independent of absolute position.
        """
        if not self.use_geometry_center_correction:
            return pose_base

        points_base = self._points_to_base(pose_source)
        if points_base is None:
            return pose_base

        low = np.percentile(points_base, 5, axis=0)
        high = np.percentile(points_base, 95, axis=0)
        box_center = (low + high) * 0.5
        old = list(pose_base.get("position") or [])
        if len(old) < 3:
            return pose_base

        corrected = list(old)
        method = "base_pointcloud_percentile_box_center_xy"
        extra = {}
        if label == "anchor":
            plane_z = None
            if rospy.has_param("~anchor_plane_z"):
                plane_z = float(rospy.get_param("~anchor_plane_z"))
            elif rospy.has_param("~socket_plane_z"):
                plane_z = float(rospy.get_param("~socket_plane_z"))
            ray_center = self._anchor_center_from_mask_ray_plane(
                pose_source, plane_z)
            dark_hole = self._anchor_dark_hole_center_from_rgb(
                pose_source, plane_z)
            mask_plane_fit = self._anchor_ring_center_from_mask_plane(
                pose_source, plane_z)
            top_band_fit = self._anchor_top_band_center_from_points(
                points_base, box_center, plane_z)
            ring_fit = self._anchor_ring_center_from_points(
                points_base, box_center)

            if dark_hole is not None and ray_center is not None:
                dark_xyz = dark_hole.get("center_xyz")
                dark_delta = float(np.linalg.norm(
                    np.asarray(dark_xyz[:2], dtype=np.float64) -
                    np.asarray(ray_center[:2], dtype=np.float64)))
                max_dark_delta = float(rospy.get_param(
                    "~anchor_dark_hole_max_delta_from_ray_m", 0.020))
                if dark_delta > max_dark_delta:
                    rospy.logwarn(
                        "[AnchorPerception] rejecting dark hole center: "
                        "delta from ray-plane center %.1fmm > %.1fmm",
                        dark_delta * 1000.0, max_dark_delta * 1000.0)
                    extra["rejected_dark_hole_center"] = dark_hole
                    dark_hole = None

            if (dark_hole is not None and
                    _param_bool("~anchor_prefer_dark_hole_rgb_center", True)):
                dark_xyz = dark_hole["center_xyz"]
                corrected[0] = float(dark_xyz[0])
                corrected[1] = float(dark_xyz[1])
                corrected[2] = float(dark_xyz[2])
                method = "dark_hole_rgb_ray_plane_xy"
                extra["dark_hole_rgb"] = dark_hole
                if ray_center is not None:
                    extra["ray_plane_center"] = [float(v) for v in ray_center]
                    extra["ray_plane_delta_xy"] = [
                        float(ray_center[0] - corrected[0]),
                        float(ray_center[1] - corrected[1]),
                    ]
                if mask_plane_fit is not None:
                    extra["mask_plane_circle_fit"] = mask_plane_fit
                if top_band_fit is not None:
                    extra["top_band_circle_fit"] = top_band_fit
                if ring_fit is not None:
                    extra["circle_fit"] = ring_fit
            elif (mask_plane_fit is not None and
                    _param_bool("~anchor_prefer_mask_plane_circle_fit", True)):
                if ray_center is not None:
                    fit_delta = float(np.linalg.norm(
                        np.asarray(mask_plane_fit["center_xy"],
                                   dtype=np.float64) -
                        np.asarray(ray_center[:2], dtype=np.float64)))
                    max_fit_delta = float(rospy.get_param(
                        "~anchor_mask_plane_max_delta_from_ray_m", 0.015))
                    if fit_delta > max_fit_delta:
                        rospy.logwarn(
                            "[AnchorPerception] rejecting %s: delta from "
                            "ray-plane center %.1fmm > %.1fmm",
                            mask_plane_fit.get(
                                "fit_method", "mask_plane_circle_fit"),
                            fit_delta * 1000.0, max_fit_delta * 1000.0)
                        extra["rejected_mask_plane_circle_fit"] = mask_plane_fit
                        mask_plane_fit = None

            if (method == "dark_hole_rgb_ray_plane_xy"):
                pass
            elif (mask_plane_fit is not None and
                    _param_bool("~anchor_prefer_mask_plane_circle_fit", True)):
                corrected[0] = float(mask_plane_fit["center_xy"][0])
                corrected[1] = float(mask_plane_fit["center_xy"][1])
                corrected[2] = float(mask_plane_fit["z"])
                method = "{}_xy".format(
                    mask_plane_fit.get("fit_method", "mask_plane_circle_fit"))
                extra["mask_plane_circle_fit"] = mask_plane_fit
                if ray_center is not None:
                    extra["ray_plane_center"] = [float(v) for v in ray_center]
                    extra["ray_plane_delta_xy"] = [
                        float(ray_center[0] - corrected[0]),
                        float(ray_center[1] - corrected[1]),
                    ]
                if ring_fit is not None:
                    extra["circle_fit"] = ring_fit
                if top_band_fit is not None:
                    extra["top_band_circle_fit"] = top_band_fit
            elif top_band_fit is not None and _param_bool(
                    "~anchor_prefer_top_band_circle_fit", True):
                if ray_center is not None:
                    top_delta = float(np.linalg.norm(
                        np.asarray(top_band_fit["center_xy"],
                                   dtype=np.float64) -
                        np.asarray(ray_center[:2], dtype=np.float64)))
                    max_top_delta = float(rospy.get_param(
                        "~anchor_top_band_max_delta_from_ray_m", 0.012))
                    if top_delta > max_top_delta:
                        rospy.logwarn(
                            "[AnchorPerception] rejecting %s: delta from "
                            "ray-plane center %.1fmm > %.1fmm",
                            top_band_fit.get(
                                "fit_method",
                                "base_pointcloud_top_band_circle_fit"),
                            top_delta * 1000.0, max_top_delta * 1000.0)
                        extra["rejected_top_band_circle_fit"] = top_band_fit
                        top_band_fit = None
                if top_band_fit is not None:
                    corrected[0] = float(top_band_fit["center_xy"][0])
                    corrected[1] = float(top_band_fit["center_xy"][1])
                    corrected[2] = float(plane_z)
                    method = "{}_xy".format(top_band_fit.get(
                        "fit_method", "base_pointcloud_top_band_circle_fit"))
                    extra["top_band_circle_fit"] = top_band_fit
                    if ray_center is not None:
                        extra["ray_plane_center"] = [float(v) for v in ray_center]
                        extra["ray_plane_delta_xy"] = [
                            float(ray_center[0] - corrected[0]),
                            float(ray_center[1] - corrected[1]),
                        ]
                    if mask_plane_fit is not None:
                        extra["mask_plane_circle_fit"] = mask_plane_fit
                    if ring_fit is not None:
                        extra["circle_fit"] = ring_fit
            elif ray_center is not None:
                corrected[0] = float(ray_center[0])
                corrected[1] = float(ray_center[1])
                corrected[2] = float(ray_center[2])
                method = "mask_center_ray_plane_xy"
                extra["ray_plane_center"] = [float(v) for v in ray_center]
                if mask_plane_fit is not None:
                    extra["mask_plane_circle_fit"] = mask_plane_fit
                if top_band_fit is not None:
                    extra["top_band_circle_fit"] = top_band_fit
                if ring_fit is not None:
                    extra["circle_fit"] = ring_fit
                    extra["circle_fit_delta_xy"] = [
                        float(ring_fit["center_xy"][0] - ray_center[0]),
                        float(ring_fit["center_xy"][1] - ray_center[1]),
                    ]
            elif ring_fit is not None:
                corrected[0] = float(ring_fit["center_xy"][0])
                corrected[1] = float(ring_fit["center_xy"][1])
                method = "base_pointcloud_circle_fit_xy"
                extra["circle_fit"] = ring_fit
            else:
                corrected[0] = float(box_center[0])
                corrected[1] = float(box_center[1])

            if _param_bool("~anchor_use_hole_depth_center", False):
                hole_center = self._anchor_hole_center_from_mask_center(
                    pose_source)
                if hole_center is not None:
                    candidate = np.asarray(hole_center[:2], dtype=np.float64)
                    reference = np.asarray(corrected[:2], dtype=np.float64)
                    max_delta = float(rospy.get_param(
                        "~anchor_hole_center_max_delta_m", 0.012))
                    if np.linalg.norm(candidate - reference) <= max_delta:
                        corrected[0] = float(hole_center[0])
                        corrected[1] = float(hole_center[1])
                        method = "mask_bbox_center_depth_patch_xy"
                    else:
                        extra["rejected_hole_center"] = [
                            float(hole_center[0]), float(hole_center[1])]
        else:
            corrected[0] = float(box_center[0])
            corrected[1] = float(box_center[1])

        pose_base = dict(pose_base)
        pose_base["position"] = corrected
        pose_base["geometry_center_correction"] = {
            "enabled": True,
            "label": label,
            "method": method,
            "old_position": [float(v) for v in old],
            "new_position": [float(v) for v in corrected],
            "delta_xyz": [
                float(corrected[0] - old[0]),
                float(corrected[1] - old[1]),
                float(corrected[2] - old[2]),
            ],
            "bounds_low_xyz": [float(v) for v in low.tolist()],
            "bounds_high_xyz": [float(v) for v in high.tolist()],
            "target_size": self.target_size,
            "anchor_size": self.anchor_size,
        }
        pose_base["geometry_center_correction"].update(extra)
        rospy.loginfo(
            "[AnchorPerception] %s geometry center correction: "
            "[%.4f %.4f %.4f] -> [%.4f %.4f %.4f] delta=[%.1f %.1f %.1f]mm "
            "method=%s",
            label,
            old[0], old[1], old[2],
            corrected[0], corrected[1], corrected[2],
            (corrected[0] - old[0]) * 1000.0,
            (corrected[1] - old[1]) * 1000.0,
            (corrected[2] - old[2]) * 1000.0,
            method)
        return pose_base

    def detect_scene(self, timeout_s=8.0):
        if not self.wait_for_pointcloud(timeout_s=timeout_s):
            return None
        target = self._detect_one("target", self.target_mask_path)
        anchor = self._detect_one("anchor", self.anchor_mask_path)
        if target is None or anchor is None:
            return None
        rospy.loginfo(
            "[AnchorPerception] target_base=[%.3f %.3f %.3f] "
            "anchor_base=[%.3f %.3f %.3f]",
            target["position_base"][0], target["position_base"][1],
            target["position_base"][2], anchor["position_base"][0],
            anchor["position_base"][1], anchor["position_base"][2])
        return {
            "target": target,
            "anchor": anchor,
            "target_mask_path": self.target_mask_path,
            "anchor_mask_path": self.anchor_mask_path,
        }


if __name__ == "__main__":
    rospy.init_node("mt3_anchor_perception_check", anonymous=True)
    detector = DualMaskAnchorPerception()
    scene = detector.detect_scene()
    if scene is None:
        raise SystemExit(1)
    print(scene)
