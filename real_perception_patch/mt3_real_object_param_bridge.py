#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge ASC60C real perception into the grasp executor's /mt3 params.

This node is intentionally small and real-only.  It does not command Sawyer.
It reads the LangSAM + registered-depth perception result, transforms it into
the Sawyer base frame with the calibrated real TF, then publishes the geometry
contract consumed by mt3_sawyer_real_grasp.py:

    /mt3/current_object_x/y/z          bottom/contact position in base
    /mt3/current_object_size_m         [sx, sy, sz] perception size estimate
    /mt3/current_object_top_z_base     top surface z in base
    /mt3/current_object_z_semantics    bottom_surface_base
"""

import argparse

import numpy as np
import rospy

from mt3_alignment_real import TrajectoryAligner, quat_rotate
from mt3_perception_real import PerceptionNode


def _global_real_param_name(name):
    return "/sawyer_auto_grasp/%s" % str(name).lstrip("~/")


def _param(name, default=None):
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


def _float_list(value, expected_len=None):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return None
        values = [float(v.strip()) for v in text.split(",") if v.strip()]
    else:
        values = [float(v) for v in value]
    if expected_len is not None and len(values) != expected_len:
        raise RuntimeError("Expected %d values, got %s" % (expected_len, values))
    return values


def _tf_xyz_quat(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    return (
        np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64),
        [float(q.x), float(q.y), float(q.z), float(q.w)],
    )


def _transform_points_to_base(aligner, points, source_frame):
    if points is None:
        return None
    tf_msg = aligner._lookup_transform(source_frame, timeout_s=2.0)
    if tf_msg is None:
        return None
    trans, quat = _tf_xyz_quat(tf_msg)

    arr = np.asarray(points, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) == 0:
        return None

    out = []
    for point in arr:
        rotated = quat_rotate(quat, point.tolist())
        out.append((trans + np.asarray(rotated, dtype=np.float64)).tolist())
    return np.asarray(out, dtype=np.float64)


def _pose_source_size(pose_source):
    for key in ("object_size_m", "visible_spread_camera_m"):
        value = pose_source.get(key)
        if value is None:
            continue
        try:
            size = np.asarray(_float_list(value, 3), dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("[RealObjectBridge] Invalid %s=%s: %s",
                          key, value, exc)
            continue
        if np.all(np.isfinite(size)) and np.all(size >= 0.0):
            return size, key
    return None, ""


class RealObjectParamBridge(object):

    def __init__(self):
        self.perception = PerceptionNode()
        self.aligner = TrajectoryAligner()
        self.size_percentile_low = float(_param("object_size_percentile_low", 5.0))
        self.size_percentile_high = float(_param("object_size_percentile_high", 95.0))
        self.top_z_percentile = float(_param("object_top_z_percentile", 90.0))
        self.publish_camera_spread = _param_bool("publish_camera_spread", True)
        self.default_top_z_offset = (
            float(_param("real_top_z_offset_m", 0.044))
            if _param_bool("enable_real_top_z_offset", True)
            else 0.0
        )
        rospy.loginfo(
            "[RealObjectBridge] top-Z config: percentile=%.1f offset=%+.4fm",
            self.top_z_percentile,
            self.default_top_z_offset)

    def _estimate_base_geometry(self):
        pose_source = self.perception.get_object_pose()
        if pose_source is None:
            return None

        pose_base = self.aligner.transform_camera_to_base(pose_source)
        if pose_base is None:
            return None

        source_frame = str(pose_source.get("source_frame") or "")
        points_base = _transform_points_to_base(
            self.aligner, pose_source.get("object_points"), source_frame)
        if points_base is None or len(points_base) < 5:
            rospy.logwarn("[RealObjectBridge] Not enough base-frame object points")
            return None

        low = np.percentile(points_base, self.size_percentile_low, axis=0)
        high = np.percentile(points_base, self.size_percentile_high, axis=0)
        visible_size_base = np.maximum(high - low, 0.0)

        size_base, size_source = _pose_source_size(pose_source)
        if size_base is None:
            size_base = visible_size_base
            size_source = "base_visible_points_percentile_fallback"

        override_size = _float_list(_param("live_object_size_override_m", None), 3)
        if override_size is not None:
            size_base = np.asarray(override_size, dtype=np.float64)
            size_source = "live_object_size_override_m"

        top_z_raw = float(np.percentile(points_base[:, 2], self.top_z_percentile))
        top_z_offset = float(
            _param("live_object_top_z_offset_m", self.default_top_z_offset)
        )
        top_z = top_z_raw + top_z_offset
        bottom_z = top_z - float(size_base[2])

        transformed_camera_median_base = np.asarray(
            pose_base["position"], dtype=np.float64)
        surface_center_base = np.median(points_base, axis=0)
        center_definition_delta = (
            surface_center_base - transformed_camera_median_base)
        bottom_position = np.array(
            [
                surface_center_base[0],
                surface_center_base[1],
                bottom_z,
            ],
            dtype=np.float64
        )

        result = {
            "pose_source": pose_source,
            "pose_base": pose_base,
            "transformed_camera_median_base": transformed_camera_median_base,
            "surface_center_base": surface_center_base,
            "center_definition_delta": center_definition_delta,
            "bottom_position": bottom_position,
            "size_base": size_base,
            "size_source": size_source,
            "visible_size_base": visible_size_base,
            "top_z_raw": top_z_raw,
            "top_z": top_z,
            "top_z_offset": top_z_offset,
            "source_frame": source_frame,
            "points_base_count": int(len(points_base)),
        }
        return result

    def _publish(self, geom):
        pos = geom["bottom_position"]
        size = geom["size_base"]

        rospy.set_param("/mt3/current_object_x", float(pos[0]))
        rospy.set_param("/mt3/current_object_y", float(pos[1]))
        rospy.set_param("/mt3/current_object_z", float(pos[2]))
        rospy.set_param("/mt3/current_object_size_m", [float(v) for v in size])
        rospy.set_param("/mt3/current_object_size_source", str(geom["size_source"]))
        rospy.set_param(
            "/mt3/current_object_size_base_visible_m",
            [float(v) for v in geom["visible_size_base"]]
        )
        rospy.set_param("/mt3/current_object_top_z_base", float(geom["top_z"]))
        rospy.set_param("/mt3/current_object_top_z_raw_base", float(geom["top_z_raw"]))
        rospy.set_param("/mt3/current_object_top_z_offset_m", float(geom["top_z_offset"]))
        rospy.set_param("/mt3/current_object_z_semantics", "bottom_surface_base")
        rospy.set_param("/mt3/current_object_source_frame", "base")
        rospy.set_param(
            "/mt3/current_object_surface_center_base",
            [float(v) for v in geom["surface_center_base"]]
        )
        rospy.set_param(
            "/mt3/current_object_transformed_camera_median_base",
            [float(v) for v in geom["transformed_camera_median_base"]]
        )
        rospy.set_param(
            "/mt3/current_object_center_definition_delta_m",
            [float(v) for v in geom["center_definition_delta"]]
        )
        rospy.set_param(
            "/mt3/current_object_points_base_count",
            int(geom["points_base_count"])
        )

        if self.publish_camera_spread:
            pose_source = geom["pose_source"]
            if pose_source.get("object_size_m") is not None:
                rospy.set_param(
                    "/mt3/current_object_size_camera_m",
                    [float(v) for v in pose_source["object_size_m"]]
                )

    def update_once(self):
        geom = self._estimate_base_geometry()
        if geom is None:
            return False

        self._publish(geom)

        rospy.loginfo("=== REAL PERCEPTION PARAM BRIDGE ===")
        rospy.loginfo(
            "camera surface center: %s frame=%s",
            geom["pose_source"].get("position"),
            geom["source_frame"]
        )
        rospy.loginfo(
            "transformed camera-median center: %s",
            geom["transformed_camera_median_base"]
        )
        rospy.loginfo("base surface center: %s", geom["surface_center_base"])
        delta = geom["center_definition_delta"]
        rospy.loginfo(
            "base median minus transformed camera-median: "
            "[%.1f %.1f %.1f]mm",
            delta[0] * 1000.0,
            delta[1] * 1000.0,
            delta[2] * 1000.0,
        )
        rospy.loginfo("published bottom object: %s", geom["bottom_position"])
        rospy.loginfo("published size: %s source=%s",
                      geom["size_base"], geom["size_source"])
        rospy.loginfo("base visible point spread diagnostic: %s",
                      geom["visible_size_base"])
        rospy.loginfo(
            "published top_z: %.4f raw=%.4f offset=%+.4f points=%d",
            geom["top_z"],
            geom["top_z_raw"],
            geom["top_z_offset"],
            geom["points_base_count"]
        )
        return True

    def spin(self, rate_hz=1.0):
        rate = rospy.Rate(float(rate_hz))
        while not rospy.is_shutdown():
            self.update_once()
            rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--rate_hz", type=float, default=1.0)
    args, _ = parser.parse_known_args()

    rospy.init_node("mt3_real_object_param_bridge", anonymous=True)
    bridge = RealObjectParamBridge()

    timeout_s = float(_param("perception_timeout_s", 8.0))
    if not bridge.perception.wait_for_registered_rgbd(timeout_s=timeout_s):
        raise RuntimeError("ASC60C RGB/depth/CameraInfo timeout")

    if args.once:
        if not bridge.update_once():
            raise RuntimeError("Failed to publish current real object geometry")
        return

    bridge.spin(rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
