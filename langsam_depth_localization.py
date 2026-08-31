#!/usr/bin/env python3
"""Localize a LangSAM mask with ROS depth data.

This is a standalone perception test:
  1. load a mask generated on Windows by LangSAM,
  2. read one organized PointCloud2 message, or depth+CameraInfo as fallback,
  3. extract masked 3D object points,
  4. optionally transform the point from camera frame to Sawyer base frame.
"""

import os
import sys
import struct

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf.transformations import quaternion_matrix


def image_to_numpy(msg):
    """Convert common ROS depth image encodings to a float depth map in meters."""
    if msg.encoding in ("32FC1", "32FC"):
        dtype = np.float32
        scale = 1.0
    elif msg.encoding in ("16UC1", "mono16"):
        dtype = np.uint16
        scale = 0.001
    else:
        raise ValueError("Unsupported depth encoding: {}".format(msg.encoding))

    depth = np.frombuffer(msg.data, dtype=dtype)
    depth = depth.reshape((msg.height, msg.width))

    if msg.is_bigendian and depth.dtype.byteorder != ">":
        depth = depth.byteswap().newbyteorder()

    return depth.astype(np.float32) * scale


def resize_mask_nearest(mask, target_height, target_width):
    """Resize a boolean mask with nearest-neighbor sampling using only numpy."""
    if mask.shape == (target_height, target_width):
        return mask

    y_idx = np.floor(np.arange(target_height) * mask.shape[0] / target_height).astype(int)
    x_idx = np.floor(np.arange(target_width) * mask.shape[1] / target_width).astype(int)
    y_idx = np.clip(y_idx, 0, mask.shape[0] - 1)
    x_idx = np.clip(x_idx, 0, mask.shape[1] - 1)
    return mask[y_idx[:, None], x_idx[None, :]]


def masked_depth_position(mask, depth, camera_info):
    """Return object center pixel, median depth, and 3D point in camera frame."""
    mask = resize_mask_nearest(mask, depth.shape[0], depth.shape[1])
    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)

    if np.count_nonzero(mask) == 0:
        raise RuntimeError("Mask is empty.")
    if np.count_nonzero(valid) < 5:
        raise RuntimeError(
            "Too few valid depth pixels in mask: {}".format(np.count_nonzero(valid))
        )

    ys, xs = np.where(valid)
    u = float(np.mean(xs))
    v = float(np.mean(ys))

    object_depth_values = depth[valid]
    z = float(np.median(object_depth_values))

    fx = float(camera_info.K[0])
    fy = float(camera_info.K[4])
    cx = float(camera_info.K[2])
    cy = float(camera_info.K[5])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return {
        "u": u,
        "v": v,
        "z": z,
        "x": x,
        "y": y,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def point_field_offsets(cloud_msg):
    offsets = {field.name: field.offset for field in cloud_msg.fields}
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise RuntimeError("PointCloud2 missing '{}' field".format(name))
    return offsets["x"], offsets["y"], offsets["z"]


def read_xyz_at_flat_indices(cloud_msg, indices):
    """Read xyz from a PointCloud2 that may be organized or flattened."""
    x_off, y_off, z_off = point_field_offsets(cloud_msg)
    endian = ">" if cloud_msg.is_bigendian else "<"
    fmt = endian + "fff"
    data = cloud_msg.data
    step = cloud_msg.point_step

    points = []
    for idx in indices:
        base = int(idx) * step
        if base + max(x_off, y_off, z_off) + 4 > len(data):
            continue
        x = struct.unpack_from(fmt[0] + "f", data, base + x_off)[0]
        y = struct.unpack_from(fmt[0] + "f", data, base + y_off)[0]
        z = struct.unpack_from(fmt[0] + "f", data, base + z_off)[0]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            points.append((x, y, z))
    return points


def masked_cloud_position(mask, cloud_msg):
    """Return median 3D point from mask pixels in PointCloud2."""
    if cloud_msg.height > 1:
        cloud_height = cloud_msg.height
        cloud_width = cloud_msg.width
    else:
        cloud_width = mask.shape[1]
        if cloud_msg.width % cloud_width != 0:
            raise RuntimeError(
                "Flattened PointCloud2 width {} does not match mask width {}".format(
                    cloud_msg.width, cloud_width
                )
            )
        cloud_height = cloud_msg.width // cloud_width

    mask = resize_mask_nearest(mask, cloud_height, cloud_width)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError("Mask is empty.")

    flat_indices = ys.astype(np.int64) * cloud_width + xs.astype(np.int64)
    points = read_xyz_at_flat_indices(cloud_msg, flat_indices)

    if len(points) < 5:
        raise RuntimeError(
            "Too few valid point cloud pixels in mask: {}".format(len(points))
        )

    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    spread = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)

    return {
        "u": float(np.mean(xs)),
        "v": float(np.mean(ys)),
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "valid_points": int(len(points)),
        "spread_x": float(spread[0]),
        "spread_y": float(spread[1]),
        "spread_z": float(spread[2]),
        "cloud_width": int(cloud_width),
        "cloud_height": int(cloud_height),
    }


def transform_point(tf_buffer, target_frame, source_frame, point_xyz):
    """Transform a 3D point using TF and return coordinates in target_frame."""
    transform = tf_buffer.lookup_transform(
        target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0)
    )

    trans = transform.transform.translation
    quat = transform.transform.rotation
    mat = quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    rot = mat[:3, :3]

    point = np.asarray(point_xyz, dtype=np.float64)
    out = rot.dot(point) + np.array([trans.x, trans.y, trans.z], dtype=np.float64)
    return out, transform


def main():
    rospy.init_node("langsam_depth_localization", anonymous=True)

    mask_path = rospy.get_param(
        "~mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy"
    )
    mode = rospy.get_param("~mode", "cloud")
    cloud_topic = rospy.get_param(
        "~cloud_topic", "/io/internal_camera/head_camera/depth/points"
    )
    depth_topic = rospy.get_param(
        "~depth_topic", "/head_camera/depth/image_raw"
    )
    camera_info_topic = rospy.get_param(
        "~camera_info_topic", "/head_camera/depth/camera_info"
    )
    target_frame = rospy.get_param("~target_frame", "base")

    if not os.path.exists(mask_path):
        rospy.logerr("Mask file not found: %s", mask_path)
        return 1

    mask = np.load(mask_path).astype(bool)
    rospy.loginfo("Loaded mask: %s shape=%s pixels=%d", mask_path, mask.shape, np.count_nonzero(mask))

    if mode == "cloud":
        rospy.loginfo("Waiting for point cloud: %s", cloud_topic)
        cloud_msg = rospy.wait_for_message(cloud_topic, PointCloud2, timeout=10.0)
        result = masked_cloud_position(mask, cloud_msg)
        source_frame = cloud_msg.header.frame_id
        method = "mask + organized PointCloud2"
    else:
        rospy.loginfo("Waiting for depth image: %s", depth_topic)
        depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=10.0)
        rospy.loginfo("Waiting for camera info: %s", camera_info_topic)
        camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=10.0)

        depth = image_to_numpy(depth_msg)
        result = masked_depth_position(mask, depth, camera_info)
        source_frame = depth_msg.header.frame_id or camera_info.header.frame_id
        method = "mask + depth image + CameraInfo"

    point_camera = np.array([result["x"], result["y"], result["z"]], dtype=np.float64)

    print("")
    print("========== LangSAM Mask Localization ==========")
    print("method: {}".format(method))
    print("mask_path: {}".format(mask_path))
    if mode == "cloud":
        print("cloud_topic: {}".format(cloud_topic))
        print(
            "cloud image size used: {}x{}".format(
                result["cloud_width"], result["cloud_height"]
            )
        )
    else:
        print("depth_topic: {}".format(depth_topic))
        print("camera_info_topic: {}".format(camera_info_topic))
    print("source_frame: {}".format(source_frame))
    if mode == "cloud":
        print("mask valid cloud points: {}".format(result["valid_points"]))
        print(
            "cloud spread p10-p90: x={:.4f}, y={:.4f}, z={:.4f}".format(
                result["spread_x"], result["spread_y"], result["spread_z"]
            )
        )
    else:
        print("mask valid depth pixels: {}".format(result["valid_pixels"]))
        print("median depth: {:.4f} m".format(result["z"]))
    print("mask center pixel: u={:.1f}, v={:.1f}".format(result["u"], result["v"]))
    print(
        "object_in_camera: x={:.4f}, y={:.4f}, z={:.4f}".format(
            point_camera[0], point_camera[1], point_camera[2]
        )
    )

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    try:
        point_base, transform = transform_point(
            tf_buffer, target_frame, source_frame, point_camera
        )
        print("target_frame: {}".format(target_frame))
        print(
            "object_in_{}: x={:.4f}, y={:.4f}, z={:.4f}".format(
                target_frame, point_base[0], point_base[1], point_base[2]
            )
        )
        print(
            "tf used: {} -> {}".format(
                transform.header.frame_id, transform.child_frame_id
            )
        )
    except Exception as exc:
        print("TF transform failed: {}".format(exc))
        print("Camera-frame localization still succeeded.")

    print("===============================================")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSException as exc:
        rospy.logerr("ROS error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        rospy.logerr("Fatal error: %s", exc)
        sys.exit(1)
