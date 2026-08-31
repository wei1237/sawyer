#!/usr/bin/env python3
"""
Build a real demo scene package from a saved RGB image, LangSAM mask, and live PointCloud2.

Use this when the object is placed at the demonstration reference pose. It creates
demo_library/scene_packages/<name>/ with RGB, mask, point cloud, intrinsics, and metadata.
"""
import os
import struct

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from mt3_scene_package import save_scene_package


def resize_mask_nearest(mask, target_height, target_width):
    if mask.shape == (target_height, target_width):
        return mask
    y_idx = np.floor(np.arange(target_height) * mask.shape[0] / target_height).astype(int)
    x_idx = np.floor(np.arange(target_width) * mask.shape[1] / target_width).astype(int)
    y_idx = np.clip(y_idx, 0, mask.shape[0] - 1)
    x_idx = np.clip(x_idx, 0, mask.shape[1] - 1)
    return mask[y_idx[:, None], x_idx[None, :]]


def point_field_offsets(cloud_msg):
    offsets = {field.name: field.offset for field in cloud_msg.fields}
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise RuntimeError("PointCloud2 missing '{}' field".format(name))
    return offsets["x"], offsets["y"], offsets["z"]


def cloud_size_for_mask(cloud_msg, mask_width):
    if cloud_msg.height > 1:
        return cloud_msg.width, cloud_msg.height
    if cloud_msg.width % mask_width != 0:
        raise RuntimeError(
            "Flattened PointCloud2 width {} does not match mask width {}".format(
                cloud_msg.width, mask_width))
    return mask_width, cloud_msg.width // mask_width


def read_xyz_at_indices(cloud_msg, indices):
    x_off, y_off, z_off = point_field_offsets(cloud_msg)
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
    return np.asarray(points, dtype=np.float64)


def build_pointcloud_from_mask(mask, cloud_msg):
    cloud_width, cloud_height = cloud_size_for_mask(cloud_msg, mask.shape[1])
    mask = resize_mask_nearest(mask.astype(bool), cloud_height, cloud_width)
    ys, xs = np.where(mask)
    if len(xs) < 5:
        raise RuntimeError("Mask too small: {} pixels".format(len(xs)))
    indices = ys.astype(np.int64) * cloud_width + xs.astype(np.int64)
    points = read_xyz_at_indices(cloud_msg, indices)
    if len(points) < 5:
        raise RuntimeError("Too few valid cloud points: {}".format(len(points)))
    return mask, points


def main():
    rospy.init_node("make_real_demo_scene_package", anonymous=True)

    name = rospy.get_param("~name", "demo_cube_top_grasp_v2_real")
    rgb_path = rospy.get_param("~rgb_path", "/mnt/hgfs2/tmp_vision/current_rgb.png")
    mask_path = rospy.get_param("~mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")
    pointcloud_topic = rospy.get_param(
        "~pointcloud_topic", "/io/internal_camera/head_camera/depth/points")
    out_root = rospy.get_param(
        "~out_root",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "demo_library", "scene_packages"))

    import cv2
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError("Failed to read RGB image: {}".format(rgb_path))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    mask = np.load(mask_path).astype(bool)

    rospy.loginfo("Loaded RGB: %s shape=%s", rgb_path, rgb.shape)
    rospy.loginfo("Loaded mask: %s shape=%s pixels=%d", mask_path, mask.shape, int(np.count_nonzero(mask)))
    rospy.loginfo("Waiting for point cloud: %s", pointcloud_topic)
    cloud_msg = rospy.wait_for_message(pointcloud_topic, PointCloud2, timeout=10.0)
    resized_mask, points = build_pointcloud_from_mask(mask, cloud_msg)
    center = np.median(points, axis=0)
    spread = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)

    # Intrinsics from Sawyer Gazebo head camera xacro.
    intrinsics = np.array([
        [407.391526, 0.0, 640.5],
        [0.0, 407.391526, 400.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    depth = np.zeros(resized_mask.shape, dtype=np.float32)
    scene_data = {
        "rgb": rgb,
        "depth": depth,
        "segmap": resized_mask,
        "intrinsics": intrinsics,
        "pose": {
            "position": center.tolist(),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "method": "real_demo_langsam_pointcloud",
            "confidence": 1.0,
            "object_points": points.tolist(),
            "source_frame": cloud_msg.header.frame_id,
        },
    }
    package = save_scene_package(
        scene_data,
        out_root,
        name=name,
        role="real_demo",
        extra_metadata={
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "pointcloud_topic": pointcloud_topic,
            "source_frame": cloud_msg.header.frame_id,
            "pointcloud_center": center.tolist(),
            "pointcloud_spread_p10_p90": spread.tolist(),
        })

    print("========== Real Demo Scene Package ==========")
    print("package:", package["package_dir"])
    print("mask pixels:", package["stats"]["segmap_pixels"])
    print("pointcloud points:", package["stats"]["pointcloud_points"])
    print("source_frame:", cloud_msg.header.frame_id)
    print("center:", ["{:.4f}".format(v) for v in center])
    print("spread p10-p90:", ["{:.4f}".format(v) for v in spread])
    print("============================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rospy.logerr("Fatal error: %s", e)
        import traceback
        traceback.print_exc()
