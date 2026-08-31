#!/usr/bin/env python3
"""
Unified MT3 scene package helpers.

This module saves the same core data for live scenes and demonstrations:
RGB, depth, target mask, target point cloud, camera intrinsics, and pose metadata.
The package is intentionally simple so the next stage can feed it to ICP or
PointNet++ without depending on the current pipeline internals.
"""
import json
import os
import time

import numpy as np


SCENE_PACKAGE_VERSION = "mt3_scene_package_v1"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _as_numpy(value, dtype=None):
    if value is None:
        return None
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _pose_to_jsonable(pose):
    if pose is None:
        return None
    out = {}
    for key, value in pose.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = [
                v.tolist() if isinstance(v, np.ndarray) else v
                for v in value
            ]
        elif key == "object_points":
            # Stored separately as pointcloud.npy.
            continue
        else:
            out[key] = value
    return out


def _depth_to_meters(depth):
    depth = np.asarray(depth)
    if depth.size == 0:
        return depth.astype(np.float64)
    valid = depth[np.isfinite(depth)]
    valid = valid[valid > 0]
    if valid.size == 0:
        return depth.astype(np.float64)
    # ROS/Gazebo depth images may be float meters or uint millimeters.
    scale = 0.001 if np.median(valid) > 20.0 else 1.0
    return depth.astype(np.float64) * scale


def pointcloud_from_depth_mask(depth, segmap, intrinsics, max_points=4096):
    """Back-project a target mask into camera-frame xyz points."""
    if depth is None or segmap is None or intrinsics is None:
        return None

    depth_m = _depth_to_meters(depth)
    mask = np.asarray(segmap).astype(bool)
    if mask.shape != depth_m.shape[:2]:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0.01) & (z < 10.0)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    if len(z) == 0:
        return None

    if len(z) > max_points:
        idx = np.linspace(0, len(z) - 1, max_points).astype(np.int64)
        xs, ys, z = xs[idx], ys[idx], z[idx]

    K = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def build_scene_arrays(scene_data):
    """Return normalized arrays from a pipeline-style scene dictionary."""
    rgb = _as_numpy(scene_data.get("rgb"), dtype=np.uint8)
    depth = _as_numpy(scene_data.get("depth"))
    segmap = _as_numpy(scene_data.get("segmap"), dtype=bool)
    intrinsics = _as_numpy(scene_data.get("intrinsics"), dtype=np.float64)

    pose = scene_data.get("pose") or {}
    pointcloud = pose.get("object_points")
    pointcloud = _as_numpy(pointcloud, dtype=np.float64)
    if pointcloud is not None and pointcloud.ndim != 2:
        pointcloud = pointcloud.reshape((-1, 3))
    if pointcloud is None:
        pointcloud = pointcloud_from_depth_mask(depth, segmap, intrinsics)

    return {
        "rgb": rgb,
        "depth": depth,
        "segmap": segmap,
        "intrinsics": intrinsics,
        "pointcloud": pointcloud,
        "pose": pose,
    }


def save_scene_package(scene_data, output_dir, name, role, extra_metadata=None):
    """
    Save a unified scene package and return its metadata dictionary.

    Files:
      - rgb.png
      - depth.npy
      - segmap.npy
      - pointcloud.npy
      - intrinsics.npy
      - metadata.json
    """
    import cv2

    package_dir = _ensure_dir(os.path.join(output_dir, name))
    arrays = build_scene_arrays(scene_data)

    files = {}
    if arrays["rgb"] is not None:
        rgb_path = os.path.join(package_dir, "rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(arrays["rgb"], cv2.COLOR_RGB2BGR))
        files["rgb"] = rgb_path

    for key in ("depth", "segmap", "intrinsics", "pointcloud"):
        arr = arrays[key]
        if arr is None:
            continue
        path = os.path.join(package_dir, f"{key}.npy")
        np.save(path, arr)
        files[key] = path

    metadata = {
        "format": SCENE_PACKAGE_VERSION,
        "role": role,
        "name": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {k: os.path.basename(v) for k, v in files.items()},
        "pose": _pose_to_jsonable(arrays["pose"]),
        "stats": {
            "rgb_shape": list(arrays["rgb"].shape) if arrays["rgb"] is not None else None,
            "depth_shape": list(arrays["depth"].shape) if arrays["depth"] is not None else None,
            "segmap_shape": list(arrays["segmap"].shape) if arrays["segmap"] is not None else None,
            "segmap_pixels": int(np.count_nonzero(arrays["segmap"])) if arrays["segmap"] is not None else 0,
            "pointcloud_points": int(len(arrays["pointcloud"])) if arrays["pointcloud"] is not None else 0,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = os.path.join(package_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    metadata["package_dir"] = package_dir
    return metadata
