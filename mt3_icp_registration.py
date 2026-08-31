#!/usr/bin/env python3
"""
Lightweight point-cloud ICP registration for MT3 scene packages.

This is the second-stage bridge toward the paper pipeline:
  demo pointcloud + live pointcloud -> relative rigid transform.

It intentionally uses only NumPy/Matplotlib so it can run in the current ROS VM
without installing Open3D first. The result is for validation and visualization;
the grasp pipeline still uses the existing center-based alignment until this
registration is validated across more scenes.
"""
import json
import os

import numpy as np


def load_package_pointcloud(package_dir):
    path = os.path.join(package_dir, "pointcloud.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    points = np.load(path).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        points = points.reshape((-1, 3))
    points = points[np.all(np.isfinite(points), axis=1)]
    return points


def robust_filter(points, keep_percentile=92.0):
    """Remove far outliers around the cloud median."""
    if len(points) < 8:
        return points
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    cutoff = np.percentile(dist, keep_percentile)
    return points[dist <= cutoff]


def uniform_sample(points, max_points=1200):
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx]


def nearest_neighbors(src, dst, chunk_size=512):
    """Return nearest dst point and distance for every src point."""
    indices = []
    distances = []
    for start in range(0, len(src), chunk_size):
        block = src[start:start + chunk_size]
        diff = block[:, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        indices.append(idx)
        distances.append(np.sqrt(d2[np.arange(len(block)), idx]))
    return np.concatenate(indices), np.concatenate(distances)


def estimate_rigid_transform(src, dst):
    """Kabsch alignment from src points to corresponding dst points."""
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(points, T):
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def icp_register(source_points, target_points, max_iterations=40,
                 distance_percentile=85.0, tolerance=1e-5):
    """
    Align source to target using point-to-point ICP.

    Returns:
      transform: 4x4 matrix mapping source -> target
      registered_source: transformed source points
      metrics: error and iteration info
    """
    source = uniform_sample(robust_filter(source_points))
    target = uniform_sample(robust_filter(target_points))
    if len(source) < 8 or len(target) < 8:
        raise ValueError("Need at least 8 valid points in both point clouds")

    # Start with centroid alignment. This makes the first NN pass stable even
    # when demo/live clouds are in different camera-relative positions.
    T_total = np.eye(4)
    T_total[:3, 3] = np.median(target, axis=0) - np.median(source, axis=0)
    moving = apply_transform(source, T_total)

    last_error = None
    history = []
    for iteration in range(max_iterations):
        nn_idx, distances = nearest_neighbors(moving, target)
        cutoff = np.percentile(distances, distance_percentile)
        keep = distances <= cutoff
        if np.count_nonzero(keep) < 8:
            keep = np.ones_like(distances, dtype=bool)

        T_step = estimate_rigid_transform(moving[keep], target[nn_idx[keep]])
        moving = apply_transform(moving, T_step)
        T_total = T_step @ T_total

        mean_error = float(np.mean(distances[keep]))
        history.append(mean_error)
        if last_error is not None and abs(last_error - mean_error) < tolerance:
            break
        last_error = mean_error

    _, final_dist = nearest_neighbors(moving, target)
    metrics = {
        "iterations": len(history),
        "mean_error_m": float(np.mean(final_dist)),
        "median_error_m": float(np.median(final_dist)),
        "p90_error_m": float(np.percentile(final_dist, 90)),
        "source_points": int(len(source)),
        "target_points": int(len(target)),
        "history": history,
    }
    return T_total, moving, metrics


def save_icp_outputs(demo_package_dir, live_package_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    demo_points = load_package_pointcloud(demo_package_dir)
    live_points = load_package_pointcloud(live_package_dir)
    T, registered_demo, metrics = icp_register(demo_points, live_points)

    np.save(os.path.join(output_dir, "T_demo_to_live_icp.npy"), T)
    np.save(os.path.join(output_dir, "registered_demo_pointcloud.npy"), registered_demo)

    metadata = {
        "format": "mt3_icp_result_v1",
        "demo_package": demo_package_dir,
        "live_package": live_package_dir,
        "transform_demo_to_live": T.tolist(),
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "icp_result.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    save_icp_visualization(registered_demo, live_points, output_dir)
    return metadata


def save_icp_visualization(registered_demo, live_points, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demo = uniform_sample(registered_demo, 1200)
    live = uniform_sample(robust_filter(live_points), 1200)

    fig = plt.figure(figsize=(10, 4), dpi=120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(live[:, 0], live[:, 1], s=4, c="#1f77b4", label="live")
    ax1.scatter(demo[:, 0], demo[:, 1], s=4, c="#ff7f0e", label="demo registered")
    ax1.set_title("XY alignment")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(live[:, 0], live[:, 2], s=4, c="#1f77b4", label="live")
    ax2.scatter(demo[:, 0], demo[:, 2], s=4, c="#ff7f0e", label="demo registered")
    ax2.set_title("XZ alignment")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.axis("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "icp_alignment.png"))
    plt.close(fig)


def main():
    import argparse

    default_root = os.path.join(os.path.dirname(__file__), "demo_library", "scene_packages")
    parser = argparse.ArgumentParser(description="Run ICP on MT3 scene packages")
    parser.add_argument("--demo", default=os.path.join(default_root, "demo_cube_top_grasp_v2"))
    parser.add_argument("--live", default=os.path.join(default_root, "live_latest"))
    parser.add_argument("--out", default=os.path.join(default_root, "icp_latest"))
    args = parser.parse_args()

    result = save_icp_outputs(args.demo, args.live, args.out)
    m = result["metrics"]
    print("========== MT3 ICP Registration ==========")
    print("demo:", args.demo)
    print("live:", args.live)
    print("out: ", args.out)
    print("iterations:", m["iterations"])
    print("mean_error_m:   {:.4f}".format(m["mean_error_m"]))
    print("median_error_m: {:.4f}".format(m["median_error_m"]))
    print("p90_error_m:    {:.4f}".format(m["p90_error_m"]))
    print("T_demo_to_live:")
    print(np.array(result["transform_demo_to_live"]))
    print("==========================================")


if __name__ == "__main__":
    main()

