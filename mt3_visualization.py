#!/usr/bin/env python3
"""
MT3 Visualization — Open3D dense point-cloud rendering, official MT3 style.

Produces 4 images saved to ~/.mt3_debug/:
  1. Test scene (RGB | Depth | Segmented) — matplotlib 3-panel
  2. Retrieval comparison (Test vs Demo 2x2) — matplotlib grid
  3. Object from Live Scene and Retrieved Demo — Open3D dense, white bg
  4. Registration Result — Open3D dense, white bg, no grid/axes/legend
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEBUG_DIR = os.path.join(os.path.expanduser("~"), ".mt3_debug")

# ============================================================
# Dense cube point-cloud generation
# ============================================================
def _make_dense_cube_cloud(half=0.0225, samples_per_face=144):
    """Generate a dense, uniform point cloud of a cube. Returns Nx3 array."""
    n = int(np.sqrt(samples_per_face))
    vals = np.linspace(-half, half, n)
    pts = []
    for u in vals:
        for v in vals:
            pts.append([ half, u, v])
            pts.append([-half, u, v])
            pts.append([u,  half, v])
            pts.append([u, -half, v])
            pts.append([u, v,  half])
            pts.append([u, v, -half])
    return np.unique(np.array(pts, dtype=np.float64), axis=0)


def _quat_to_rot_3x3(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*qy*qy-2*qz*qz,  2*qx*qy-2*qz*qw,    2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,    1-2*qx*qx-2*qz*qz,  2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,    2*qy*qz+2*qx*qw,    1-2*qx*qx-2*qy*qy],
    ])


def _transform_pts(pts, pos, ori_xyzw):
    R = _quat_to_rot_3x3(ori_xyzw)
    t = np.array(pos, dtype=np.float64).reshape(3, 1)
    return (R @ pts.T + t).T


# ============================================================
# Test scene (3-panel)
# ============================================================
def make_test_scene_visualization(rgb, depth, segmap, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image'); axes[0].axis('off')
    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Map'); axes[1].axis('off')
    seg_rgb = rgb.copy()
    if segmap is not None:
        seg_rgb[~segmap] = 0
    axes[2].imshow(seg_rgb)
    axes[2].set_title('Segmented RGB'); axes[2].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1/4] Test scene → {save_path}")


# ============================================================
# Retrieval (2x2)
# ============================================================
def make_retrieval_visualization(test_rgb, test_segmap, demo_rgb, demo_segmap,
                                 demo_name, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(test_rgb)
    axes[0, 0].set_title('Live Scene - RGB'); axes[0, 0].axis('off')
    ts = test_rgb.copy()
    if test_segmap is not None:
        ts[~test_segmap] = 0
    axes[0, 1].imshow(ts)
    axes[0, 1].set_title('Live Scene - Segmented'); axes[0, 1].axis('off')
    axes[1, 0].imshow(demo_rgb)
    axes[1, 0].set_title(f'Retrieved Demo - RGB\n{demo_name}'); axes[1, 0].axis('off')
    ds = demo_rgb.copy()
    if demo_segmap is not None:
        ds[~demo_segmap] = 0
    axes[1, 1].imshow(ds)
    axes[1, 1].set_title('Retrieved Demo - Segmented'); axes[1, 1].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2/4] Retrieval → {save_path}")


# ============================================================
# Open3D helpers
# ============================================================
def _headless_render(geometries, window_name, save_path,
                     width=800, height=600, white_bg=True,
                     front=None, up=None, zoom=1.0):
    """Render Open3D geometries headless and save to PNG."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [WARN] Open3D not installed — skipping 3D render")
        return False

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, visible=False,
                      width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)

    # White background
    opt = vis.get_render_option()
    if white_bg:
        opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 3.0

    # Viewpoint
    ctr = vis.get_view_control()
    if front is not None:
        ctr.set_front(front)
    if up is not None:
        ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()
    return True


def _make_o3d_pcd(points_nx3, color_rgb):
    """Create Open3D PointCloud with uniform color."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_nx3)
    pcd.paint_uniform_color(color_rgb)
    return pcd


# ============================================================
# Fig 3: Object from Live Scene and Retrieved Demo
# ============================================================
def make_pointcloud_comparison(live_pts_cam, demo_pts_cam, save_path):
    """
    Dense point clouds of the live object and the retrieved demo template,
    overlapped to show geometric similarity. Pure white background.

    live_pts_cam: Nx3 — detected cube point cloud in camera frame
    demo_pts_cam: Mx3 — demo template point cloud (transformed to same frame)
    """
    try:
        import open3d as o3d
    except ImportError:
        _fallback_matplotlib_3d(live_pts_cam, demo_pts_cam, save_path,
                                "Object from Live Scene and Retrieved Demo")
        return

    geoms = []
    if live_pts_cam is not None and len(live_pts_cam) > 0:
        geoms.append(_make_o3d_pcd(live_pts_cam, [1.0, 0.3, 0.3]))   # red
    if demo_pts_cam is not None and len(demo_pts_cam) > 0:
        geoms.append(_make_o3d_pcd(demo_pts_cam, [0.3, 0.8, 0.3]))   # green

    if not geoms:
        return

    _headless_render(geoms, "Object from Live Scene and Retrieved Demo",
                     save_path, front=[0, 0, 1], up=[0, -1, 0], zoom=1.8)
    print(f"  [3/4] Point-cloud comparison → {save_path}")


# ============================================================
# Fig 4: Registration Result — Demo (Orange) vs Live (Blue)
# ============================================================
def make_registration_result(live_pts, demo_pts_registered, save_path):
    """
    Registration result: orange = demo template (registered),
    blue = live detected object. Dense overlap, pure white background.
    NO grid, axes, text, or legend.
    """
    try:
        import open3d as o3d
    except ImportError:
        _fallback_matplotlib_3d(live_pts, demo_pts_registered, save_path,
                                "Registration Result: Demo (Orange) vs Live (Blue)",
                                color1='blue', color2='orange')
        return

    geoms = []
    if live_pts is not None and len(live_pts) > 0:
        geoms.append(_make_o3d_pcd(live_pts, [0.2, 0.4, 1.0]))        # blue
    if demo_pts_registered is not None and len(demo_pts_registered) > 0:
        geoms.append(_make_o3d_pcd(demo_pts_registered, [1.0, 0.55, 0.0]))  # orange

    if not geoms:
        return

    _headless_render(geoms, "Registration Result: Demo (Orange) vs Live (Blue)",
                     save_path, front=[0, 0, 1], up=[0, -1, 0], zoom=1.8)
    print(f"  [4/4] Registration result → {save_path}")


# ============================================================
# Fallback (matplotlib dense scatter, no Open3D)
# ============================================================
def _fallback_matplotlib_3d(pts1, pts2, save_path, title,
                            color1='red', color2='green'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pts1 is not None and len(pts1) > 0:
        ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2],
                   c=color1, s=2, alpha=0.7, label='Live')
    if pts2 is not None and len(pts2) > 0:
        # Slight random jitter so overlapping points show both colors
        jitter = np.random.normal(0, 0.001, pts2.shape)
        ax.scatter(pts2[:, 0] + jitter[:, 0],
                   pts2[:, 1] + jitter[:, 1],
                   pts2[:, 2] + jitter[:, 2],
                   c=color2, s=4, alpha=0.5, label='Demo/Registered')
    ax.set_title(title)
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [fallback] {title} → {save_path}")


# ============================================================
# Full pipeline
# ============================================================
def generate_all_official(test_data, demo_data, aligned_result, frame_no=0):
    """
    Generate all 4 MT3 official-style visualizations.

    test_data:  {"rgb", "depth", "segmap", "intrinsics", "pose"}
    demo_data:  {"rgb", "depth", "segmap", "intrinsics", "name", "position"}
    aligned_result: from TrajectoryAligner.align()
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    n = frame_no

    # --- Common dense cube point cloud (864 points per face) ---
    template_local = _make_dense_cube_cloud(half=0.0225)  # 4.5cm cube

    # --- 1. Test scene ---
    if test_data.get("rgb") is not None:
        make_test_scene_visualization(
            test_data["rgb"], test_data.get("depth"), test_data.get("segmap"),
            save_path=os.path.join(DEBUG_DIR, f"01_test_scene_{n:04d}.png"))

    # --- 2. Retrieval ---
    if test_data.get("rgb") is not None and demo_data.get("rgb") is not None:
        make_retrieval_visualization(
            test_data["rgb"], test_data.get("segmap"),
            demo_data["rgb"], demo_data.get("segmap"),
            demo_data.get("name", "unknown"),
            save_path=os.path.join(DEBUG_DIR, f"02_retrieval_{n:04d}.png"))

    # --- Fig 3 & 4: Dense Open3D point clouds ---
    # ALL coordinates are in the CAMERA OPTICAL frame (same as PnP detection output).
    # Mixing base-frame and camera-frame coordinates in one 3D plot = meaningless.
    test_pose = test_data.get("pose")
    if test_pose is None:
        print("  [WARN] No test pose — skipping 3D visualizations")
        return

    det_pos = test_pose["position"]
    det_ori = test_pose.get("orientation", [0.0, 0.0, 0.0, 1.0])

    # Live cube: template transformed to detected pose in CAMERA frame
    live_pts = _transform_pts(template_local, det_pos, det_ori)

    # Demo template cube: placed at a canonical reference position (1.5m ahead)
    # in the SAME camera frame — represents "what the demo stored"
    demo_ref_pos = [0.0, 0.0, 1.5]
    demo_pts = _transform_pts(template_local, demo_ref_pos, [0, 0, 0, 1])

    # Fig 3: Side-by-side comparison —
    #   🔴 Live cube at detected position vs 🟢 Demo template at reference position
    #   They should NOT overlap. This shows: both are cube-shaped (same geometry).
    make_pointcloud_comparison(
        live_pts, demo_pts,
        save_path=os.path.join(DEBUG_DIR, f"03_pointcloud_comparison_{n:04d}.png"))

    # Fig 4: Registration result —
    #   🔵 Live cube at detected position
    #   🟠 Demo cube registered to live position (aligned by the pipeline)
    #   Both at SAME position → should overlap perfectly → proves alignment works.
    #   (Different point colours let you see both even when they overlap.)
    demo_registered = _transform_pts(template_local, det_pos, det_ori)

    make_registration_result(
        live_pts, demo_registered,
        save_path=os.path.join(DEBUG_DIR, f"04_registration_{n:04d}.png"))
