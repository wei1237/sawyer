#!/usr/bin/env python3
"""
Generate cube_template.pkl: a 3D point cloud template of the 4.5cm cube
for geometric retrieval matching. Uses pure Python (no numpy dependency).
"""
import pickle
import os
import math

HALF = 0.0225  # half of 4.5cm


def generate_cube_corners(half_size=HALF):
    """Generate 8 corner points of a cube in local frame."""
    corners = []
    for x in [-half_size, half_size]:
        for y in [-half_size, half_size]:
            for z in [-half_size, half_size]:
                corners.append([x, y, z])
    return corners


def generate_surface_samples(half_size=HALF, samples_per_face=25):
    """Generate uniform surface sample points for each face."""
    points = []
    h = half_size
    n = int(math.sqrt(samples_per_face))
    step = (2 * h) / max(n - 1, 1)
    vals = [-h + i * step for i in range(n)]

    for u in vals:
        for v in vals:
            points.append([ h, u, v])  # +X face
            points.append([-h, u, v])  # -X face
            points.append([u,  h, v])  # +Y face
            points.append([u, -h, v])  # -Y face
            points.append([u, v,  h])  # +Z face
            points.append([u, v, -h])  # -Z face

    return points


def generate_geometric_descriptor(half_size=HALF):
    """Compute geometric features for matching."""
    h = half_size
    side = 2 * h
    return {
        "shape": "box",
        "dimensions": [side, side, side],
        "half_extents": [h, h, h],
        "volume": side ** 3,
        "surface_area": 6 * (side ** 2),
        "aspect_ratio": 1.0,
        "bounding_box": [[-h, -h, -h], [h, h, h]],
        "convexity": 1.0,
        "edge_lengths": [side] * 12,
        "face_normals": [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ],
        "num_vertices": 8,
        "num_edges": 12,
        "num_faces": 6
    }


def main():
    template = {
        "object_type": "cube",
        "size_m": 0.045,
        "half_size_m": HALF,
        "corners_3d": generate_cube_corners(),
        "surface_points": generate_surface_samples(),
        "geometric_descriptor": generate_geometric_descriptor(),
        "description": "4.5cm red cube - template for geometric retrieval in MT3"
    }

    output_path = os.path.join(os.path.dirname(__file__), "cube_template.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(template, f)

    print(f"Template saved to {output_path}")
    print(f"  Corners: {len(template['corners_3d'])} points")
    print(f"  Surface samples: {len(template['surface_points'])} points")
    for k, v in template["geometric_descriptor"].items():
        if not isinstance(v, list) or len(str(v)) < 100:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
