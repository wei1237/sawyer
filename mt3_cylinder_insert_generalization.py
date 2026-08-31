#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generalization helpers for cylinder insertion into a shallow socket."""

from mt3_anchor_place_generalization import (
    compute_anchor_place_target,
    compute_target_displacement_place_target,
)


DEFAULT_SOCKET_PROFILE = {
    "name": "blue_insert_socket",
    "category": "shallow_circular_socket",
    "size_m": [0.085, 0.085, 0.100],
    "opening_m": [0.055, 0.055],
    "default_place_offset_xyz": [0.0, 0.0, 0.0],
    "surface_z_offset": 0.0,
}


DEFAULT_CYLINDER_SIZE = [0.045, 0.045, 0.100]


def compute_insert_target(socket_position_base,
                          cylinder_position_base=None,
                          cylinder_size=None,
                          demo_entry=None,
                          socket_profile=None,
                          override_offset_xyz=None,
                          relation_alignment_mode="target_anchor"):
    """Compute the insertion target from the live socket pose.

    The target is the center of the socket opening plus a demo-relative offset.
    For the first baseline demo the offset should normally be [0, 0, 0].
    """
    profile = dict(DEFAULT_SOCKET_PROFILE)
    if socket_profile:
        profile.update(socket_profile)
    if relation_alignment_mode in (
            "target_displacement", "target_only", "no_relation"):
        return compute_target_displacement_place_target(
            cylinder_position_base,
            object_size=cylinder_size or DEFAULT_CYLINDER_SIZE,
            demo_entry=demo_entry,
            default_offset_xyz=profile["default_place_offset_xyz"])
    return compute_anchor_place_target(
        socket_position_base,
        object_position_base=cylinder_position_base,
        object_size=cylinder_size or DEFAULT_CYLINDER_SIZE,
        demo_entry=demo_entry,
        anchor_profile=profile,
        override_offset_xyz=override_offset_xyz)


def cylinder_top_grasp(cylinder_position_base, q_xyzw=None):
    """Top-grasp baseline for the upright green cylinder.

    The existing Sawyer place executor adds TOP_FLANGE_Z_OFFSET internally, so
    this function returns the object/socket contact pose rather than a flange
    pose.
    """
    q = q_xyzw or [-1.0, 0.0, 0.0, 0.0]
    return {
        "position": [float(v) for v in cylinder_position_base[:3]],
        "orientation": [float(v) for v in q],
        "object_size": list(DEFAULT_CYLINDER_SIZE),
    }
