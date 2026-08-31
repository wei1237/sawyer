#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchor-object place target generalization.

For anchored placement the target is not an absolute table coordinate.  A demo
stores the relative placement in the anchor frame:

    demo_place_xyz - demo_anchor_xyz

At test time we detect the live anchor and translate that same relative
placement to the live scene.
"""

import copy


DEFAULT_ANCHOR_PROFILE = {
    "name": "blue_placement_platform",
    "category": "small_platform",
    "size_m": [0.10, 0.10, 0.02],
    "default_place_offset_xyz": [0.0, 0.0, 0.0],
    "surface_z_offset": 0.0,
}


def _as_xyz(value, name):
    if value is None or len(value) < 3:
        raise ValueError("%s must contain at least [x, y, z]" % name)
    return [float(value[0]), float(value[1]), float(value[2])]


def demo_anchor_place_offset(demo_entry, default_offset=None):
    """Return demo place offset relative to the demo anchor center.

    The returned vector is [dx, dy, dz] in base/table axes.  For the blue tray
    baseline this is normally close to [0, 0, 0], meaning "place at tray center".
    """
    default_offset = default_offset or DEFAULT_ANCHOR_PROFILE["default_place_offset_xyz"]
    anchor_info = demo_entry.get("anchor_info", {}) or {}
    place_info = demo_entry.get("place_info", {}) or {}

    anchor_pos = (
        anchor_info.get("position_base")
        or anchor_info.get("position")
        or anchor_info.get("pose_base_frame", {}).get("position")
    )
    place_pos = (
        place_info.get("place_xyz")
        or place_info.get("place_pose_base_frame", {}).get("position")
        or place_info.get("alignment_pose_base_frame", {}).get("position")
    )
    if not anchor_pos or not place_pos:
        return [float(v) for v in default_offset]

    anchor_xyz = _as_xyz(anchor_pos, "demo anchor position")
    place_xyz = _as_xyz(place_pos, "demo place position")
    return [
        place_xyz[0] - anchor_xyz[0],
        place_xyz[1] - anchor_xyz[1],
        place_xyz[2] - anchor_xyz[2],
    ]


def demo_target_place_offset(demo_entry, default_offset=None):
    """Return the demonstrated place displacement from the target object.

    This intentionally ignores the live anchor.  It is useful as an ablation:
    if the anchor moves independently of the target, this fixed target-relative
    displacement should no longer resolve the correct destination.
    """
    default_offset = default_offset or [0.0, 0.0, 0.0]
    object_info = demo_entry.get("object_info", {}) or {}
    place_info = demo_entry.get("place_info", {}) or {}

    object_pos = (
        object_info.get("position_base")
        or object_info.get("position")
        or object_info.get("pose_base_frame", {}).get("position")
    )
    place_pos = (
        place_info.get("place_xyz")
        or place_info.get("place_pose_base_frame", {}).get("position")
        or place_info.get("alignment_pose_base_frame", {}).get("position")
    )
    if not object_pos or not place_pos:
        return [float(v) for v in default_offset]

    object_xyz = _as_xyz(object_pos, "demo object position")
    place_xyz = _as_xyz(place_pos, "demo place position")
    return [
        place_xyz[0] - object_xyz[0],
        place_xyz[1] - object_xyz[1],
        place_xyz[2] - object_xyz[2],
    ]


def compute_target_displacement_place_target(target_position_base,
                                             object_size=None,
                                             demo_entry=None,
                                             default_offset_xyz=None):
    """Ablation target that reuses the demo displacement from the live target.

    Unlike :func:`compute_anchor_place_target`, this baseline never uses the
    current anchor pose.  It isolates the contribution of live target-anchor
    conditioning when the two objects move independently.
    """
    target_xyz = _as_xyz(target_position_base, "target_position_base")
    if demo_entry:
        offset = demo_target_place_offset(
            demo_entry, default_offset=default_offset_xyz)
        source = "demo_target_relative_displacement"
    else:
        offset = [
            float(v) for v in (default_offset_xyz or [0.0, 0.0, 0.0])
        ]
        source = "target_relative_default"

    place_xyz = [
        target_xyz[0] + offset[0],
        target_xyz[1] + offset[1],
        target_xyz[2] + offset[2],
    ]
    return {
        "mode": "target_displacement_ablation",
        "anchor_name": "",
        "anchor_category": "",
        "anchor_xyz": None,
        "object_xyz": target_xyz,
        "object_size": object_size,
        "offset_xyz": [float(v) for v in offset],
        "place_xyz": [float(v) for v in place_xyz],
        "resolution_method": source,
        "reason": "ablation ignores the live anchor pose",
        "confidence": 0.0,
    }


def compute_anchor_place_target(anchor_position_base,
                                object_position_base=None,
                                object_size=None,
                                demo_entry=None,
                                anchor_profile=None,
                                override_offset_xyz=None):
    """Compute runtime place target from live anchor pose.

    Args:
        anchor_position_base: detected/live anchor center or placement surface
            reference in robot base frame.
        object_position_base: optional live target object pose; included only
            for logging/metadata.
        object_size: optional target object size.
        demo_entry: optional recorded anchored demo.  If provided, the demo's
            place-vs-anchor relative offset is replayed in the live scene.
        anchor_profile: geometry defaults for the selected anchor.
        override_offset_xyz: explicit [dx, dy, dz] relative to anchor; useful
            for calibration or ablation.
    """
    profile = copy.deepcopy(DEFAULT_ANCHOR_PROFILE)
    if anchor_profile:
        profile.update(anchor_profile)

    anchor_xyz = _as_xyz(anchor_position_base, "anchor_position_base")
    if override_offset_xyz is not None:
        offset = _as_xyz(override_offset_xyz, "override_offset_xyz")
        source = "override_offset"
    elif demo_entry:
        offset = demo_anchor_place_offset(
            demo_entry, default_offset=profile["default_place_offset_xyz"])
        source = "demo_anchor_relative_offset"
    else:
        offset = [float(v) for v in profile["default_place_offset_xyz"]]
        source = "profile_default"

    offset[2] += float(profile.get("surface_z_offset", 0.0))
    place_xyz = [
        anchor_xyz[0] + offset[0],
        anchor_xyz[1] + offset[1],
        anchor_xyz[2] + offset[2],
    ]

    return {
        "mode": "anchor",
        "anchor_name": profile.get("name", "anchor"),
        "anchor_category": profile.get("category", "unknown"),
        "anchor_xyz": anchor_xyz,
        "object_xyz": (
            None if object_position_base is None
            else _as_xyz(object_position_base, "object_position_base")
        ),
        "object_size": object_size,
        "offset_xyz": [float(v) for v in offset],
        "place_xyz": [float(v) for v in place_xyz],
        "resolution_method": source,
        "reason": "place target follows the detected anchor pose",
        "confidence": 0.9 if source != "profile_default" else 0.65,
    }


def top_grasp_from_object(object_position_base, object_size=None,
                          q_xyzw=None, flange_z_offset=0.050):
    """Simple top grasp used by the blue-tray baseline task."""
    obj = _as_xyz(object_position_base, "object_position_base")
    q = q_xyzw or [-1.0, 0.0, 0.0, 0.0]
    return {
        "position": [obj[0], obj[1], obj[2]],
        "orientation": [float(v) for v in q],
        "flange_z_offset": float(flange_z_offset),
        "object_size": object_size,
    }
