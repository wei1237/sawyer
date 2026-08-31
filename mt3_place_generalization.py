#!/usr/bin/env python3
"""Deterministic place-target generalization for MT3 pick-place rollouts."""

import re


PLACE_DIRECTION_OFFSETS = {
    "left": (0.0, 0.18),
    "right": (0.0, -0.18),
    "front": (0.15, 0.0),
    "back": (-0.15, 0.0),
}

PLACE_DIRECTION_KEYWORDS = {
    "left": ("left", "左", "左边", "左侧", "左手边"),
    "right": ("right", "右", "右边", "右侧", "右手边"),
    "front": ("front", "forward", "前", "前面", "前方", "正前方"),
    "back": ("back", "behind", "后", "后面", "后方", "背后"),
}

COMBINED_DIRECTION_OFFSETS = {
    "右前": (0.10, -0.12),
    "前右": (0.10, -0.12),
    "右后": (-0.10, -0.12),
    "后右": (-0.10, -0.12),
    "左前": (0.10, 0.12),
    "前左": (0.10, 0.12),
    "左后": (-0.10, 0.12),
    "后左": (-0.10, 0.12),
}


def _normalize(text):
    return (text or "").strip().lower()


def parse_place_direction(query_text, default_direction="right"):
    """Return a direction/custom-offset dict from a language command."""
    text = _normalize(query_text)

    for key, offset in COMBINED_DIRECTION_OFFSETS.items():
        if key in text:
            return {
                "mode": "custom_offset",
                "direction": None,
                "offset_xy": [float(offset[0]), float(offset[1])],
                "confidence": 0.85,
                "method": "keyword_fallback",
                "reason": "combined direction keyword: %s" % key,
            }

    matched = []
    for direction, keywords in PLACE_DIRECTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.append(direction)

    if len(matched) >= 2:
        dx = sum(PLACE_DIRECTION_OFFSETS[d][0] for d in matched)
        dy = sum(PLACE_DIRECTION_OFFSETS[d][1] for d in matched)
        return {
            "mode": "custom_offset",
            "direction": None,
            "offset_xy": [float(dx), float(dy)],
            "confidence": 0.8,
            "method": "keyword_fallback",
            "reason": "combined: %s" % "+".join(matched),
        }

    if matched:
        direction = matched[0]
        dx, dy = PLACE_DIRECTION_OFFSETS[direction]
        return {
            "mode": "direction",
            "direction": direction,
            "offset_xy": [float(dx), float(dy)],
            "confidence": 0.9,
            "method": "keyword_fallback",
            "reason": "keyword matched: %s" % direction,
        }

    direction = default_direction if default_direction in PLACE_DIRECTION_OFFSETS else "right"
    dx, dy = PLACE_DIRECTION_OFFSETS[direction]
    return {
        "mode": "direction",
        "direction": direction,
        "offset_xy": [float(dx), float(dy)],
        "confidence": 0.3,
        "method": "keyword_fallback",
        "reason": "no direction keyword found, defaulting to %s" % direction,
    }


def clamp_place_offset(offset_xy, min_radius=0.05, max_radius=0.30):
    """Clamp custom XY offset to a reasonable tabletop radius."""
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    radius = (dx * dx + dy * dy) ** 0.5
    if radius <= 1e-9:
        return [dx, dy]
    if radius < min_radius:
        scale = min_radius / radius
    elif radius > max_radius:
        scale = max_radius / radius
    else:
        scale = 1.0
    return [dx * scale, dy * scale]


def compute_place_target(object_position, object_size, direction_info,
                         surface_z_offset=0.0):
    """Compute runtime place params from live object pose and direction.

    object_position is the live object base/table-contact pose in robot base
    frame. place_z intentionally remains the placement surface z; the Sawyer
    place executor adds gripper flange and clearance internally.
    """
    if len(object_position) < 3:
        raise ValueError("object_position must contain [x, y, z]")

    ox, oy, oz = [float(v) for v in object_position[:3]]
    size = [float(v) for v in (object_size or [0.045, 0.045, 0.045])]
    if len(size) != 3:
        size = [0.045, 0.045, 0.045]

    offset_xy = direction_info.get("offset_xy", [0.0, -0.18])
    offset_xy = clamp_place_offset(offset_xy)
    place_x = ox + offset_xy[0]
    place_y = oy + offset_xy[1]
    place_z = oz + float(surface_z_offset)
    mode = direction_info.get("mode", "direction")
    direction = direction_info.get("direction")
    label = "custom" if mode == "custom_offset" else (direction or "right")

    return {
        "mode": mode,
        "direction": direction,
        "label": label,
        "offset_xy": [float(offset_xy[0]), float(offset_xy[1])],
        "place_xyz": [float(place_x), float(place_y), float(place_z)],
        "object_xyz": [ox, oy, oz],
        "object_size": size,
        "surface_z_offset": float(surface_z_offset),
        "resolution_method": direction_info.get("method", "unknown"),
        "confidence": float(direction_info.get("confidence", 0.0)),
        "reason": direction_info.get("reason", ""),
    }


def parse_numeric_xy_offset(text):
    """Optional helper for commands containing explicit dx/dy meters."""
    text = _normalize(text)
    match = re.search(r"dx\s*=\s*(-?\d+(?:\.\d+)?)\s*,?\s*dy\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))]
