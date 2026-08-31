#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene-package export helpers for two-object relational tasks."""

import json
import os
import time

import numpy as np

from mt3_scene_package import save_scene_package


SCENE_PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo_library",
    "scene_packages")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()
                if k != "object_points"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_mask(mask_path):
    if not mask_path or not os.path.exists(mask_path):
        return None
    return np.load(mask_path).astype(bool)


def _entry_scene_data(entry):
    pose_source = (entry or {}).get("pose_source") or {}
    return {
        "rgb": (entry or {}).get("rgb"),
        "depth": (entry or {}).get("depth"),
        "segmap": _load_mask((entry or {}).get("mask_path")),
        "intrinsics": (entry or {}).get("intrinsics"),
        "pose": pose_source,
    }


def _package_one(scene, key, name, role, label, extra_metadata):
    entry = (scene or {}).get(key) or {}
    pose_source = entry.get("pose_source")
    if not pose_source:
        return None

    metadata = {
        "object_role": key,
        "object_label": label,
        "mask_path": entry.get("mask_path"),
        "position_base": entry.get("position_base"),
        "orientation_base": entry.get("orientation_base"),
        "estimated_size": entry.get("estimated_size"),
        "perception_method": entry.get("method"),
    }
    metadata.update(extra_metadata or {})

    return save_scene_package(
        _entry_scene_data(entry),
        SCENE_PACKAGE_DIR,
        name=name,
        role=role,
        extra_metadata=metadata)


def save_dual_object_scene_packages(scene, task_id, role,
                                    target_label="target",
                                    anchor_label="anchor",
                                    relation_kind="relational_task",
                                    extra_metadata=None):
    """Save target and anchor pointcloud packages plus a relation metadata file."""
    os.makedirs(SCENE_PACKAGE_DIR, exist_ok=True)
    safe_task_id = str(task_id).replace(os.sep, "_")
    prefix = "%s_%s" % (str(role), safe_task_id)

    target_pkg = _package_one(
        scene, "target", prefix + "_target", role + "_target",
        target_label, extra_metadata)
    anchor_pkg = _package_one(
        scene, "anchor", prefix + "_anchor", role + "_anchor",
        anchor_label, extra_metadata)

    target = (scene or {}).get("target") or {}
    anchor = (scene or {}).get("anchor") or {}
    target_xyz = target.get("position_base")
    anchor_xyz = anchor.get("position_base")
    offset = None
    if target_xyz is not None and anchor_xyz is not None:
        offset = [
            float(target_xyz[0]) - float(anchor_xyz[0]),
            float(target_xyz[1]) - float(anchor_xyz[1]),
            float(target_xyz[2]) - float(anchor_xyz[2]),
        ]

    relation = {
        "format": "mt3_dual_object_scene_relation_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_id": safe_task_id,
        "role": role,
        "relation_kind": relation_kind,
        "target_label": target_label,
        "anchor_label": anchor_label,
        "target_package": (target_pkg or {}).get("name"),
        "anchor_package": (anchor_pkg or {}).get("name"),
        "target_position_base": _jsonable(target_xyz),
        "anchor_position_base": _jsonable(anchor_xyz),
        "target_minus_anchor_xyz": offset,
        "target_mask_path": (scene or {}).get("target_mask_path"),
        "anchor_mask_path": (scene or {}).get("anchor_mask_path"),
        "extra_metadata": _jsonable(extra_metadata or {}),
    }
    relation_path = os.path.join(
        SCENE_PACKAGE_DIR, prefix + "_relation.json")
    with open(relation_path, "w", encoding="utf-8") as f:
        json.dump(relation, f, indent=2, ensure_ascii=False)
    relation["relation_path"] = relation_path

    return {
        "target_package": target_pkg,
        "anchor_package": anchor_pkg,
        "relation": relation,
    }
