#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command MT3-style generalization entry point.

The task profiles here only expand concise experiment commands into the
existing task-specific pipelines.  The lower-level grasp/place/insert code
remains the execution authority.
"""

import os
import subprocess
import sys


CODE_DIR = os.path.dirname(os.path.abspath(__file__))


TASK_ALIASES = {
    "top": "top_grasp",
    "top_grasp": "top_grasp",
    "cube_top": "top_grasp",
    "cube_top_grasp": "top_grasp",
    "grasp": "top_grasp",

    "rot": "rotated_top_grasp",
    "rotated": "rotated_top_grasp",
    "rotated_grasp": "rotated_top_grasp",
    "rotated_top": "rotated_top_grasp",
    "rotated_top_grasp": "rotated_top_grasp",
    "cuboid_top": "rotated_top_grasp",
    "cuboid_yaw": "rotated_top_grasp",
    "yaw_grasp": "rotated_top_grasp",

    "place": "directional_place",
    "pick_place": "directional_place",
    "directional_place": "directional_place",
    "place_directional": "directional_place",

    "anchor": "anchor_place",
    "anchor_place": "anchor_place",
    "anchor_pick_place": "anchor_place",
    "tray": "anchor_place",
    "plate": "anchor_place",

    "insert": "vertical_insert",
    "insertion": "vertical_insert",
    "vertical_insert": "vertical_insert",
    "vertical_insertion": "vertical_insert",
    "socket": "vertical_insert",
}


PROFILES = {
    "top_grasp": {
        "pipeline": "mt3_pipeline_top_lift.py",
        "defaults": {
            "query": "pick up the green cube",
            "use_perception": "true",
            "use_pointcloud_pose": "true",
            "run_icp": "false",
            "use_icp_object_pose": "false",
            "dry_run": "false",
            "use_demo_replay": "true",
            "use_top_grasp_replay": "true",
            "prefer_pose_replay": "false",
            "use_segmented_replay": "true",
            "close_on_replay_blocked": "true",
            "replay_close_on_blocked_min_progress": "0.35",
            "object_shape": "cube",
            "object_label": "green_cube",
            "object_size": "[0.045, 0.045, 0.045]",
            "gazebo_model_name": "grasp_object",
            "method_variant": "top_lift_after_close",
            "experiment_group": "top_grasp",
        },
    },
    "rotated_top_grasp": {
        "pipeline": "mt3_pipeline.py",
        "defaults": {
            "query": "pick up the green cuboid with rotated top grasp",
            "use_perception": "true",
            "use_pointcloud_pose": "true",
            "use_icp_object_pose": "true",
            "dry_run": "false",
            "use_demo_replay": "true",
            "use_top_grasp_replay": "true",
            "object_shape": "rectangular_prism",
            "object_label": "green_cuboid",
            "gazebo_model_name": "green_rectangular_prism",
            "object_long_axis_local": "y",
            "object_size": "[0.03, 0.10, 0.035]",
            "method_variant": "full",
            "experiment_group": "rotated_top_grasp",
        },
    },
    "directional_place": {
        "pipeline": "mt3_pipeline.py",
        "defaults": {
            "query": "pick up the green cube and place it to the right",
            "use_perception": "true",
            "use_pointcloud_pose": "true",
            "use_icp_object_pose": "true",
            "dry_run": "false",
            "use_demo_replay": "true",
            "object_shape": "cube",
            "object_label": "green_cube",
            "object_size": "[0.045, 0.045, 0.045]",
            "method_variant": "full",
            "experiment_group": "directional_place",
        },
    },
    "anchor_place": {
        "pipeline": "mt3_anchor_place_pipeline.py",
        "defaults": {
            "query": "pick the green cube and place it on the blue tray",
            "use_perception": "true",
            "dry_run": "false",
            "use_demo_replay": "true",
            "object_shape": "cube",
            "target_label": "green_cube",
            "object_size": "[0.045, 0.045, 0.045]",
            "anchor_name": "blue_anchor_tray",
            "anchor_category": "tray",
            "method_variant": "full",
            "experiment_group": "anchor_place",
            "condition_id": "anchor_x060_y-018_target_x060_y000",
        },
    },
    "vertical_insert": {
        "pipeline": "mt3_cylinder_insert_pipeline.py",
        "defaults": {
            "query": "insert the green cylinder into the blue socket",
            "use_perception": "true",
            "dry_run": "false",
            "use_demo_replay": "true",
            "object_shape": "cylinder",
            "target_label": "green_cylinder",
            "object_size": "[0.045, 0.045, 0.100]",
            "socket_name": "blue_insert_socket",
            "socket_category": "shallow_circular_socket",
            "socket_size": "[0.085, 0.085, 0.100]",
            "socket_opening": "[0.055, 0.055]",
            "method_variant": "full",
            "experiment_group": "vertical_insert",
            "condition_id": "insert_socket_x060_y-018_target_x060_y000",
        },
    },
}


TOP_GRASP_SHAPE_DEFAULTS = {
    "cube": {
        "query": "pick up the green cube",
        "object_label": "green_cube",
        "object_size": "[0.045, 0.045, 0.045]",
        "gazebo_model_name": "grasp_object",
    },
    "sphere": {
        "query": "pick up the green sphere",
        "object_label": "green_sphere",
        "object_size": "[0.050, 0.050, 0.050]",
        "gazebo_model_name": "green_sphere",
    },
    "ball": {
        "query": "pick up the green sphere",
        "object_label": "green_sphere",
        "object_shape": "sphere",
        "object_size": "[0.050, 0.050, 0.050]",
        "gazebo_model_name": "green_sphere",
    },
    "cylinder": {
        "query": "pick up the green cylinder",
        "object_label": "green_cylinder",
        "object_size": "[0.040, 0.040, 0.075]",
        "gazebo_model_name": "green_short_cylinder",
    },
    "small_cylinder": {
        "query": "pick up the green cylinder",
        "object_label": "green_cylinder",
        "object_shape": "cylinder",
        "object_size": "[0.040, 0.040, 0.075]",
        "gazebo_model_name": "green_short_cylinder",
    },
    "rectangular_prism": {
        "query": "pick up the green cuboid",
        "object_label": "green_cuboid",
        "object_size": "[0.03, 0.10, 0.035]",
        "gazebo_model_name": "green_rectangular_prism",
        "object_long_axis_local": "y",
    },
    "cuboid": {
        "query": "pick up the green cuboid",
        "object_label": "green_cuboid",
        "object_shape": "rectangular_prism",
        "object_size": "[0.03, 0.10, 0.035]",
        "gazebo_model_name": "green_rectangular_prism",
        "object_long_axis_local": "y",
    },
}


VARIANTS = {
    "full": {},
    "replay": {"method_variant": "replay"},
    "no_replay": {
        "use_demo_replay": "false",
        "use_top_grasp_replay": "false",
        "method_variant": "no_replay",
    },
    "no_stage_replay": {
        "use_demo_replay": "false",
        "use_top_grasp_replay": "false",
        "method_variant": "no_stage_replay",
    },
    "no_relation": {
        "relation_alignment_mode": "target_displacement",
        "method_variant": "no_relation",
    },
    "no_relation_no_stage_replay": {
        "relation_alignment_mode": "target_displacement",
        "use_demo_replay": "false",
        "use_top_grasp_replay": "false",
        "method_variant": "no_relation_no_stage_replay",
    },
    "no_icp": {
        "run_icp": "false",
        "use_icp_object_pose": "false",
        "method_variant": "no_icp",
    },
    "no_perception": {
        "use_perception": "false",
        "method_variant": "no_perception",
    },
    "scripted": {
        "use_demo_replay": "false",
        "use_top_grasp_replay": "false",
        "run_icp": "false",
        "use_icp_object_pose": "false",
        "method_variant": "scripted",
    },
}


CONTROL_ARGS = set([
    "task",
    "variant",
    "pipeline",
    "print_only",
    "direction",
    "yaw_deg",
])


def _arg_name(arg):
    if not arg.startswith("_") or ":=" not in arg:
        return ""
    return arg[1:].split(":=", 1)[0]


def _ros_arg(name, default=""):
    prefix = "_%s:=" % name
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _has_arg(name):
    return any(_arg_name(arg) == name for arg in sys.argv[1:])


def _task_name():
    raw = (_ros_arg("task", "") or "").strip().lower().replace("-", "_")
    return TASK_ALIASES.get(raw, raw)


def _variant_name():
    return (_ros_arg("variant", _ros_arg("method_variant", "full")) or "full").strip().lower()


def _text():
    return " ".join([
        _ros_arg("query", ""),
        _ros_arg("task_type", ""),
        _ros_arg("task", ""),
    ]).lower()


def _contains_any(text, words):
    return any(word in text for word in words)


def _yaw_label():
    yaw = _ros_arg("yaw_deg", "").strip()
    if not yaw:
        return "current"
    try:
        value = int(round(float(yaw)))
        return ("%+03d" % value).replace("+", "")
    except Exception:
        return yaw.replace(".", "p").replace("-", "m")


def _cm_label(prefix, value):
    try:
        cm = int(round(float(value) * 100.0))
        sign = "-" if cm < 0 else ""
        return "%s%s%03d" % (prefix, sign, abs(cm))
    except Exception:
        return "%s%s" % (
            prefix,
            str(value).replace(".", "p").replace("-", "m"))


def _shape_label(shape):
    shape = str(shape or "unknown").strip().lower()
    return {
        "rectangular_prism": "rect",
        "cuboid": "rect",
        "small_cylinder": "cyl",
        "cylinder": "cyl",
        "sphere": "sphere",
        "ball": "sphere",
        "cube": "cube",
    }.get(shape, shape.replace(" ", "_"))


def _direction():
    return (_ros_arg("direction", "") or _ros_arg("place_direction", "") or "right").strip().lower()


def _profile_defaults(task):
    profile = dict((PROFILES.get(task) or {}).get("defaults", {}))

    if task == "top_grasp":
        shape = (_ros_arg("object_shape", profile.get("object_shape", "cube"))
                 or "cube").strip().lower()
        shape_defaults = dict(TOP_GRASP_SHAPE_DEFAULTS.get(shape, {}))
        if shape_defaults:
            shape = str(shape_defaults.get("object_shape", shape))
            profile.update(shape_defaults)
        profile["object_shape"] = shape
        x_value = _ros_arg("x", "0.60").strip() or "0.60"
        y_value = _ros_arg("y", "0.00").strip() or "0.00"
        profile.setdefault(
            "condition_id",
            "top_%s_%s_%s" % (
                _cm_label("x", x_value),
                _cm_label("y", y_value),
                _shape_label(shape)))

    if task == "rotated_top_grasp":
        profile.setdefault("condition_id", "rot_yaw%s_x060_y000" % _yaw_label())

    if task == "directional_place":
        direction = _direction()
        profile["query"] = "pick up the green cube and place it to the %s" % direction
        profile.setdefault("condition_id", "place_%s_x060_y000" % direction)

    return profile


def _selected_profile():
    task = _task_name()
    if task in PROFILES:
        return task, PROFILES[task]
    return "", None


def _select_pipeline(task="", profile=None):
    override = _ros_arg("pipeline", "").strip().lower()
    if override in ("mt3", "grasp", "place", "directional_place"):
        return "mt3_pipeline.py"
    if override in ("anchor", "anchor_place", "tray", "plate"):
        return "mt3_anchor_place_pipeline.py"
    if override in ("insert", "insertion", "socket", "vertical_insert"):
        return "mt3_cylinder_insert_pipeline.py"

    if profile:
        return profile["pipeline"]

    text = _text()
    if _contains_any(text, [
            "insert", "insertion", "socket", "hole", "peg",
            "插入", "孔", "圆孔", "套筒"]):
        return "mt3_cylinder_insert_pipeline.py"

    if _contains_any(text, [
            "tray", "plate", "anchor", "on the blue", "into the plate",
            "托盘", "盘", "盘子", "锚点"]):
        return "mt3_anchor_place_pipeline.py"

    return "mt3_pipeline.py"


def _expanded_args(task, profile):
    user_names = set(_arg_name(arg) for arg in sys.argv[1:] if _arg_name(arg))
    defaults = {}
    if profile:
        defaults.update(_profile_defaults(task))

    variant = _variant_name()
    defaults.update(VARIANTS.get(variant, {}))
    if variant not in VARIANTS and not _has_arg("method_variant"):
        defaults["method_variant"] = variant

    generated = []
    for name, value in defaults.items():
        if name not in user_names:
            generated.append("_%s:=%s" % (name, value))

    passthrough = [
        arg for arg in sys.argv[1:]
        if _arg_name(arg) not in CONTROL_ARGS
    ]
    return generated + passthrough


def _print_tasks():
    print("Available _task profiles:")
    for name in [
            "top_grasp", "rotated_top_grasp", "directional_place",
            "anchor_place", "vertical_insert"]:
        print("  -", name)


def main():
    if _ros_arg("task", "").strip().lower() in ("list", "help", "?"):
        _print_tasks()
        return 0

    task, profile = _selected_profile()
    script_name = _select_pipeline(task=task, profile=profile)
    script = os.path.join(CODE_DIR, script_name)
    if not os.path.exists(script):
        raise RuntimeError("pipeline script not found: %s" % script)

    child_args = _expanded_args(task, profile)
    cmd = [sys.executable, script] + child_args

    if task:
        print("[mt3_generalize] task:", task)
    print("[mt3_generalize] selected:", os.path.basename(script))
    print("[mt3_generalize] command:", " ".join(cmd))

    if _ros_arg("print_only", "false").lower() in ("1", "true", "yes"):
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
