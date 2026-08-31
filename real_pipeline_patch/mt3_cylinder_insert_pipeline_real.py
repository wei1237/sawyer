#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-Sawyer wrapper for mt3_cylinder_insert_pipeline.py.

The original simulation insertion pipeline is not modified.  This wrapper
injects the ASC60C registered-depth two-object perception, disables Gazebo GT,
uses a separate real executor path, and defaults to dry-run.
"""

import os
import sys

import rospy

import mt3_cylinder_insert_pipeline as _sim
from mt3_anchor_perception_real import DualMaskAnchorPerception as RealDualMaskAnchorPerception


_ORIG_LOAD_SCENE = _sim._load_scene
_ORIG_RUN_EXECUTOR = _sim._run_executor


def _global(name):
    return "/sawyer_auto_grasp/%s" % str(name).lstrip("~/")


def _param(name, default=None):
    private = "~%s" % str(name).lstrip("~/")
    if rospy.has_param(private):
        return rospy.get_param(private)
    return rospy.get_param(_global(name), default)


def _bool(name, default=False):
    value = _param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _optional_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "success", "pass"):
        return True
    if text in ("0", "false", "no", "off", "failed", "fail"):
        return False
    return None


def _seed_private(name, default=None):
    private = "~%s" % name
    if rospy.has_param(private):
        return rospy.get_param(private)
    value = rospy.get_param(_global(name), default)
    if value is not None:
        rospy.set_param(private, value)
    return value


def _real_load_scene(demo=None):
    """Preflight after rospy.init_node(), then call shared scene logic."""
    env = str(_param("execution_environment", "")).strip().lower()
    if env != "real":
        raise RuntimeError(
            "mt3_cylinder_insert_pipeline_real.py requires "
            "/sawyer_auto_grasp/execution_environment=real")

    # Real wrapper defaults.  Private launch args still override these.
    _seed_private("use_perception", True)
    _seed_private("dry_run", True)
    _seed_private("target_mask_path", _param(
        "langsam_mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy"))
    if not rospy.has_param("~socket_mask_path"):
        rospy.set_param(
            "~socket_mask_path",
            _param("anchor_mask_path", "/mnt/hgfs2/tmp_vision/current_anchor_mask.npy"))

    if not _bool("use_perception", True):
        if not _bool("allow_manual_scene", False):
            raise RuntimeError(
                "Real insertion requires perception. To use manual coordinates, set "
                "allow_manual_scene=true AND explicitly provide target_x/y/z and socket_x/y/z.")
        required = (
            "~target_x", "~target_y", "~target_z",
            "~socket_x", "~socket_y", "~socket_z",
        )
        missing = [name for name in required if not rospy.has_param(name)]
        if missing:
            raise RuntimeError(
                "Manual real scene requested but parameters are missing: %s" %
                ", ".join(missing))

    rospy.loginfo(
        "[InsertReal] ASC60C perception active; dry_run=%s",
        _bool("dry_run", True))
    return _ORIG_LOAD_SCENE(demo)


def _real_executor_path():
    configured = _param("place_executor_path", "")
    if configured:
        return os.path.expanduser(str(configured))
    return os.path.expanduser(
        "~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_place_real.py")


def _real_run_executor():
    if not _bool("allow_real_execution", False):
        raise RuntimeError(
            "Real insertion execution blocked: allow_real_execution=false. "
            "Keep dry_run=true until mt3_sawyer_place_real.py is verified.")
    script = _real_executor_path()
    if not os.path.isfile(script):
        raise RuntimeError("Real Sawyer place executor not found: %s" % script)
    rospy.logwarn("REAL ROBOT EXECUTION: launching %s", script)
    return _ORIG_RUN_EXECUTOR()


def _real_gazebo_pose(model_param, fallback_keywords):
    """No Gazebo ground truth on the real robot."""
    return None


def _real_post_insert_success(insert_result, cylinder_size, socket_profile,
                              initial_socket_xyz):
    """Explicit real postcheck policy; never fake Gazebo validation."""
    mode = str(_param("real_postcheck_mode", "executor_only")).strip().lower()
    if mode in ("manual", "manual_param"):
        parsed = _optional_bool(_param("manual_success_label", ""))
        if parsed is None:
            rospy.logwarn(
                "[InsertReal] manual postcheck requested but manual_success_label is unset; "
                "keeping executor result and recording postcheck as unknown.")
            return {
                "ok": True,
                "failure_stage": "",
                "failure_reason": "",
                "postcheck_success": "",
                "postcheck_reason": "manual_success_label_unset",
                "final_object_model_name": "",
                "final_socket_model_name": "",
                "final_object_xyz": None,
                "final_socket_xyz": None,
                "final_target_error_xy_m": "",
                "final_relation_error_xy_m": "",
                "insert_depth_m": "",
            }
        return {
            "ok": bool(parsed),
            "failure_stage": "" if parsed else "insertion_verification",
            "failure_reason": "" if parsed else "manual_success_label_false",
            "postcheck_success": bool(parsed),
            "postcheck_reason": "manual_success_label",
            "final_object_model_name": "",
            "final_socket_model_name": "",
            "final_object_xyz": None,
            "final_socket_xyz": None,
            "final_target_error_xy_m": "",
            "final_relation_error_xy_m": "",
            "insert_depth_m": "",
        }

    rospy.logwarn(
        "[InsertReal] no Gazebo GT/post-insertion vision verifier; "
        "keeping executor result and recording postcheck as unknown.")
    return {
        "ok": True,
        "failure_stage": "",
        "failure_reason": "",
        "postcheck_success": "",
        "postcheck_reason": "real_executor_only_no_gazebo_gt",
        "final_object_model_name": "",
        "final_socket_model_name": "",
        "final_object_xyz": None,
        "final_socket_xyz": None,
        "final_target_error_xy_m": "",
        "final_relation_error_xy_m": "",
        "insert_depth_m": "",
    }


# Process-local dependency injection.  No simulation source file is edited.
_sim.DualMaskAnchorPerception = RealDualMaskAnchorPerception
_sim._load_scene = _real_load_scene
_sim._executor_path = _real_executor_path
_sim._run_executor = _real_run_executor
_sim._gazebo_pose = _real_gazebo_pose
_sim._validate_post_insert_success = _real_post_insert_success


if __name__ == "__main__":
    try:
        ok = _sim.main()
        sys.exit(0 if ok else 1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Real cylinder insertion pipeline interrupted")
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("mt3_cylinder_insert_pipeline_real failed: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)
