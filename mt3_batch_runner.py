#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner for MT3 Gazebo experiments.

Run this on Ubuntu/ROS side after Gazebo, controllers, camera relay, MoveIt,
and the Windows LangSAM mask worker are running.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion
from intera_interface import Gripper, Limb
from sensor_msgs.msg import Image


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SAFE_JOINTS = {
    "right_j0": 0.0,
    "right_j1": -0.8,
    "right_j2": 0.0,
    "right_j3": 1.8,
    "right_j4": 0.0,
    "right_j5": 0.0,
    "right_j6": 0.0,
}


def yaw_to_quaternion(yaw_rad):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_object_size(value, default_height=0.045):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return [float(v) for v in value]
    except Exception:
        return [0.045, 0.045, float(default_height)]


def resolve_spawn_pose(trial, config):
    pose = dict(trial.get("pose", {}))
    if "x" not in pose or "y" not in pose:
        raise RuntimeError("trial pose must include x and y: {}".format(
            trial.get("id", "")))

    z_value = pose.get("z", "auto")
    if z_value in ("", None, "auto"):
        object_size = parse_object_size(
            trial.get("object_size", config.get("default_object_size", "")))
        workbench_top_z = float(config.get("workbench_top_z", 0.325))
        spawn_z_clearance = float(config.get("spawn_z_clearance", 0.001))
        pose["z"] = workbench_top_z + float(object_size[2]) * 0.5 + spawn_z_clearance
        pose["z_source"] = "auto_from_object_size"
    else:
        pose["z"] = float(z_value)
        pose["z_source"] = "explicit"
    return pose


def set_gazebo_model_pose(model_name, pose):
    rospy.wait_for_service("/gazebo/set_model_state", timeout=20)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    state = ModelState()
    state.model_name = model_name
    state.reference_frame = "world"
    state.pose.position.x = float(pose["x"])
    state.pose.position.y = float(pose["y"])
    state.pose.position.z = float(pose["z"])
    yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
    state.pose.orientation = yaw_to_quaternion(yaw)
    resp = set_state(state)
    if not resp.success:
        raise RuntimeError("Failed to set Gazebo model '{}': {}".format(
            model_name, resp.status_message))


def delete_gazebo_model(model_name):
    rospy.wait_for_service("/gazebo/delete_model", timeout=20)
    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    try:
        delete_model(model_name)
        rospy.sleep(0.3)
    except Exception:
        pass


def cleanup_gazebo_objects(model_names):
    for name in model_names:
        if name:
            delete_gazebo_model(name)


def reset_sawyer_arm(limb, gripper=None, wait_sec=1.0):
    """Return Sawyer to the fixed observation/start pose used by demo recording."""
    rospy.loginfo("[Batch] Resetting Sawyer arm to safe start pose...")
    try:
        if gripper is not None:
            gripper.open()
            rospy.sleep(0.5)
    except Exception as exc:
        rospy.logwarn("[Batch] Gripper open during reset failed: %s", exc)

    try:
        limb.set_joint_position_speed(0.25)
    except Exception:
        pass
    limb.move_to_joint_positions(SAFE_JOINTS, timeout=15.0)
    rospy.sleep(wait_sec)
    rospy.loginfo("[Batch] Sawyer reset complete.")


def spawn_gazebo_sdf_model(model_name, sdf_path, pose):
    if not sdf_path:
        set_gazebo_model_pose(model_name, pose)
        return
    if not os.path.exists(sdf_path):
        raise RuntimeError("SDF file not found: {}".format(sdf_path))

    rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=20)
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_gazebo_model(model_name)

    with open(sdf_path, "r", encoding="utf-8") as f:
        model_xml = f.read()

    model_pose = ModelState().pose
    model_pose.position.x = float(pose["x"])
    model_pose.position.y = float(pose["y"])
    model_pose.position.z = float(pose["z"])
    yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
    model_pose.orientation = yaw_to_quaternion(yaw)

    resp = spawn_model(model_name, model_xml, "", model_pose, "world")
    if not resp.success:
        raise RuntimeError("Failed to spawn '{}': {}".format(
            model_name, resp.status_message))


def save_rgb(topic, out_path, timeout=10):
    bridge = CvBridge()
    msg = rospy.wait_for_message(topic, Image, timeout=timeout)
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    ensure_dir(os.path.dirname(out_path))
    if not cv2.imwrite(out_path, img):
        raise RuntimeError("Failed to write RGB image: {}".format(out_path))
    return out_path


def wait_for_mask(mask_path, timeout_sec):
    start = time.time()
    while time.time() - start < timeout_sec and not rospy.is_shutdown():
        if os.path.exists(mask_path):
            return True
        time.sleep(1.0)
    return False


def run_pipeline(trial, trial_dir, config):
    query = trial.get("query", config.get("default_query", "抓取"))
    object_shape = trial.get("object_shape", "unknown")
    object_label = trial.get("object_label", object_shape)
    mask_path = os.path.join(trial_dir, "mask.npy")
    use_pointcloud = bool(trial.get(
        "use_pointcloud_pose", config.get("default_use_pointcloud_pose", True)))
    use_replay = bool(trial.get(
        "use_demo_replay", config.get("default_use_demo_replay", False)))
    use_icp_pose = bool(trial.get(
        "use_icp_object_pose", config.get("default_use_icp_object_pose", True)))
    auto_record_success = bool(trial.get(
        "auto_record_success", config.get("default_auto_record_success", True)))
    pose = trial.get("_resolved_pose", trial.get("pose", {}))

    note = trial.get("trial_note", "")
    if not note:
        note = "batch={} trial={} gazebo_pose_x{}_y{}_z{}_yaw{}".format(
            config.get("batch_name", "mt3_batch"),
            trial.get("id", "trial"),
            pose.get("x", ""),
            pose.get("y", ""),
            pose.get("z", ""),
            pose.get("yaw_deg", 0))

    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, "mt3_pipeline.py"),
        "_query:={}".format(query),
        "_use_pointcloud_pose:={}".format(str(use_pointcloud).lower()),
        "_langsam_mask_path:={}".format(mask_path),
        "_use_icp_object_pose:={}".format(str(use_icp_pose).lower()),
        "_use_demo_replay:={}".format(str(use_replay).lower()),
        "_auto_record_success:={}".format(str(auto_record_success).lower()),
        "_object_shape:={}".format(object_shape),
        "_object_label:={}".format(object_label),
        "_trial_note:={}".format(note),
        "_experiment_log_dir:={}".format(
            config.get(
                "experiment_log_dir",
                os.path.join(PROJECT_DIR, "demo_library", "experiment_logs"))),
    ]

    optional_args = {
        "experiment_group": trial.get(
            "experiment_group", config.get("experiment_group", "")),
        "condition_id": trial.get("condition_id", ""),
        "repeat_id": trial.get("repeat_id", ""),
        "method_variant": trial.get(
            "method_variant", config.get("method_variant", "")),
        "gazebo_model_name": trial.get(
            "gazebo_model_name", trial.get(
                "model_name", config.get("default_model_name", ""))),
        "object_size": trial.get("object_size", config.get("default_object_size", "")),
        "x": pose.get("x", ""),
        "y": pose.get("y", ""),
        "z": pose.get("z", ""),
        "yaw_deg": pose.get("yaw_deg", ""),
    }
    for name, value in optional_args.items():
        if value != "" and value is not None:
            cmd.append("_{}:={}".format(name, value))

    log_path = os.path.join(trial_dir, "mt3_pipeline.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait(), log_path


def prepare_trial_manifest(trial, trial_dir, config):
    manifest = {
        "batch_name": config.get("batch_name", ""),
        "trial": trial,
        "trial_dir": trial_dir,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "rgb_saved_waiting_for_mask",
    }
    path = os.path.join(trial_dir, "trial.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def update_manifest(trial_dir, **updates):
    path = os.path.join(trial_dir, "trial.json")
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.update(updates)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        PROJECT_DIR, "experiments", "mt3_batch_example.json"))
    parser.add_argument("--rgb-topic", default="/io/internal_camera/head_camera/image_raw")
    parser.add_argument("--mask-timeout", type=float, default=240.0)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--only-save-rgb", action="store_true",
                        help="Only set poses and save RGBs; do not wait for masks or run MT3.")
    args = parser.parse_args()

    rospy.init_node("mt3_batch_runner", anonymous=True)
    limb = Limb("right")
    gripper = None
    try:
        gripper = Gripper("right")
    except Exception as exc:
        rospy.logwarn("Could not initialize gripper for batch reset: %s", exc)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    root = ensure_dir(config.get("shared_trial_root", "/mnt/hgfs2/tmp_vision/mt3_batch"))
    batch_name = config.get("batch_name", "mt3_batch")
    batch_dir = ensure_dir(os.path.join(root, batch_name))
    latest_dir = os.path.join(root, "latest")
    if os.path.isdir(latest_dir):
        shutil.rmtree(latest_dir)
    ensure_dir(latest_dir)

    trials = [t for t in config.get("trials", []) if t.get("enabled", True)]
    print("Batch '{}' enabled trials: {}".format(batch_name, len(trials)))
    print("Shared batch dir: {}".format(batch_dir))
    cleanup_model_names = config.get("cleanup_model_names", [
        "grasp_object", "green_cylinder", "green_rectangular_prism",
        "green_sphere", "simple_shoe"])

    for index, trial in enumerate(trials, 1):
        trial_id = trial.get("id", "trial_{:03d}".format(index))
        trial_dir = ensure_dir(os.path.join(batch_dir, "{:03d}_{}".format(index, trial_id)))
        model_name = trial.get("model_name", config.get("default_model_name", "green_cube"))
        sdf_path = trial.get("sdf_path", config.get("default_sdf_path", ""))
        pose = resolve_spawn_pose(trial, config)
        trial["_resolved_pose"] = pose
        print("\n========== Trial {}/{}: {} ==========".format(index, len(trials), trial_id))
        print("model={} sdf={} pose={}".format(model_name, sdf_path, pose))

        prepare_trial_manifest(trial, trial_dir, config)
        cleanup_gazebo_objects(cleanup_model_names)
        do_reset = bool(trial.get("reset_arm", config.get("default_reset_arm", True)))
        if do_reset:
            update_manifest(trial_dir, status="resetting_arm")
            reset_sawyer_arm(limb, gripper=gripper)
        spawn_gazebo_sdf_model(model_name, sdf_path, pose)
        time.sleep(args.settle_sec)

        rgb_path = save_rgb(args.rgb_topic, os.path.join(trial_dir, "rgb.png"))
        print("RGB saved:", rgb_path)
        update_manifest(trial_dir, status="rgb_saved_waiting_for_mask", rgb_path=rgb_path)

        # Keep a convenient latest pointer for quick manual inspection.
        shutil.copy2(rgb_path, os.path.join(latest_dir, "rgb.png"))
        shutil.copy2(os.path.join(trial_dir, "trial.json"), os.path.join(latest_dir, "trial.json"))

        if args.only_save_rgb:
            continue

        mask_path = os.path.join(trial_dir, "mask.npy")
        print("Waiting for LangSAM mask:", mask_path)
        if not wait_for_mask(mask_path, args.mask_timeout):
            update_manifest(trial_dir, status="failed_mask_timeout")
            print("Mask timeout, skipped:", trial_id)
            continue

        update_manifest(trial_dir, status="mask_ready_running_mt3", mask_path=mask_path)
        timeout_sec = float(trial.get("timeout_sec", config.get("default_timeout_sec", 180)))
        rc, log_path = run_pipeline(trial, trial_dir, config)
        status = "mt3_finished" if rc == 0 else "mt3_failed"
        update_manifest(trial_dir, status=status, mt3_return_code=rc, mt3_log_path=log_path)
        print("Trial {} done rc={} log={}".format(trial_id, rc, log_path))

    print("\nBatch done. MT3 CSV:")
    print(os.path.join(PROJECT_DIR, "demo_library", "experiment_logs", "mt3_trials.csv"))


if __name__ == "__main__":
    main()
