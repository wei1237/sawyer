#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute a simple MT3-style pick-and-place rollout in Sawyer Gazebo.

The MT3 pipeline writes grasp and place targets into /sawyer_auto_grasp/*.
This script intentionally keeps the first version conservative:
top grasp -> lift -> move above place target -> descend until the object is
on the table -> open gripper -> retreat upward.
"""

import json
import math
import os
import re
import sys
import threading
import copy
import time
import xml.etree.ElementTree as ET

import numpy as np
import geometry_msgs.msg
import moveit_commander
import rospy
import tf
from gazebo_msgs.srv import GetModelState
from intera_core_msgs.msg import RobotAssemblyState
from intera_interface import Gripper, RobotEnable
from mt3_sawyer_grasp import execute_demo_replay


ROS_NAMESPACE = "/robot"
PLANNING_GROUP = "right_arm"
END_EFFECTOR_LINK = "right_hand"
INSERT_DIAG_VERSION = "2026-08-17_joint_contact_diag_v3"

ORI_VEL_SCALE = 0.25
ORI_ACC_SCALE = 0.25
DOWN_VEL_SCALE = 0.08
DOWN_ACC_SCALE = 0.08
CART_STEP = 0.006
TOP_FLANGE_Z_OFFSET = 0.050
TOP_GRASP_FLANGE_Z_OFFSET = 0.040
ALLOWED_ERROR = 0.005
PLACE_REPLAY_DEFAULT_PRE_SAMPLES = 75
PLACE_REPLAY_DEFAULT_POST_SAMPLES = 165
PLACE_REPLAY_XY_CLAMP = 0.045
PLACE_REPLAY_Z_DOWN_CLAMP = 0.20
PLACE_REPLAY_Z_UP_CLAMP = 0.16


def _create_robot_enable_with_retry(max_attempts=5, retry_delay_s=1.0):
    """
    RobotEnable() creates a fresh /robot/state subscriber and can occasionally
    time out in Gazebo even when the topic is otherwise healthy.
    """
    last_exc = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            rospy.loginfo(
                "Initializing RobotEnable (%d/%d)",
                attempt, int(max_attempts))
            robot_enable = RobotEnable()
            rospy.loginfo(
                "RobotEnable initialized successfully (%d/%d)",
                attempt, int(max_attempts))
            return robot_enable
        except Exception as exc:
            last_exc = exc
            is_robot_state_timeout = (
                getattr(exc, "errno", None) == 110 or
                "Failed to get robot state" in str(exc))
            if not is_robot_state_timeout:
                raise
            rospy.logwarn(
                "RobotEnable timed out waiting for /robot/state "
                "(attempt %d/%d): %s",
                attempt, int(max_attempts), exc)
            if attempt < int(max_attempts):
                # Use wall-clock sleep because simulated time may pause.
                time.sleep(float(retry_delay_s))
    raise RuntimeError(
        "RobotEnable failed after %d attempts; last error: %s" %
        (int(max_attempts), last_exc))


def install_moveit_timing(move_group):
    """Accumulate MoveIt planning/execution wall time in ROS params."""
    plan_param = "/sawyer_auto_grasp/planning_time_s"
    exec_param = "/sawyer_auto_grasp/robot_execution_time_s"
    count_plan_param = "/sawyer_auto_grasp/planning_call_count"
    count_exec_param = "/sawyer_auto_grasp/robot_execution_call_count"
    rospy.set_param(plan_param, 0.0)
    rospy.set_param(exec_param, 0.0)
    rospy.set_param(count_plan_param, 0)
    rospy.set_param(count_exec_param, 0)
    rospy.set_param("/sawyer_auto_grasp/timing_source", "moveit_wrapper_v1")

    def _accumulate(param, count_param, dt):
        try:
            rospy.set_param(param, float(rospy.get_param(param, 0.0)) + float(dt))
            rospy.set_param(count_param, int(rospy.get_param(count_param, 0)) + 1)
        except Exception:
            pass

    original_plan = move_group.plan
    original_execute = move_group.execute
    original_go = move_group.go

    def timed_plan(*args, **kwargs):
        t0 = time.time()
        try:
            return original_plan(*args, **kwargs)
        finally:
            _accumulate(plan_param, count_plan_param, time.time() - t0)

    def timed_execute(*args, **kwargs):
        t0 = time.time()
        try:
            return original_execute(*args, **kwargs)
        finally:
            _accumulate(exec_param, count_exec_param, time.time() - t0)

    def timed_go(*args, **kwargs):
        t0 = time.time()
        try:
            return original_go(*args, **kwargs)
        finally:
            _accumulate(exec_param, count_exec_param, time.time() - t0)

    move_group.plan = timed_plan
    move_group.execute = timed_execute
    move_group.go = timed_go


class EndEffectorTrajectoryRecorder(object):
    def __init__(self, move_group, gripper=None, rate_hz=10.0):
        self.move_group = move_group
        self.gripper = gripper
        self.rate_hz = float(rate_hz)
        self.samples = []
        self.events = []
        # Synchronous diagnostic snapshots are independent of the background
        # sampler, so they still work after the grasp replay stops sampling.
        self.diagnostic_snapshots = []
        self._stop = threading.Event()
        self._thread = None
        self.record_model_states = _param_bool(
            "/sawyer_auto_grasp/record_model_state_samples", False)
        self.target_model_name = (
            str(rospy.get_param(
                "/sawyer_auto_grasp/target_model_name", "")).strip()
            if self.record_model_states else "")
        self.socket_model_name = (
            str(rospy.get_param(
                "/sawyer_auto_grasp/socket_model_name", "")).strip()
            if self.record_model_states else "")
        self.desired_relation_offset_xy = [
            float(rospy.get_param(
                "/sawyer_auto_grasp/insert_desired_offset_x", 0.0)),
            float(rospy.get_param(
                "/sawyer_auto_grasp/insert_desired_offset_y", 0.0)),
        ]

        # Joint/Jacobian diagnostics are read-only. They never change planning
        # targets or execution.  The translational Jacobian (3xN) is the main
        # metric because insertion failure is dominated by Cartesian position
        # tracking rather than wrist orientation.
        try:
            self.joint_names = list(self.move_group.get_active_joints())
        except Exception:
            self.joint_names = []
        self.joint_limits = self._load_joint_limits()

        self._gazebo_state = None
        if self.target_model_name or self.socket_model_name:
            try:
                rospy.wait_for_service("/gazebo/get_model_state", timeout=1.0)
                self._gazebo_state = rospy.ServiceProxy(
                    "/gazebo/get_model_state", GetModelState)
            except Exception as exc:
                rospy.logwarn(
                    "Rollout model-state sampling disabled: %s", exc)

    def _load_joint_limits(self):
        """Load current robot joint limits from the active URDF."""
        result = {}
        xml_text = ""
        for param_name in (
                "/robot/robot_description",
                "robot_description",
                "/robot_description"):
            try:
                xml_text = rospy.get_param(param_name, "")
            except Exception:
                xml_text = ""
            if xml_text:
                break
        if not xml_text:
            rospy.logwarn(
                "Joint-limit diagnostic disabled: robot_description not found")
            return result

        try:
            root = ET.fromstring(xml_text)
            for joint in root.findall("joint"):
                name = str(joint.attrib.get("name", ""))
                jtype = str(joint.attrib.get("type", ""))
                limit = joint.find("limit")
                if not name or limit is None or jtype == "continuous":
                    continue
                if "lower" not in limit.attrib or "upper" not in limit.attrib:
                    continue
                lo = float(limit.attrib["lower"])
                hi = float(limit.attrib["upper"])
                if hi > lo:
                    result[name] = (lo, hi)
        except Exception as exc:
            rospy.logwarn("Joint-limit diagnostic parse failed: %s", exc)
            return {}
        return result

    def _model_state_diag(self, model_name):
        """Return pose/twist diagnostics from Gazebo without changing state."""
        if not self._gazebo_state or not model_name:
            return None
        try:
            resp = self._gazebo_state(model_name, "world")
            if not resp.success:
                return None
            q = [
                float(resp.pose.orientation.x),
                float(resp.pose.orientation.y),
                float(resp.pose.orientation.z),
                float(resp.pose.orientation.w),
            ]
            tilt_deg = None
            try:
                rot = tf.transformations.quaternion_matrix(q)
                world_z_component = max(
                    -1.0, min(1.0, float(rot[2, 2])))
                tilt_deg = math.degrees(math.acos(world_z_component))
            except Exception:
                pass
            vx = float(resp.twist.linear.x)
            vy = float(resp.twist.linear.y)
            vz = float(resp.twist.linear.z)
            return {
                "xyz_world": [
                    float(resp.pose.position.x),
                    float(resp.pose.position.y),
                    float(resp.pose.position.z),
                ],
                "orientation_xyzw_world": q,
                "linear_velocity_xyz_world": [vx, vy, vz],
                "angular_velocity_xyz_world": [
                    float(resp.twist.angular.x),
                    float(resp.twist.angular.y),
                    float(resp.twist.angular.z),
                ],
                "speed_xy_m_s": math.sqrt(vx * vx + vy * vy),
                "speed_xyz_m_s": math.sqrt(vx * vx + vy * vy + vz * vz),
                "tilt_deg": tilt_deg,
            }
        except Exception:
            return None

    def _joint_jacobian_diag(self):
        """Read joint configuration and local Cartesian conditioning.

        Primary quantities:
          * translational_jacobian_condition_number:
              larger -> worse local Cartesian conditioning.
          * translational_jacobian_sigma_min:
              smaller -> closer to a translational singularity.
          * vertical_joint_velocity_gain:
              joint-speed norm required for unit downward EE velocity;
              larger -> harder to generate the insertion direction.
          * joint_limit_min_margin_normalized:
              0 at a joint limit, ~0.5 near the middle of a symmetric range.
        """
        try:
            q = [float(x) for x in self.move_group.get_current_joint_values()]
            names = list(self.joint_names)
            if len(names) != len(q):
                try:
                    names = list(self.move_group.get_active_joints())
                except Exception:
                    names = ["joint_%d" % i for i in range(len(q))]

            diag = {
                "joint_names": names,
                "joint_positions_rad": q,
            }

            # Joint-limit margins.
            margin_abs = []
            margin_norm = []
            per_joint_margin = {}
            for name, value in zip(names, q):
                bounds = self.joint_limits.get(name)
                if bounds is None:
                    continue
                lo, hi = bounds
                span = hi - lo
                m = min(value - lo, hi - value)
                mn = m / span if span > 0.0 else None
                per_joint_margin[name] = {
                    "lower": lo,
                    "upper": hi,
                    "margin_rad": float(m),
                    "margin_normalized": (
                        float(mn) if mn is not None else None),
                }
                margin_abs.append(float(m))
                if mn is not None:
                    margin_norm.append(float(mn))
            if per_joint_margin:
                diag["joint_limit_margins"] = per_joint_margin
            if margin_abs:
                diag["joint_limit_min_margin_rad"] = min(margin_abs)
            if margin_norm:
                diag["joint_limit_min_margin_normalized"] = min(margin_norm)

            # MoveIt returns a 6xN geometric Jacobian. Use the first three rows
            # for the position-only insertion conditioning diagnostic.
            jac = np.asarray(
                self.move_group.get_jacobian_matrix(q), dtype=float)
            if jac.ndim == 2 and jac.shape[0] >= 3 and jac.shape[1] == len(q):
                jv = jac[:3, :]
                s = np.linalg.svd(jv, compute_uv=False)
                s = [float(x) for x in s]
                diag["translational_jacobian_singular_values"] = s
                if s:
                    sigma_min = min(s)
                    sigma_max = max(s)
                    diag["translational_jacobian_sigma_min"] = sigma_min
                    diag["translational_manipulability"] = float(
                        np.prod(np.asarray(s, dtype=float)))
                    if sigma_min > 1.0e-9:
                        diag["translational_jacobian_condition_number"] = (
                            sigma_max / sigma_min)

                # Directional difficulty for downward insertion. This is a
                # comparative metric between trials at the same waypoint.
                try:
                    qdot_unit_down = np.linalg.pinv(jv).dot(
                        np.asarray([0.0, 0.0, -1.0], dtype=float))
                    gain = float(np.linalg.norm(qdot_unit_down))
                    if math.isfinite(gain):
                        diag["vertical_joint_velocity_gain"] = gain
                except Exception:
                    pass

            return diag
        except Exception as exc:
            rospy.logwarn("Joint/Jacobian diagnostic failed: %s", exc)
            return {}

    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def mark_event(self, name):
        self.events.append({
            "name": str(name),
            "t": float(rospy.get_time()),
            "sample_index": max(0, len(self.samples) - 1),
        })

    def _model_xyz(self, model_name):
        if not self._gazebo_state or not model_name:
            return None
        try:
            msg = self._gazebo_state(model_name, "world")
            if not msg.success:
                return None
            return [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ]
        except Exception:
            return None


    def _set_diag_param(self, suffix, value):
        try:
            rospy.set_param("/sawyer_auto_grasp/%s" % str(suffix), value)
        except Exception:
            pass

    def _publish_diagnostic_params(self, label, snapshot):
        """Publish compact scalar diagnostics for CSV/JSONL trial logging."""
        safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", str(label)).strip("_").lower()
        prefix = "diag_%s" % safe_label
        keys = (
            "cylinder_hand_offset_x_m",
            "cylinder_hand_offset_y_m",
            "cylinder_hand_offset_xy_m",
            "cylinder_socket_offset_x_m",
            "cylinder_socket_offset_y_m",
            "cylinder_socket_error_xy_m",
            "hand_tracking_error_xyz_m",
            "translational_jacobian_condition_number",
            "translational_jacobian_sigma_min",
            "translational_manipulability",
            "vertical_joint_velocity_gain",
            "joint_limit_min_margin_rad",
            "joint_limit_min_margin_normalized",
            "cylinder_speed_xy_m_s",
            "cylinder_speed_xyz_m_s",
            "cylinder_tilt_deg",
        )
        for key in keys:
            if key in snapshot:
                self._set_diag_param("%s_%s" % (prefix, key), snapshot[key])

        # Keep summary values for Step F in ROS params.  Detailed per-chunk
        # snapshots remain in the rollout JSON.
        if safe_label.startswith("step_f_chunk_"):
            relation_err = snapshot.get("cylinder_socket_error_xy_m")
            tracking_err = snapshot.get("hand_tracking_error_xyz_m")

            if relation_err is not None:
                first_name = "diag_step_f_first_cylinder_socket_error_xy_m"
                last_name = "diag_step_f_last_cylinder_socket_error_xy_m"
                max_name = "diag_step_f_max_cylinder_socket_error_xy_m"
                try:
                    first_val = rospy.get_param(
                        "/sawyer_auto_grasp/%s" % first_name, "")
                except Exception:
                    first_val = ""
                if first_val in ("", None):
                    self._set_diag_param(first_name, float(relation_err))
                self._set_diag_param(last_name, float(relation_err))
                try:
                    old_max = rospy.get_param(
                        "/sawyer_auto_grasp/%s" % max_name, "")
                    old_max = float(old_max) if old_max not in ("", None) else None
                except Exception:
                    old_max = None
                self._set_diag_param(
                    max_name,
                    float(relation_err) if old_max is None
                    else max(float(old_max), float(relation_err)))

            # Joint/contact-diagnostic summaries across Step F.
            summary_specs = (
                ("translational_jacobian_condition_number", "max"),
                ("vertical_joint_velocity_gain", "max"),
                ("cylinder_speed_xy_m_s", "max"),
                ("cylinder_tilt_deg", "max"),
                ("translational_jacobian_sigma_min", "min"),
                ("translational_manipulability", "min"),
                ("joint_limit_min_margin_normalized", "min"),
            )
            for metric_name, mode in summary_specs:
                metric_value = snapshot.get(metric_name)
                if metric_value is None:
                    continue
                try:
                    metric_value = float(metric_value)
                    if not math.isfinite(metric_value):
                        continue
                except Exception:
                    continue

                first_name = "diag_step_f_first_%s" % metric_name
                last_name = "diag_step_f_last_%s" % metric_name
                agg_name = "diag_step_f_%s_%s" % (mode, metric_name)

                try:
                    first_val = rospy.get_param(
                        "/sawyer_auto_grasp/%s" % first_name, "")
                except Exception:
                    first_val = ""
                if first_val in ("", None):
                    self._set_diag_param(first_name, metric_value)
                self._set_diag_param(last_name, metric_value)

                try:
                    old_value = rospy.get_param(
                        "/sawyer_auto_grasp/%s" % agg_name, "")
                    old_value = (
                        float(old_value)
                        if old_value not in ("", None) else None)
                except Exception:
                    old_value = None

                if old_value is None:
                    new_value = metric_value
                elif mode == "max":
                    new_value = max(old_value, metric_value)
                else:
                    new_value = min(old_value, metric_value)
                self._set_diag_param(agg_name, new_value)

            if tracking_err is not None:
                last_name = "diag_step_f_last_hand_tracking_error_xyz_m"
                max_name = "diag_step_f_max_hand_tracking_error_xyz_m"
                self._set_diag_param(last_name, float(tracking_err))
                try:
                    old_max = rospy.get_param(
                        "/sawyer_auto_grasp/%s" % max_name, "")
                    old_max = float(old_max) if old_max not in ("", None) else None
                except Exception:
                    old_max = None
                self._set_diag_param(
                    max_name,
                    float(tracking_err) if old_max is None
                    else max(float(old_max), float(tracking_err)))

    def capture_diagnostic(self, label, planned_pose=None):
        """Capture one diagnostic snapshot without commanding robot motion.

        right_hand/planned hand are in Sawyer base. Gazebo model states are in
        world. In this simulation their X-Y axes are aligned, so XY differences
        are useful. We intentionally do not form cylinder-hand Z offsets across
        these frames.
        """
        try:
            hand_pose = self.move_group.get_current_pose().pose
            hand_xyz = [
                float(hand_pose.position.x),
                float(hand_pose.position.y),
                float(hand_pose.position.z),
            ]
            snapshot = {
                "label": str(label),
                "t": float(rospy.get_time()),
                "hand_xyz_base": hand_xyz,
                "hand_orientation_xyzw": [
                    float(hand_pose.orientation.x),
                    float(hand_pose.orientation.y),
                    float(hand_pose.orientation.z),
                    float(hand_pose.orientation.w),
                ],
            }

            target_state = self._model_state_diag(self.target_model_name)
            socket_state = self._model_state_diag(self.socket_model_name)
            target_xyz = (
                target_state.get("xyz_world") if target_state else None)
            socket_xyz = (
                socket_state.get("xyz_world") if socket_state else None)

            if target_state is not None:
                snapshot["cylinder_model_state"] = target_state
                snapshot["cylinder_speed_xy_m_s"] = target_state.get(
                    "speed_xy_m_s")
                snapshot["cylinder_speed_xyz_m_s"] = target_state.get(
                    "speed_xyz_m_s")
                snapshot["cylinder_tilt_deg"] = target_state.get("tilt_deg")

            if socket_state is not None:
                snapshot["socket_model_state"] = socket_state

            if target_xyz is not None:
                snapshot["cylinder_model_xyz_world"] = target_xyz
                ch_dx = float(target_xyz[0]) - hand_xyz[0]
                ch_dy = float(target_xyz[1]) - hand_xyz[1]
                snapshot["cylinder_hand_offset_x_m"] = ch_dx
                snapshot["cylinder_hand_offset_y_m"] = ch_dy
                snapshot["cylinder_hand_offset_xy_m"] = math.sqrt(
                    ch_dx * ch_dx + ch_dy * ch_dy)

            if socket_xyz is not None:
                snapshot["socket_model_xyz_world"] = socket_xyz

            if target_xyz is not None and socket_xyz is not None:
                rel_x = (
                    float(target_xyz[0]) - float(socket_xyz[0]) -
                    self.desired_relation_offset_xy[0])
                rel_y = (
                    float(target_xyz[1]) - float(socket_xyz[1]) -
                    self.desired_relation_offset_xy[1])
                snapshot["cylinder_socket_offset_x_m"] = rel_x
                snapshot["cylinder_socket_offset_y_m"] = rel_y
                snapshot["cylinder_socket_error_xy_m"] = math.sqrt(
                    rel_x * rel_x + rel_y * rel_y)

            if planned_pose is not None:
                planned_xyz = [
                    float(planned_pose.position.x),
                    float(planned_pose.position.y),
                    float(planned_pose.position.z),
                ]
                snapshot["planned_hand_xyz_base"] = planned_xyz
                ex = hand_xyz[0] - planned_xyz[0]
                ey = hand_xyz[1] - planned_xyz[1]
                ez = hand_xyz[2] - planned_xyz[2]
                snapshot["hand_tracking_error_xyz_m_components"] = [ex, ey, ez]
                snapshot["hand_tracking_error_xyz_m"] = math.sqrt(
                    ex * ex + ey * ey + ez * ez)

            joint_diag = self._joint_jacobian_diag()
            if joint_diag:
                snapshot.update(joint_diag)

            self.diagnostic_snapshots.append(snapshot)
            self._publish_diagnostic_params(label, snapshot)

            rospy.loginfo(
                "INSERT DIAG %s: hand=[%.3f %.3f %.3f] "
                "cyl_hand_xy=%s cyl_socket_xy=%s tracking=%s "
                "Jcond=%s sigma_min=%s vert_gain=%s qmargin=%s "
                "cyl_vxy=%s tilt=%s",
                str(label),
                hand_xyz[0], hand_xyz[1], hand_xyz[2],
                ("%.1fmm" % (
                    float(snapshot["cylinder_hand_offset_xy_m"]) * 1000.0)
                 if "cylinder_hand_offset_xy_m" in snapshot else "n/a"),
                ("%.1fmm" % (
                    float(snapshot["cylinder_socket_error_xy_m"]) * 1000.0)
                 if "cylinder_socket_error_xy_m" in snapshot else "n/a"),
                ("%.1fmm" % (
                    float(snapshot["hand_tracking_error_xyz_m"]) * 1000.0)
                 if "hand_tracking_error_xyz_m" in snapshot else "n/a"),
                ("%.2f" % float(
                    snapshot["translational_jacobian_condition_number"])
                 if snapshot.get(
                    "translational_jacobian_condition_number") is not None
                 else "n/a"),
                ("%.4f" % float(
                    snapshot["translational_jacobian_sigma_min"])
                 if snapshot.get(
                    "translational_jacobian_sigma_min") is not None
                 else "n/a"),
                ("%.3f" % float(snapshot["vertical_joint_velocity_gain"])
                 if snapshot.get("vertical_joint_velocity_gain") is not None
                 else "n/a"),
                ("%.3f" % float(
                    snapshot["joint_limit_min_margin_normalized"])
                 if snapshot.get(
                    "joint_limit_min_margin_normalized") is not None
                 else "n/a"),
                ("%.3fm/s" % float(snapshot["cylinder_speed_xy_m_s"])
                 if snapshot.get("cylinder_speed_xy_m_s") is not None
                 else "n/a"),
                ("%.2fdeg" % float(snapshot["cylinder_tilt_deg"])
                 if snapshot.get("cylinder_tilt_deg") is not None
                 else "n/a"))
            return snapshot
        except Exception as exc:
            rospy.logwarn("INSERT DIAG %s capture failed: %s", str(label), exc)
            return None

    def _run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self._stop.is_set():
            try:
                pose = self.move_group.get_current_pose().pose
                gripper_pos = None
                if self.gripper is not None:
                    try:
                        gripper_pos = float(self.gripper.get_position())
                    except Exception:
                        gripper_pos = None
                sample = {
                    "t": float(rospy.get_time()),
                    "position": [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ],
                    "orientation": [
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ],
                    "gripper_position": gripper_pos,
                }
                target_xyz = self._model_xyz(self.target_model_name)
                socket_xyz = self._model_xyz(self.socket_model_name)
                if target_xyz is not None:
                    sample["target_model_xyz"] = target_xyz
                if socket_xyz is not None:
                    sample["socket_model_xyz"] = socket_xyz
                if target_xyz is not None and socket_xyz is not None:
                    dx = (
                        float(target_xyz[0]) - float(socket_xyz[0]) -
                        self.desired_relation_offset_xy[0])
                    dy = (
                        float(target_xyz[1]) - float(socket_xyz[1]) -
                        self.desired_relation_offset_xy[1])
                    sample["target_socket_relation_error_xy_m"] = math.sqrt(
                        dx * dx + dy * dy)
                self.samples.append(sample)
            except Exception:
                pass
            rate.sleep()

    def save(self, path, success):
        if not path:
            return None
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        self._annotate_gripper_states()
        velocities = []
        for i, sample in enumerate(self.samples):
            if i == 0:
                linear = [0.0, 0.0, 0.0]
            else:
                prev = self.samples[i - 1]
                dt = max(1e-6, sample["t"] - prev["t"])
                linear = [
                    (sample["position"][j] - prev["position"][j]) / dt
                    for j in range(3)
                ]
            velocities.append({
                "t": sample["t"],
                "linear": [float(v) for v in linear],
                "angular": [0.0, 0.0, 0.0],
                "gripper_position": sample.get("gripper_position"),
            })
        data = {
            "format": "sampled_pick_place_rollout_v1",
            "frame": "base",
            "sample_rate_hz": self.rate_hz,
            "num_waypoints": len(self.samples),
            "poses": self.samples,
            "velocities": velocities,
            "events": self.events,
            "diagnostic_snapshots": self.diagnostic_snapshots,
            "target_model_name": self.target_model_name,
            "socket_model_name": self.socket_model_name,
            "desired_relation_offset_xy": self.desired_relation_offset_xy,
            "success": bool(success),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _annotate_gripper_states(self):
        positions = [
            float(s["gripper_position"])
            for s in self.samples
            if s.get("gripper_position") is not None
        ]
        if len(positions) < 5:
            return
        low = min(positions)
        high = max(positions)
        if abs(high - low) < 1e-5:
            return

        first = positions[:min(10, len(positions))]
        open_reference = sum(first) / float(len(first))
        midpoint = 0.5 * (low + high)
        open_is_high = open_reference >= midpoint

        for i, sample in enumerate(self.samples):
            value = sample.get("gripper_position")
            if value is None:
                continue
            value = float(value)
            closed = value < midpoint if open_is_high else value > midpoint
            state = 1 if closed else 0
            sample["gripper_state"] = state
            if i + 1 < len(self.samples):
                next_value = self.samples[i + 1].get("gripper_position")
                if next_value is not None:
                    next_closed = (
                        float(next_value) < midpoint
                        if open_is_high else float(next_value) > midpoint)
                    sample["gripper_next"] = 1 if next_closed else 0


def _make_pose(x, y, z, q):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def _sample_orientation_xyzw(sample, fallback_q):
    ori = (sample or {}).get("orientation")
    if ori is None:
        ori = (sample or {}).get("orientation_xyzw")
    if isinstance(ori, dict):
        try:
            return [
                float(ori.get("x", fallback_q[0])),
                float(ori.get("y", fallback_q[1])),
                float(ori.get("z", fallback_q[2])),
                float(ori.get("w", fallback_q[3])),
            ]
        except Exception:
            return fallback_q
    if isinstance(ori, (list, tuple)) and len(ori) >= 4:
        try:
            return [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
        except Exception:
            return fallback_q
    return fallback_q


def _param_bool(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _go_pose(move_group, pose, label, velocity=ORI_VEL_SCALE,
             acceleration=ORI_ACC_SCALE, attempts=3, planning_time=8.0):
    rospy.loginfo(
        "%s target: [%.3f, %.3f, %.3f]",
        label, pose.position.x, pose.position.y, pose.position.z)
    move_group.set_max_velocity_scaling_factor(float(velocity))
    move_group.set_max_acceleration_scaling_factor(float(acceleration))
    move_group.set_planning_time(float(planning_time))
    for attempt in range(attempts):
        move_group.set_pose_target(pose)
        plan_result = move_group.plan()
        ok = bool(plan_result[0])
        if ok:
            if not move_group.execute(plan_result[1], wait=True):
                rospy.logwarn("%s execution failed on attempt %d/%d",
                              label, attempt + 1, attempts)
                move_group.stop()
                move_group.clear_pose_targets()
                continue
            rospy.sleep(0.4)
            move_group.stop()
            move_group.clear_pose_targets()
            return True
        rospy.logwarn("%s planning retry %d/%d", label, attempt + 1, attempts)
    move_group.clear_pose_targets()
    rospy.logerr("%s failed", label)
    return False


def _cartesian_to(move_group, pose, label, min_fraction=0.90,
                  eef_step=CART_STEP, velocity_scale=None,
                  acceleration_scale=None, fallback_step_z=0.012,
                  fallback_sleep=0.15):
    """Staged Cartesian move: XY alignment first, then Z in small steps.

    The single-segment Cartesian path triggers J4/J5 CONTROL_FAILED at the
    workspace edge (x >= 0.60). Splitting into XY + incremental Z avoids this.
    """
    import copy

    vel = float(DOWN_VEL_SCALE if velocity_scale is None else velocity_scale)
    acc = float(DOWN_ACC_SCALE if acceleration_scale is None else acceleration_scale)
    move_group.set_max_velocity_scaling_factor(vel)
    move_group.set_max_acceleration_scaling_factor(acc)

    # Step 1: Cartesian XY align (keep current Z)
    current = move_group.get_current_pose().pose
    xy_target = copy.deepcopy(pose)
    xy_target.position.z = current.position.z
    wp_xy = [copy.deepcopy(current), copy.deepcopy(xy_target)]
    plan_xy, frac_xy = move_group.compute_cartesian_path(wp_xy, float(eef_step), True)
    rospy.loginfo("%s XY fraction: %.1f%%", label, frac_xy * 100.0)
    if frac_xy >= 0.9 and plan_xy.joint_trajectory.points:
        if not move_group.execute(plan_xy, wait=True):
            move_group.stop()
            rospy.logwarn("%s XY execute failed", label)
            return False
        move_group.stop()
        rospy.sleep(0.3)
    elif frac_xy < min_fraction:
        rospy.logwarn("%s XY cartesian insufficient", label)
        return False

    # Step 2: Cartesian Z segment
    current = move_group.get_current_pose().pose
    z_target = copy.deepcopy(pose)
    z_target.position.x = current.position.x
    z_target.position.y = current.position.y
    wp_z = [copy.deepcopy(current), copy.deepcopy(z_target)]
    plan_z, frac_z = move_group.compute_cartesian_path(wp_z, float(eef_step), True)
    rospy.loginfo("%s Z fraction: %.1f%%", label, frac_z * 100.0)
    if frac_z >= 0.9 and plan_z.joint_trajectory.points:
        if move_group.execute(plan_z, wait=True):
            move_group.stop()
            rospy.sleep(0.3)
            return True
        move_group.stop()
        rospy.logwarn("%s Z execute failed; trying small-step fallback", label)
        rospy.sleep(0.3)
    elif frac_z < min_fraction:
        rospy.logwarn("%s Z cartesian insufficient; trying small-step fallback", label)

    # Step 3: Fallback — tiny Z steps to avoid joint tracking spikes
    rospy.logwarn("%s Z fallback: small steps", label)
    move_group.set_max_velocity_scaling_factor(vel)
    move_group.set_max_acceleration_scaling_factor(acc)
    for i in range(20):
        current = move_group.get_current_pose().pose
        remaining = z_target.position.z - current.position.z
        if abs(remaining) < 0.005:
            rospy.loginfo("  %s fallback step %d: reached (error=%.3f)", label, i, abs(remaining))
            return True
        step_limit = abs(float(fallback_step_z))
        step_z = max(-step_limit, min(step_limit, remaining))
        step_pose = copy.deepcopy(current)
        step_pose.position.z += step_z
        step_pose.orientation = copy.deepcopy(z_target.orientation)
        wp = [copy.deepcopy(current), copy.deepcopy(step_pose)]
        plan, frac = move_group.compute_cartesian_path(wp, 0.003, True)
        if frac >= 0.8 and plan.joint_trajectory.points:
            if not move_group.execute(plan, wait=True):
                move_group.stop()
                rospy.logwarn("  %s fallback step %d: execute failed", label, i)
                break
            move_group.stop()
            rospy.sleep(float(fallback_sleep))
        else:
            rospy.logwarn("  %s fallback step %d: plan failed (frac=%.1f%%)", label, i, frac * 100.0)
            break

    actual = move_group.get_current_pose().pose
    xy_error = ((actual.position.x - pose.position.x) ** 2 +
                (actual.position.y - pose.position.y) ** 2) ** 0.5
    z_error = abs(actual.position.z - z_target.position.z)
    rospy.loginfo(
        "%s final error: xy=%.3fm z=%.3fm "
        "(target=[%.3f, %.3f, %.3f] actual=[%.3f, %.3f, %.3f])",
        label, xy_error, z_error,
        pose.position.x, pose.position.y, z_target.position.z,
        actual.position.x, actual.position.y, actual.position.z)
    return xy_error < 0.020 and z_error < 0.015


def _cartesian_to_legacy_grasp(move_group, pose, label,
                               min_fraction=0.90, eef_step=CART_STEP):
    """Original scripted grasp motion used before insertion replay changes."""
    start = move_group.get_current_pose().pose
    move_group.set_max_velocity_scaling_factor(float(DOWN_VEL_SCALE))
    move_group.set_max_acceleration_scaling_factor(float(DOWN_ACC_SCALE))
    plan, fraction = move_group.compute_cartesian_path(
        [start, pose],
        float(eef_step),
        True,
    )
    rospy.loginfo("%s legacy cartesian fraction: %.1f%%",
                  label, fraction * 100.0)
    if fraction < min_fraction or not plan.joint_trajectory.points:
        rospy.logwarn(
            "%s legacy cartesian insufficient; using pose target fallback",
            label)
        return _go_pose(
            move_group, pose, label + " legacy fallback",
            velocity=DOWN_VEL_SCALE, acceleration=DOWN_ACC_SCALE,
            attempts=2, planning_time=6.0)
    if not move_group.execute(plan, wait=True):
        move_group.stop()
        rospy.logwarn("%s legacy cartesian execute failed", label)
        return False
    move_group.stop()
    rospy.sleep(0.4)
    return True


def _cartesian_to_top_grasp(move_group, pose, label):
    """Top-grasp descent matching mt3_sawyer_grasp edge behavior.

    At workspace-edge y positions the Sawyer wrist often stops several cm above
    the planned flange z.  The standalone MT3 grasp executor accepts that as a
    valid top grasp if XY is centered and the z error is still within the close
    window, instead of repeatedly pushing downward.
    """
    edge_mode = abs(float(pose.position.y)) >= float(rospy.get_param(
        "/sawyer_auto_grasp/edge_y_threshold", 0.08))
    pregrasp_clearance = float(rospy.get_param(
        "/sawyer_auto_grasp/pregrasp_clearance", 0.025))
    if edge_mode:
        pregrasp_clearance += float(rospy.get_param(
            "/sawyer_auto_grasp/edge_pregrasp_extra", 0.015))
    final_step = float(rospy.get_param(
        "/sawyer_auto_grasp/final_descent_step",
        0.002 if edge_mode else 0.003))
    accept_xy = float(rospy.get_param(
        "/sawyer_auto_grasp/top_grasp_accept_xy", 0.030 if edge_mode else 0.020))
    accept_z = float(rospy.get_param(
        "/sawyer_auto_grasp/top_grasp_accept_z", 0.070))

    move_group.set_max_velocity_scaling_factor(0.05)
    move_group.set_max_acceleration_scaling_factor(0.05)

    current = move_group.get_current_pose().pose
    recenter = copy.deepcopy(current)
    recenter.position.x = pose.position.x
    recenter.position.y = pose.position.y
    recenter.position.z = pose.position.z + pregrasp_clearance
    recenter.orientation = copy.deepcopy(pose.orientation)
    rospy.loginfo(
        "%s top pregrasp recenter: z=%.3f clearance=%.3f step=%.3f edge=%s",
        label, recenter.position.z, pregrasp_clearance, final_step, edge_mode)

    # Recenter XY at a low but still safe height.  Repeat because the edge
    # workspace sometimes leaves a centimeter-level residual after execution.
    for attempt in range(3):
        start = move_group.get_current_pose().pose
        recenter_start = copy.deepcopy(start)
        recenter_goal = copy.deepcopy(recenter)
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(recenter_start), copy.deepcopy(recenter_goal)],
            0.003,
            True)
        rospy.loginfo("%s top recenter %d fraction: %.1f%%",
                      label, attempt + 1, fraction * 100.0)
        if fraction >= 0.90 and plan.joint_trajectory.points:
            if not move_group.execute(plan, wait=True):
                move_group.stop()
                rospy.logwarn("%s top recenter %d execute failed",
                              label, attempt + 1)
                break
            move_group.stop()
            rospy.sleep(0.25)
        cur = move_group.get_current_pose().pose
        xy_error = ((cur.position.x - pose.position.x) ** 2 +
                    (cur.position.y - pose.position.y) ** 2) ** 0.5
        rospy.loginfo("%s top recenter %d xy_error=%.3fm",
                      label, attempt + 1, xy_error)
        if xy_error <= 0.003:
            break

    descend_start = move_group.get_current_pose().pose
    descend_goal = copy.deepcopy(descend_start)
    descend_goal.position.x = pose.position.x
    descend_goal.position.y = pose.position.y
    descend_goal.position.z = pose.position.z
    descend_goal.orientation = copy.deepcopy(pose.orientation)
    plan, fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(descend_start), copy.deepcopy(descend_goal)],
        final_step,
        True)
    rospy.loginfo("%s top final descent fraction: %.1f%%",
                  label, fraction * 100.0)
    if fraction >= 0.90 and plan.joint_trajectory.points:
        if not move_group.execute(plan, wait=True):
            move_group.stop()
            rospy.logwarn("%s top final descent execute failed; checking close window",
                          label)
        else:
            move_group.stop()
            rospy.sleep(0.25)
    else:
        rospy.logwarn("%s top final descent plan insufficient; checking close window",
                      label)

    actual = move_group.get_current_pose().pose
    xy_error = ((actual.position.x - pose.position.x) ** 2 +
                (actual.position.y - pose.position.y) ** 2) ** 0.5
    z_error = abs(actual.position.z - pose.position.z)
    rospy.loginfo(
        "%s top final error: xy=%.3fm z=%.3fm "
        "(target=[%.3f, %.3f, %.3f] actual=[%.3f, %.3f, %.3f])",
        label, xy_error, z_error,
        pose.position.x, pose.position.y, pose.position.z,
        actual.position.x, actual.position.y, actual.position.z)
    if xy_error <= accept_xy and z_error <= accept_z:
        rospy.logwarn(
            "%s accepted inside top-grasp close window (xy=%.1fcm z=%.1fcm)",
            label, xy_error * 100.0, z_error * 100.0)
        return True
    return False


def _lift_current_xy(move_group, z, q, label):
    current = move_group.get_current_pose().pose
    lift_pose = copy.deepcopy(current)
    lift_pose.position.z = float(z)
    lift_pose.orientation.x = float(q[0])
    lift_pose.orientation.y = float(q[1])
    lift_pose.orientation.z = float(q[2])
    lift_pose.orientation.w = float(q[3])
    return _cartesian_to_legacy_grasp(move_group, lift_pose, label)


def _pose_position_array(pose):
    return [float(pose.position.x), float(pose.position.y), float(pose.position.z)]


def _array_to_pose_like(reference_pose, xyz):
    pose = copy.deepcopy(reference_pose)
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    return pose


def _lookup_tf_point(listener, frame):
    try:
        listener.waitForTransform("base", frame, rospy.Time(0), rospy.Duration(0.3))
        trans, _ = listener.lookupTransform("base", frame, rospy.Time(0))
        return [float(trans[0]), float(trans[1]), float(trans[2])]
    except Exception:
        return None


def _get_gripper_mouth_state(listener, move_group):
    left_frame = str(rospy.get_param(
        '/sawyer_auto_grasp/left_finger_tip_frame',
        'right_gripper_l_finger_tip'))
    right_frame = str(rospy.get_param(
        '/sawyer_auto_grasp/right_finger_tip_frame',
        'right_gripper_r_finger_tip'))
    left = _lookup_tf_point(listener, left_frame)
    right = _lookup_tf_point(listener, right_frame)
    hand_pose = move_group.get_current_pose().pose
    hand = _pose_position_array(hand_pose)
    if left is None or right is None:
        rospy.logwarn(
            "Gripper mouth TF unavailable; using right_hand as fallback center")
        return {
            "available": False,
            "left": None, "right": None,
            "center": hand, "opening": 0.0,
            "hand": hand, "offset": [0.0, 0.0, 0.0],
        }
    center = [
        0.5 * (left[0] + right[0]),
        0.5 * (left[1] + right[1]),
        0.5 * (left[2] + right[2]),
    ]
    opening = math.sqrt(
        (left[0] - right[0]) ** 2 +
        (left[1] - right[1]) ** 2 +
        (left[2] - right[2]) ** 2)
    return {
        "available": True,
        "left": left, "right": right,
        "center": center, "opening": opening,
        "hand": hand,
        "offset": [
            center[0] - hand[0],
            center[1] - hand[1],
            center[2] - hand[2],
        ],
    }


def _log_top_mouth_xy_check(listener, move_group, label, desired_xy):
    state = _get_gripper_mouth_state(listener, move_group)
    center = state["center"]
    hand = state["hand"]
    dx = center[0] - float(desired_xy[0])
    dy = center[1] - float(desired_xy[1])
    offset = state.get("offset", [0.0, 0.0, 0.0])
    rospy.loginfo(
        "%s top mouth-center check: hand=[%.3f, %.3f, %.3f] "
        "mouth=[%.3f, %.3f, %.3f] desired_xy=[%.3f, %.3f] "
        "mouth_err_xy=[%.1f, %.1f]cm hand_to_mouth_xy=[%.1f, %.1f]cm "
        "tf_available=%s",
        label, hand[0], hand[1], hand[2],
        center[0], center[1], center[2],
        float(desired_xy[0]), float(desired_xy[1]),
        dx * 100.0, dy * 100.0,
        float(offset[0]) * 100.0, float(offset[1]) * 100.0,
        state.get("available", False))
    return state, dx, dy


def _align_top_mouth_xy_before_descent(listener, move_group, label, desired_xy):
    """Align the open finger-mouth center before descending around the object."""
    if not rospy.get_param(
            '/sawyer_auto_grasp/use_top_mouth_center_predescent_alignment',
            True):
        return None

    state, dx, dy = _log_top_mouth_xy_check(
        listener, move_group, "%s predescent" % label, desired_xy)
    if not state.get("available", False):
        rospy.logwarn(
            "%s predescent alignment skipped: gripper mouth TF unavailable",
            label)
        return None

    offset = state.get("offset", [0.0, 0.0, 0.0])
    target_x = float(desired_xy[0]) - float(offset[0])
    target_y = float(desired_xy[1]) - float(offset[1])
    current = move_group.get_current_pose().pose
    shift = math.sqrt(
        (target_x - current.position.x) ** 2 +
        (target_y - current.position.y) ** 2)
    max_shift = float(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_predescent_max_shift', 0.050))
    if shift > max_shift:
        scale = max_shift / max(shift, 1e-6)
        target_x = current.position.x + (target_x - current.position.x) * scale
        target_y = current.position.y + (target_y - current.position.y) * scale
        rospy.logwarn(
            "%s predescent mouth alignment clamped to %.1fcm shift",
            label, max_shift * 100.0)

    target = copy.deepcopy(current)
    target.position.x = target_x
    target.position.y = target_y
    plan, fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(current), copy.deepcopy(target)], 0.003, True)
    rospy.loginfo(
        "%s predescent mouth alignment: hand_xy [%.3f, %.3f] -> "
        "[%.3f, %.3f] fraction=%.1f%%",
        label, current.position.x, current.position.y,
        target_x, target_y, fraction * 100.0)
    if fraction >= 0.90 and len(plan.joint_trajectory.points) > 0:
        ok = move_group.execute(plan, wait=True)
        move_group.stop()
        rospy.sleep(0.3)
        if ok:
            _log_top_mouth_xy_check(
                listener, move_group, "%s predescent final" % label,
                desired_xy)
            return [target_x, target_y]
        rospy.logwarn("%s predescent mouth alignment execute failed", label)
    else:
        rospy.logwarn(
            "%s predescent mouth alignment planning insufficient: %.1f%%",
            label, fraction * 100.0)
    return None


def _correct_top_mouth_xy_before_close(listener, move_group, label, desired_xy):
    """Small final XY correction using finger-tip TF before gripper close."""
    if not rospy.get_param(
            '/sawyer_auto_grasp/use_top_mouth_center_final_correction', True):
        return False

    tol = float(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_tolerance', 0.003))
    max_step = float(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_final_max_step', 0.018))
    attempts = int(rospy.get_param(
        '/sawyer_auto_grasp/top_mouth_xy_final_attempts', 3))

    for attempt in range(max(1, attempts)):
        state, dx, dy = _log_top_mouth_xy_check(
            listener, move_group, "%s correction %d" % (label, attempt + 1),
            desired_xy)
        if not state.get("available", False):
            rospy.logwarn(
                "%s correction skipped: gripper mouth TF unavailable", label)
            return False
        if abs(dx) <= tol and abs(dy) <= tol:
            rospy.loginfo(
                "%s correction done: mouth_err_xy=[%.1f, %.1f]cm",
                label, dx * 100.0, dy * 100.0)
            rospy.set_param(
                '/sawyer_auto_grasp/top_mouth_xy_final_error_m',
                [float(dx), float(dy)])
            return True

        err = math.sqrt(dx * dx + dy * dy)
        scale = min(1.0, max_step / max(err, 1e-6))
        current = move_group.get_current_pose().pose
        target = copy.deepcopy(current)
        target.position.x = current.position.x - dx * scale
        target.position.y = current.position.y - dy * scale
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(current), copy.deepcopy(target)], 0.003, True)
        rospy.loginfo(
            "%s correction %d cartesian fraction: %.1f%% target_hand_xy=[%.3f, %.3f]",
            label, attempt + 1, fraction * 100.0,
            target.position.x, target.position.y)
        if fraction >= 0.90 and len(plan.joint_trajectory.points) > 0:
            ok = move_group.execute(plan, wait=True)
            move_group.stop()
            rospy.sleep(0.3)
            if not ok:
                rospy.logwarn("%s correction %d execute failed", label, attempt + 1)
        else:
            rospy.logwarn("%s correction %d planning insufficient: %.1f%%",
                          label, attempt + 1, fraction * 100.0)
    return False


def _record_before_close_mouth_xy(listener, move_group, label, desired_xy):
    """Log mouth-center error before gripper close for diagnostics."""
    try:
        state, dx, dy = _log_top_mouth_xy_check(
            listener, move_group, label, desired_xy)
        center = state.get("center")
        if center is None:
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy', ["", ""])
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy', ["", ""])
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', "")
            rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', "")
            return
        err = math.sqrt(float(dx) ** 2 + float(dy) ** 2)
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy',
                        [float(center[0]), float(center[1])])
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy',
                        [float(dx), float(dy)])
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', float(center[0]))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', float(center[1]))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', float(dx))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', float(dy))
        rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', float(err))
    except Exception as exc:
        rospy.logwarn("record_before_close_mouth_xy failed: %s", exc)


def _execute_mt3_top_grasp_core(move_group, gripper, grasp_x, grasp_y, grasp_z,
                                grasp_flange_z, q, object_size,
                                trajectory_recorder=None):
    """Use the same scripted top-grasp core as mt3_sawyer_grasp.py.

    The standalone grasp script succeeds at workspace edges because it does not
    force the flange to exactly reach the nominal target. It aligns above the
    object, performs a short low pregrasp recenter, accepts a z-close window,
    closes the gripper, then lifts from the actual reached pose.
    """
    target_pose = _make_pose(grasp_x, grasp_y, grasp_flange_z, q)
    transition_x = float(rospy.get_param(
        "/sawyer_auto_grasp/top_grasp_transition_x", 0.50))
    safe_approach_height = float(rospy.get_param(
        "/sawyer_auto_grasp/top_grasp_safe_approach_height", 0.15))
    transition_z = float(grasp_z) + 0.30
    overhead_z = float(grasp_z) + safe_approach_height
    edge_mode = abs(float(grasp_y)) >= float(rospy.get_param(
        "/sawyer_auto_grasp/edge_y_threshold", 0.08))
    pregrasp_clearance = float(rospy.get_param(
        "/sawyer_auto_grasp/pregrasp_clearance", 0.025))
    if edge_mode:
        pregrasp_clearance += float(rospy.get_param(
            "/sawyer_auto_grasp/edge_pregrasp_extra", 0.015))
    final_step = float(rospy.get_param(
        "/sawyer_auto_grasp/final_descent_step",
        0.002 if edge_mode else 0.003))

    rospy.loginfo("MT3 top grasp core: edge_mode=%s target=[%.3f, %.3f, %.3f]",
                  edge_mode, grasp_x, grasp_y, grasp_flange_z)

    rospy.loginfo("TopGrasp Step1: transition point")
    transition_pose = copy.deepcopy(target_pose)
    transition_pose.position.x = transition_x
    transition_pose.position.y = float(grasp_y)
    transition_pose.position.z = transition_z
    if not _go_pose(move_group, transition_pose, "TopGrasp Step1 transition",
                    velocity=ORI_VEL_SCALE, acceleration=ORI_ACC_SCALE,
                    attempts=2, planning_time=5.0):
        return False, None

    rospy.loginfo("TopGrasp Step2: move above object")
    overhead_pose = copy.deepcopy(target_pose)
    overhead_pose.position.x = float(grasp_x)
    overhead_pose.position.y = float(grasp_y)
    overhead_pose.position.z = overhead_z
    if not _go_pose(move_group, overhead_pose, "TopGrasp Step2 overhead",
                    velocity=ORI_VEL_SCALE, acceleration=ORI_ACC_SCALE,
                    attempts=3, planning_time=5.0):
        rospy.logwarn("TopGrasp Step2 overhead failed; continuing with current pose")

    rospy.loginfo("TopGrasp Step3: precise XY align")
    start_pose = move_group.get_current_pose().pose
    target_align_pose = copy.deepcopy(start_pose)
    target_align_pose.position.x = float(grasp_x)
    target_align_pose.position.y = float(grasp_y)
    target_align_pose.position.z = overhead_z
    target_align_pose.orientation = copy.deepcopy(target_pose.orientation)
    plan, fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(start_pose), copy.deepcopy(target_align_pose)],
        CART_STEP,
        True)
    rospy.loginfo("TopGrasp Step3 XY cartesian fraction: %.1f%%",
                  fraction * 100.0)
    if fraction >= 0.90 and plan.joint_trajectory.points:
        move_group.execute(plan, wait=True)
        move_group.stop()
        rospy.sleep(0.5)
    else:
        if not _go_pose(move_group, target_align_pose,
                        "TopGrasp Step3 XY fallback",
                        velocity=ORI_VEL_SCALE, acceleration=ORI_ACC_SCALE,
                        attempts=2, planning_time=5.0):
            return False, None

    for xy_retry in range(3):
        final_align = move_group.get_current_pose().pose
        x_error = abs(final_align.position.x - float(grasp_x))
        y_error = abs(final_align.position.y - float(grasp_y))
        if x_error <= ALLOWED_ERROR and y_error <= ALLOWED_ERROR:
            break
        rospy.logwarn(
            "TopGrasp XY residual before descent: x=%.4fm y=%.4fm; correction %d/3",
            x_error, y_error, xy_retry + 1)
        correction_pose = copy.deepcopy(final_align)
        correction_pose.position.x = float(grasp_x)
        correction_pose.position.y = float(grasp_y)
        correction_pose.orientation = copy.deepcopy(target_pose.orientation)
        correction_plan, correction_fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(final_align), copy.deepcopy(correction_pose)],
            0.003,
            True)
        if correction_fraction >= 0.98 and correction_plan.joint_trajectory.points:
            move_group.execute(correction_plan, wait=True)
            move_group.stop()
            rospy.sleep(0.3)
        else:
            rospy.logwarn(
                "TopGrasp XY correction planning insufficient: %.1f%%",
                correction_fraction * 100.0)
            break

    final_align = move_group.get_current_pose().pose
    rospy.loginfo(
        "TopGrasp XY final before descent x=%.3f err=%.4fm y=%.3f err=%.4fm",
        final_align.position.x, abs(final_align.position.x - float(grasp_x)),
        final_align.position.y, abs(final_align.position.y - float(grasp_y)))

    hand_grasp_x = float(grasp_x)
    hand_grasp_y = float(grasp_y)
    predescent_listener = tf.TransformListener()
    rospy.sleep(0.2)
    aligned_hand_xy = _align_top_mouth_xy_before_descent(
        predescent_listener, move_group, "TopGrasp", [grasp_x, grasp_y])
    if aligned_hand_xy:
        hand_grasp_x = float(aligned_hand_xy[0])
        hand_grasp_y = float(aligned_hand_xy[1])

    rospy.loginfo("TopGrasp Step4: descend to grasp z=%.3f", grasp_flange_z)
    move_group.set_max_velocity_scaling_factor(DOWN_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(DOWN_ACC_SCALE)
    move_group.set_goal_position_tolerance(0.01)
    move_group.set_goal_orientation_tolerance(0.05)

    descent_success = False
    final_pose_after_descent = None
    pregrasp_pose = move_group.get_current_pose().pose
    pregrasp_goal = copy.deepcopy(pregrasp_pose)
    pregrasp_goal.position.x = hand_grasp_x
    pregrasp_goal.position.y = hand_grasp_y
    pregrasp_goal.position.z = float(grasp_flange_z) + pregrasp_clearance
    pregrasp_goal.orientation = copy.deepcopy(target_pose.orientation)
    rospy.loginfo(
        "  pregrasp recenter: hand_xy=[%.3f, %.3f] z=%.3f "
        "clearance=%.3f step=%.3f",
        hand_grasp_x, hand_grasp_y, pregrasp_goal.position.z,
        pregrasp_clearance, final_step)
    pregrasp_plan, pregrasp_fraction = move_group.compute_cartesian_path(
        [copy.deepcopy(pregrasp_pose), copy.deepcopy(pregrasp_goal)],
        0.003,
        True)
    if pregrasp_fraction >= 0.98 and pregrasp_plan.joint_trajectory.points:
        move_group.execute(pregrasp_plan, wait=True)
        move_group.stop()
        rospy.sleep(0.3)
    else:
        rospy.logwarn("  pregrasp planning insufficient: %.1f%%",
                      pregrasp_fraction * 100.0)

    for retry in range(3):
        descend_start = move_group.get_current_pose().pose
        descend_goal = copy.deepcopy(descend_start)
        descend_goal.position.x = hand_grasp_x
        descend_goal.position.y = hand_grasp_y
        descend_goal.position.z = float(grasp_flange_z)
        descend_goal.orientation = copy.deepcopy(target_pose.orientation)
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(descend_start), copy.deepcopy(descend_goal)],
            final_step,
            True)
        if fraction >= 0.98 and plan.joint_trajectory.points:
            if move_group.execute(plan, wait=True):
                move_group.stop()
                rospy.sleep(0.3)
                final_pose_after_descent = move_group.get_current_pose().pose
                actual_z = final_pose_after_descent.position.z
                z_error = abs(actual_z - float(grasp_flange_z))
                rospy.loginfo(
                    "  Cartesian descent: target_z=%.3f actual_z=%.3f error=%.1fcm",
                    grasp_flange_z, actual_z, z_error * 100.0)
                if z_error < 0.05:
                    descent_success = True
                    rospy.loginfo("  descent success")
                    break
                rospy.logwarn("  retry %d/3 z error %.1fcm",
                              retry + 1, z_error * 100.0)
            else:
                move_group.stop()
                rospy.logwarn("  retry %d/3 descent execute failed",
                              retry + 1)
        else:
            rospy.logwarn(
                "  retry %d/3 Cartesian descent insufficient: %.1f%%",
                retry + 1, fraction * 100.0)

    if not descent_success:
        rospy.logwarn("  Cartesian descent failed; trying small-step pose descent")
        best_error = float("inf")
        stalled_steps = 0
        for step_idx in range(18):
            current_pose = move_group.get_current_pose().pose
            current_error = abs(current_pose.position.z - float(grasp_flange_z))
            if current_error < 0.050:
                final_pose_after_descent = current_pose
                descent_success = True
                rospy.loginfo(
                    "  small-step descent success: z=%.3f error=%.1fcm",
                    current_pose.position.z, current_error * 100.0)
                break
            remaining = float(grasp_flange_z) - current_pose.position.z
            dz = max(-0.010, min(0.010, remaining))
            step_goal = copy.deepcopy(target_pose)
            step_goal.position.x = hand_grasp_x
            step_goal.position.y = hand_grasp_y
            step_goal.position.z = current_pose.position.z + dz
            step_goal.orientation = copy.deepcopy(target_pose.orientation)

            move_group.set_pose_target(step_goal)
            ok = move_group.go(wait=True)
            move_group.stop()
            rospy.sleep(0.25)

            actual_pose = move_group.get_current_pose().pose
            error = abs(actual_pose.position.z - float(grasp_flange_z))
            rospy.loginfo(
                "  small-step descent %02d: ok=%s actual_z=%.3f target_z=%.3f error=%.1fcm",
                step_idx + 1, ok, actual_pose.position.z, grasp_flange_z,
                error * 100.0)
            if error < best_error - 0.002:
                best_error = error
                stalled_steps = 0
            else:
                stalled_steps += 1
            if error < 0.050:
                final_pose_after_descent = actual_pose
                descent_success = True
                rospy.loginfo("  small-step descent reached close height")
                break
            if stalled_steps >= 3 and best_error <= 0.060:
                final_pose_after_descent = actual_pose
                descent_success = True
                rospy.logwarn(
                    "  small-step descent stalled near target; accepting z error %.1fcm",
                    best_error * 100.0)
                break

    if not descent_success:
        rospy.logwarn("  all descents failed; trying grasp at current height")
        final_pose_after_descent = move_group.get_current_pose().pose

    move_group.set_goal_position_tolerance(0.005)
    move_group.set_goal_orientation_tolerance(0.02)
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
    move_group.set_num_planning_attempts(2)
    move_group.set_planning_time(5.0)

    rospy.loginfo("  final descent z=%.3f", final_pose_after_descent.position.z)

    descent_z_error = abs(final_pose_after_descent.position.z -
                          float(grasp_flange_z))
    if descent_z_error > 0.070:
        rospy.logwarn(
            "  Step4 final z error is %.1fcm; trying one fine descent before grasp",
            descent_z_error * 100.0)
        fine_tune_pose = copy.deepcopy(final_pose_after_descent)
        fine_tune_pose.position.x = hand_grasp_x
        fine_tune_pose.position.y = hand_grasp_y
        fine_tune_pose.position.z = float(grasp_flange_z)
        fine_tune_pose.orientation = copy.deepcopy(target_pose.orientation)
        fine_plan, fine_fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(final_pose_after_descent), copy.deepcopy(fine_tune_pose)],
            0.003,
            True)
        if fine_fraction >= 0.98 and fine_plan.joint_trajectory.points:
            move_group.execute(fine_plan, wait=True)
            move_group.stop()
            rospy.sleep(0.5)
            final_pose_after_descent = move_group.get_current_pose().pose
            descent_z_error = abs(final_pose_after_descent.position.z -
                                  float(grasp_flange_z))
            rospy.loginfo(
                "  fine descent result: target_z=%.3f actual_z=%.3f error=%.1fcm",
                grasp_flange_z, final_pose_after_descent.position.z,
                descent_z_error * 100.0)
        else:
            rospy.logwarn("  fine descent planning insufficient: %.1f%%",
                          fine_fraction * 100.0)

    if descent_z_error > 0.070:
        rospy.logerr("  descent still too high (%.1fcm); abort grasp",
                     descent_z_error * 100.0)
        safe_pose = copy.deepcopy(final_pose_after_descent)
        safe_pose.position.z += 0.10
        move_group.set_pose_target(safe_pose)
        move_group.go(wait=True)
        move_group.stop()
        return False, final_pose_after_descent

    rospy.loginfo("TopGrasp Step5: mouth-center correction before close")
    tf_listener = tf.TransformListener()
    rospy.sleep(0.2)
    _correct_top_mouth_xy_before_close(
        tf_listener, move_group, "Place grasp before close",
        [grasp_x, grasp_y])
    _record_before_close_mouth_xy(
        tf_listener, move_group, "Place grasp before close",
        [grasp_x, grasp_y])

    rospy.loginfo("TopGrasp Step6: close gripper")
    if trajectory_recorder is not None:
        try:
            trajectory_recorder.mark_event("gripper_close")
        except Exception:
            pass
    obj_width = float(object_size[1]) if len(object_size) >= 2 else 0.045
    expected_closed = max(0.005, obj_width - 0.005)
    is_gripped = False
    try:
        initial_gripper_pos = float(gripper.get_position())
        rospy.loginfo(
            "  gripper initial: %.3fm expected holding position: ~%.3fm",
            initial_gripper_pos, expected_closed)
        gripper.close()
        rospy.sleep(2.0)
        current_gripper_pos = float(gripper.get_position())
        closure = initial_gripper_pos - current_gripper_pos
        if closure > 0.005 and current_gripper_pos > 0.003:
            is_gripped = True
            rospy.loginfo(
                "  grasp success: %.3f -> %.3f (closure %.0fmm)",
                initial_gripper_pos, current_gripper_pos, closure * 1000.0)
        elif current_gripper_pos < 0.005:
            rospy.logwarn("  gripper fully closed %.3fm; retry once",
                          current_gripper_pos)
            gripper.open()
            rospy.sleep(1.0)
            gripper.close()
            rospy.sleep(2.0)
            current_gripper_pos = float(gripper.get_position())
            closure = initial_gripper_pos - current_gripper_pos
            if closure > 0.005 and current_gripper_pos > 0.003:
                is_gripped = True
                rospy.loginfo("  retry grasp success")
            else:
                rospy.logwarn("  retry still did not confirm grasp")
        else:
            rospy.logwarn("  gripper closure too small %.0fmm",
                          closure * 1000.0)
    except Exception as exc:
        rospy.logwarn("  gripper SDK exception during top grasp: %s", exc)

    rospy.loginfo("TopGrasp Step6: vertical lift")
    lift_pose_final = copy.deepcopy(final_pose_after_descent)
    lift_pose_final.position.z = float(grasp_z) + 0.15
    move_group.set_pose_target(lift_pose_final)
    plan_result = move_group.plan()
    if bool(plan_result[0]):
        move_group.execute(plan_result[1], wait=True)
        move_group.stop()
        rospy.sleep(0.5)
        if not is_gripped:
            rospy.logwarn(
                "  gripper encoder did not confirm closure, but lift completed; continuing")
        rospy.loginfo("TopGrasp lift success: z=%.3f", lift_pose_final.position.z)
        return True, move_group.get_current_pose().pose

    rospy.logerr("TopGrasp lift planning failed")
    return False, final_pose_after_descent


def _gripper_binary(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return 1 if float(value) >= 0.5 else 0
    except Exception:
        text = str(value).strip().lower()
        if text in ("closed", "close", "closing", "1", "true"):
            return 1
        if text in ("open", "opening", "0", "false"):
            return 0
    return None


def _load_replay_payload(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        rospy.logwarn("Failed to read replay file %s: %s", path, exc)
        return None


def _replay_position_xyz(value):
    if not isinstance(value, dict):
        return None
    pos = value.get("position") or value.get("position_m") or value
    try:
        if isinstance(pos, dict):
            return [
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            ]
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        return None
    return None


def _replay_trajectory_base_close_xyz(payload):
    trajectory = (payload or {}).get("trajectory", {})
    if not isinstance(trajectory, dict):
        return None, None, None
    poses = trajectory.get("poses", [])
    if not poses:
        return None, None, None
    base_xyz = _replay_position_xyz(
        {"position": trajectory.get("base_position")})
    if base_xyz is None:
        base_xyz = _replay_position_xyz(poses[0])
    try:
        close_idx = int(trajectory.get(
            "close_index", (payload or {}).get("close_index", 0)))
    except Exception:
        close_idx = 0
    close_idx = max(0, min(len(poses) - 1, close_idx))
    close_xyz = _replay_position_xyz(poses[close_idx])
    return base_xyz, close_xyz, close_idx


def _load_replay_poses(path):
    if not path:
        return []
    if not os.path.exists(path):
        rospy.logwarn("Place replay file does not exist: %s", path)
        return []
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        rospy.logwarn("Failed to read place replay file %s: %s", path, exc)
        return []

    trajectory = payload.get("trajectory", payload)
    poses = trajectory.get("poses", [])
    release_index = trajectory.get("release_index")
    valid = []
    explicit_release_idx = None
    if release_index is not None:
        try:
            explicit_release_idx = int(release_index)
        except Exception:
            explicit_release_idx = None
    if explicit_release_idx is not None:
        rospy.loginfo("Place replay using explicit release_index=%d",
                      explicit_release_idx)
    for idx, sample in enumerate(poses):
        pos = sample.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            if explicit_release_idx is not None:
                # Insertion trajectories already carry the release event index.
                # Force replay to use that event instead of noisy delayed
                # gripper sensor transitions from the recorded rollout.
                sample["gripper_state"] = 1 if idx < explicit_release_idx else 0
                sample["gripper_next"] = 1 if idx < explicit_release_idx else 0
            valid.append(sample)
    return valid


def _find_release_open_index(poses):
    """Find the demo sample where the place-stage gripper opens.

    Recorded pick-place demos use gripper_next=1 while carrying the object and
    gripper_next=0 after release.  The last 1 -> 0 transition is the placement
    open event.  If explicit transitions are missing, fall back to the first
    open state after a closed state has appeared.
    """
    last_transition = None
    prev = None
    seen_closed = False
    fallback = None
    for idx, sample in enumerate(poses):
        state = _gripper_binary(sample.get("gripper_next"))
        if state is None:
            state = _gripper_binary(sample.get("gripper_state"))
        if prev == 1 and state == 0:
            last_transition = idx
        if seen_closed and state == 0 and fallback is None:
            fallback = idx
        if state == 1:
            seen_closed = True
        if state is not None:
            prev = state
    return last_transition if last_transition is not None else fallback


def _relative_replay_pose(sample, demo_start_position, runtime_start_pose, q):
    pos = sample.get("position", [0.0, 0.0, 0.0])
    dx = float(pos[0]) - float(demo_start_position[0])
    dy = float(pos[1]) - float(demo_start_position[1])
    dz = float(pos[2]) - float(demo_start_position[2])

    dx = max(-PLACE_REPLAY_XY_CLAMP, min(PLACE_REPLAY_XY_CLAMP, dx))
    dy = max(-PLACE_REPLAY_XY_CLAMP, min(PLACE_REPLAY_XY_CLAMP, dy))
    dz = max(-PLACE_REPLAY_Z_DOWN_CLAMP, min(PLACE_REPLAY_Z_UP_CLAMP, dz))
    replay_q = (
        _sample_orientation_xyzw(sample, q)
        if _param_bool("/sawyer_auto_grasp/place_replay_use_recorded_orientation", True)
        else q)

    return _make_pose(
        runtime_start_pose.position.x + dx,
        runtime_start_pose.position.y + dy,
        runtime_start_pose.position.z + dz,
        replay_q)


def _relative_replay_pose_to_anchor(sample, demo_anchor_position,
                                    runtime_anchor_pose, q, lock_xy=False):
    pos = sample.get("position", [0.0, 0.0, 0.0])
    dx = float(pos[0]) - float(demo_anchor_position[0])
    dy = float(pos[1]) - float(demo_anchor_position[1])
    dz = float(pos[2]) - float(demo_anchor_position[2])

    if lock_xy:
        dx = 0.0
        dy = 0.0
    else:
        dx = max(-PLACE_REPLAY_XY_CLAMP, min(PLACE_REPLAY_XY_CLAMP, dx))
        dy = max(-PLACE_REPLAY_XY_CLAMP, min(PLACE_REPLAY_XY_CLAMP, dy))
    dz = max(-PLACE_REPLAY_Z_DOWN_CLAMP, min(PLACE_REPLAY_Z_UP_CLAMP, dz))
    replay_q = (
        _sample_orientation_xyzw(sample, q)
        if _param_bool("/sawyer_auto_grasp/place_replay_use_recorded_orientation", True)
        else q)

    return _make_pose(
        runtime_anchor_pose.position.x + dx,
        runtime_anchor_pose.position.y + dy,
        runtime_anchor_pose.position.z + dz,
        replay_q)


def _execute_waypoints(move_group, waypoints, label,
                       min_fraction=0.75, eef_step=0.004,
                       velocity_scale=None, acceleration_scale=None,
                       post_sleep=0.25, diagnostic_recorder=None,
                       diagnostic_prefix="",
                       tracking_error_max_m=None,
                       tracking_failure_stage=""):
    """Execute place/insertion replay waypoints, segmented by default.

    The legacy implementation planned every replay waypoint into one long
    Cartesian trajectory and sent it to the controller in a single execute().
    At the Sawyer workspace edge this can produce CONTROL_FAILED even when
    compute_cartesian_path() reports 100%%.  Segmented execution preserves the
    same replay waypoints and ordering, but replans each chunk from the robot's
    actual current pose so tracking error does not accumulate across the whole
    insertion trajectory.
    """
    if not waypoints:
        return True

    vel = float(DOWN_VEL_SCALE if velocity_scale is None else velocity_scale)
    acc = float(DOWN_ACC_SCALE if acceleration_scale is None else acceleration_scale)
    move_group.set_max_velocity_scaling_factor(vel)
    move_group.set_max_acceleration_scaling_factor(acc)

    use_segmented = _param_bool(
        "/sawyer_auto_grasp/place_replay_use_segmented_execution", True)
    chunk_size = max(1, int(rospy.get_param(
        "/sawyer_auto_grasp/place_replay_chunk_size", 12)))

    if not use_segmented:
        current = move_group.get_current_pose().pose
        path = [copy.deepcopy(current)] + [copy.deepcopy(p) for p in waypoints]
        plan, fraction = move_group.compute_cartesian_path(
            path, float(eef_step), True)
        rospy.loginfo("%s replay fraction: %.1f%%", label, fraction * 100.0)
        if fraction >= float(min_fraction) and plan.joint_trajectory.points:
            execute_ok = bool(move_group.execute(plan, wait=True))
            move_group.stop()
            planned = waypoints[-1]
            actual = move_group.get_current_pose().pose
            dx = actual.position.x - planned.position.x
            dy = actual.position.y - planned.position.y
            dz = actual.position.z - planned.position.z
            err_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            rospy.loginfo(
                "%s replay DEBUG endpoint: planned=[%.3f, %.3f, %.3f] "
                "actual=[%.3f, %.3f, %.3f] err=[%.1f, %.1f, %.1f]mm "
                "norm=%.1fmm execute_ok=%s",
                label,
                planned.position.x, planned.position.y, planned.position.z,
                actual.position.x, actual.position.y, actual.position.z,
                dx * 1000.0, dy * 1000.0, dz * 1000.0,
                err_norm * 1000.0, execute_ok)
            if diagnostic_recorder is not None and diagnostic_prefix:
                diagnostic_recorder.capture_diagnostic(
                    "%s_end" % diagnostic_prefix, planned_pose=planned)

            if (tracking_error_max_m is not None and
                    err_norm > float(tracking_error_max_m)):
                rospy.logerr(
                    "%s TRACKING FAILURE: endpoint error %.1fmm > %.1fmm "
                    "although execute_ok=%s",
                    label, err_norm * 1000.0,
                    float(tracking_error_max_m) * 1000.0, execute_ok)
                if tracking_failure_stage:
                    rospy.set_param(
                        "/sawyer_auto_grasp/insertion_replay_failure_stage",
                        str(tracking_failure_stage))
                    rospy.set_param(
                        "/sawyer_auto_grasp/failure_stage_detail",
                        str(tracking_failure_stage) + "_tracking_error")
                rospy.set_param(
                    "/sawyer_auto_grasp/insert_tracking_failure", True)
                rospy.set_param(
                    "/sawyer_auto_grasp/insert_tracking_failure_error_m",
                    float(err_norm))
                return False

            if not execute_ok:
                rospy.logwarn("%s replay execute failed", label)
                return False
            rospy.sleep(float(post_sleep))
            return True
        rospy.logwarn("%s replay path rejected (fraction=%.1f%%)",
                      label, fraction * 100.0)
        return False

    total = len(waypoints)
    num_chunks = (total + chunk_size - 1) // chunk_size
    rospy.loginfo(
        "%s segmented replay execution: %d waypoints, chunk_size=%d, chunks=%d",
        label, total, chunk_size, num_chunks)

    for chunk_index, start in enumerate(range(0, total, chunk_size), 1):
        end = min(start + chunk_size, total)
        chunk = waypoints[start:end]
        current = move_group.get_current_pose().pose
        path = [copy.deepcopy(current)] + [copy.deepcopy(p) for p in chunk]

        plan, fraction = move_group.compute_cartesian_path(
            path, float(eef_step), True)
        rospy.loginfo(
            "%s chunk %02d [%d-%d/%d] fraction: %.1f%%",
            label, chunk_index, start + 1, end, total,
            fraction * 100.0)

        if fraction < float(min_fraction) or not plan.joint_trajectory.points:
            rospy.logwarn(
                "%s segmented replay path rejected at chunk %02d "
                "[%d-%d/%d] (fraction=%.1f%%)",
                label, chunk_index, start + 1, end, total,
                fraction * 100.0)
            if diagnostic_recorder is not None and diagnostic_prefix:
                diagnostic_recorder.capture_diagnostic(
                    "%s_chunk_%02d_plan_rejected" % (
                        diagnostic_prefix, chunk_index),
                    planned_pose=chunk[-1])
            return False

        execute_ok = bool(move_group.execute(plan, wait=True))
        move_group.stop()

        planned = chunk[-1]
        actual = move_group.get_current_pose().pose
        dx = actual.position.x - planned.position.x
        dy = actual.position.y - planned.position.y
        dz = actual.position.z - planned.position.z
        err_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        rospy.loginfo(
            "%s DEBUG chunk endpoint: chunk %02d [%d-%d/%d] "
            "planned=[%.3f, %.3f, %.3f] actual=[%.3f, %.3f, %.3f] "
            "err=[%.1f, %.1f, %.1f]mm norm=%.1fmm execute_ok=%s",
            label, chunk_index, start + 1, end, total,
            planned.position.x, planned.position.y, planned.position.z,
            actual.position.x, actual.position.y, actual.position.z,
            dx * 1000.0, dy * 1000.0, dz * 1000.0,
            err_norm * 1000.0, execute_ok)

        if diagnostic_recorder is not None and diagnostic_prefix:
            diagnostic_recorder.capture_diagnostic(
                "%s_chunk_%02d" % (diagnostic_prefix, chunk_index),
                planned_pose=planned)

        if (tracking_error_max_m is not None and
                err_norm > float(tracking_error_max_m)):
            rospy.logerr(
                "%s TRACKING FAILURE at chunk %02d [%d-%d/%d]: "
                "endpoint error %.1fmm > %.1fmm although execute_ok=%s",
                label, chunk_index, start + 1, end, total,
                err_norm * 1000.0,
                float(tracking_error_max_m) * 1000.0,
                execute_ok)
            if tracking_failure_stage:
                rospy.set_param(
                    "/sawyer_auto_grasp/insertion_replay_failure_stage",
                    str(tracking_failure_stage))
                rospy.set_param(
                    "/sawyer_auto_grasp/failure_stage_detail",
                    str(tracking_failure_stage) + "_tracking_error")
            rospy.set_param(
                "/sawyer_auto_grasp/insert_tracking_failure", True)
            rospy.set_param(
                "/sawyer_auto_grasp/insert_tracking_failure_chunk",
                int(chunk_index))
            rospy.set_param(
                "/sawyer_auto_grasp/insert_tracking_failure_error_m",
                float(err_norm))
            return False

        if not execute_ok:
            rospy.logwarn(
                "%s segmented replay execute failed at chunk %02d "
                "[%d-%d/%d] (progress=%.1f%%)",
                label, chunk_index, start + 1, end, total,
                100.0 * float(end) / float(total))
            return False

    rospy.sleep(float(post_sleep))
    return True



def _safe_insert_transport(move_group, target_pose, recorder=None):
    """Move a grasped object to the pre-insertion pose without sweeping low.

    The held cylinder is not attached to the MoveIt planning scene, so a normal
    pose-target motion may choose a joint-space path that translates toward the
    socket before the cylinder has been lifted high enough.  This helper forces
    the free-space transport into three geometric stages:

      E1) vertical raise at the current XY,
      E2) horizontal translation at a safe constant Z,
      E3) vertical lower to the original pre-insertion target.

    The insertion replay itself is untouched and still starts after this
    transition.
    """
    current = move_group.get_current_pose().pose

    extra_clearance = max(0.0, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_extra_clearance", 0.030)))
    xy_step = max(0.005, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_xy_step", 0.025)))
    eef_step = max(0.001, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_eef_step", 0.004)))
    vel = max(0.005, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_velocity_scale", 0.060)))
    acc = max(0.005, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_acceleration_scale", 0.060)))
    post_sleep = max(0.0, float(rospy.get_param(
        "/sawyer_auto_grasp/insert_transport_post_sleep", 0.30)))

    # The existing Step-E target is already deliberately above the mapped
    # insertion bottleneck. Lift a little higher than that before any XY
    # translation. Never lower the robot just to reach this transport height.
    safe_z = max(
        float(current.position.z),
        float(target_pose.position.z) + extra_clearance)

    rospy.loginfo(
        "Step E safe transport: current=[%.3f, %.3f, %.3f] "
        "preinsert=[%.3f, %.3f, %.3f] safe_z=%.3f extra_clearance=%.3f",
        current.position.x, current.position.y, current.position.z,
        target_pose.position.x, target_pose.position.y, target_pose.position.z,
        safe_z, extra_clearance)

    # E1: raise vertically before moving toward the socket.
    if safe_z > float(current.position.z) + 0.003:
        raise_pose = copy.deepcopy(current)
        raise_pose.position.z = safe_z
        rospy.loginfo(
            "Step E1: raise carried object vertically before transport "
            "target_z=%.3f", safe_z)
        if not _cartesian_to(
                move_group, raise_pose,
                "Step E1: safe vertical raise",
                min_fraction=0.85,
                eef_step=eef_step,
                velocity_scale=vel,
                acceleration_scale=acc,
                fallback_step_z=0.010,
                fallback_sleep=0.20):
            rospy.set_param(
                "/sawyer_auto_grasp/failure_stage_detail",
                "transport_raise")
            rospy.logerr(
                "Safe insert transport failed during vertical raise.")
            return False
    else:
        rospy.loginfo(
            "Step E1: already at/above safe transport height; "
            "vertical raise skipped.")
    if recorder is not None:
        recorder.capture_diagnostic("step_e1_end")

    # E2: constant-height straight XY translation. Generate explicit
    # intermediate waypoints so the carried cylinder cannot sweep through the
    # socket while MoveIt chooses a curved joint-space transition.
    current = move_group.get_current_pose().pose
    dx = float(target_pose.position.x) - float(current.position.x)
    dy = float(target_pose.position.y) - float(current.position.y)
    distance_xy = math.sqrt(dx * dx + dy * dy)
    steps = max(1, int(math.ceil(distance_xy / xy_step)))

    horizontal_waypoints = []
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        wp = copy.deepcopy(current)
        wp.position.x = float(current.position.x) + alpha * dx
        wp.position.y = float(current.position.y) + alpha * dy
        wp.position.z = safe_z
        # Keep the grasp orientation fixed during transport.
        wp.orientation = copy.deepcopy(current.orientation)
        horizontal_waypoints.append(wp)

    rospy.loginfo(
        "Step E2: horizontal carried-object transport at safe height "
        "distance_xy=%.3fm waypoints=%d step<=%.3fm z=%.3f",
        distance_xy, len(horizontal_waypoints), xy_step, safe_z)

    if not _execute_waypoints(
            move_group, horizontal_waypoints,
            "Step E2: safe horizontal transport",
            min_fraction=0.90,
            eef_step=eef_step,
            velocity_scale=vel,
            acceleration_scale=acc,
            post_sleep=post_sleep,
            diagnostic_recorder=recorder,
            diagnostic_prefix="step_e2"):
        rospy.set_param(
            "/sawyer_auto_grasp/failure_stage_detail",
            "transport_xy")
        rospy.logerr(
            "Safe insert transport failed during horizontal translation.")
        return False
    if recorder is not None:
        recorder.capture_diagnostic("step_e2_end")

    # E3: lower only after the cylinder is horizontally centered over the
    # pre-insertion target. This preserves the original Step-E target and does
    # not modify the subsequent demonstrated insertion replay.
    current = move_group.get_current_pose().pose
    lower_target = copy.deepcopy(target_pose)
    if abs(float(current.position.z) -
           float(lower_target.position.z)) > 0.003:
        rospy.loginfo(
            "Step E3: lower vertically to pre-insertion height "
            "target=[%.3f, %.3f, %.3f]",
            lower_target.position.x,
            lower_target.position.y,
            lower_target.position.z)
        if not _cartesian_to(
                move_group, lower_target,
                "Step E3: lower to pre-insertion height",
                min_fraction=0.85,
                eef_step=eef_step,
                velocity_scale=vel,
                acceleration_scale=acc,
                fallback_step_z=0.010,
                fallback_sleep=0.20):
            rospy.set_param(
                "/sawyer_auto_grasp/failure_stage_detail",
                "transport_lower")
            rospy.logerr(
                "Safe insert transport failed while lowering to "
                "the pre-insertion pose.")
            return False
    else:
        rospy.loginfo(
            "Step E3: already at pre-insertion height; lower skipped.")
    if recorder is not None:
        recorder.capture_diagnostic(
            "step_e3_end", planned_pose=target_pose)

    final_pose = move_group.get_current_pose().pose
    dx = float(final_pose.position.x) - float(target_pose.position.x)
    dy = float(final_pose.position.y) - float(target_pose.position.y)
    dz = float(final_pose.position.z) - float(target_pose.position.z)
    err = math.sqrt(dx * dx + dy * dy + dz * dz)
    rospy.loginfo(
        "Step E safe transport complete: target=[%.3f, %.3f, %.3f] "
        "actual=[%.3f, %.3f, %.3f] error=%.1fmm",
        target_pose.position.x, target_pose.position.y, target_pose.position.z,
        final_pose.position.x, final_pose.position.y, final_pose.position.z,
        err * 1000.0)
    if recorder is not None:
        recorder.capture_diagnostic(
            "pre_step_f", planned_pose=target_pose)
    return True

def _payload_place_xyz(payload):
    place_info = (payload or {}).get("place_info") or {}
    xyz = place_info.get("place_xyz")
    if xyz is None:
        pose = place_info.get("place_pose_base_frame") or {}
        xyz = pose.get("position")
    try:
        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except Exception:
        return None
    return None


def _payload_pose_xyz(payload, key):
    return _replay_position_xyz(((payload or {}).get(key) or {}))


def _mapped_place_bottleneck_pose_from_replay(replay_path, q):
    payload = _load_replay_payload(replay_path)
    mapped_place_bn = _payload_pose_xyz(
        payload, "aligned_place_bottleneck_pose")
    demo_place_bn = _replay_position_xyz(
        (payload or {}).get("place_bottleneck_pose_base_frame") or {})
    if not mapped_place_bn or not demo_place_bn:
        return None
    return _make_pose(
        mapped_place_bn[0], mapped_place_bn[1], mapped_place_bn[2], q)


def _execute_place_release_replay(move_group, gripper, replay_path, q,
                                  place_anchor_pose=None, recorder=None):
    """Replay only the local release segment from a recorded demo.

    The global place target is still produced by language/MT3.  This function
    takes the demo's end-effector motion around the final gripper-open event
    and translates it to the current runtime pose above the generalized place
    target.
    """
    rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                    "insertion_replay_prepare")
    payload = _load_replay_payload(replay_path)
    poses = _load_replay_poses(replay_path)
    if len(poses) < 5:
        rospy.set_param('/sawyer_auto_grasp/insertion_replay_failure_stage',
                        "insertion_replay_invalid_input")
        rospy.logwarn("Place replay skipped: not enough valid poses")
        return False

    open_idx = _find_release_open_index(poses)
    if open_idx is None:
        rospy.set_param('/sawyer_auto_grasp/insertion_replay_failure_stage',
                        "insertion_release_event_missing")
        rospy.logwarn("Place replay skipped: could not find release open event")
        return False

    pre_samples = int(rospy.get_param(
        "/sawyer_auto_grasp/place_replay_pre_samples",
        PLACE_REPLAY_DEFAULT_PRE_SAMPLES))
    post_samples = int(rospy.get_param(
        "/sawyer_auto_grasp/place_replay_post_samples",
        PLACE_REPLAY_DEFAULT_POST_SAMPLES))
    stride = max(1, int(rospy.get_param(
        "/sawyer_auto_grasp/place_replay_stride", 2)))
    min_fraction = float(rospy.get_param(
        "/sawyer_auto_grasp/place_replay_min_fraction", 0.70))
    lock_xy = _param_bool(
        "/sawyer_auto_grasp/place_replay_lock_xy", True)
    approach_vel = None
    approach_acc = None
    approach_step = 0.004
    approach_sleep = 0.25
    if str(rospy.get_param(
            "/sawyer_auto_grasp/place_direction", "")).strip() == "insert_into_socket":
        approach_vel = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_replay_velocity_scale", 0.025))
        approach_acc = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_replay_acceleration_scale", 0.025))
        approach_step = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_replay_eef_step", 0.002))
        approach_sleep = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_replay_post_sleep", 0.40))

    mapped_place_bn = _payload_pose_xyz(payload, "aligned_place_bottleneck_pose")
    demo_place_bn = _replay_position_xyz(
        (payload or {}).get("place_bottleneck_pose_base_frame") or {})
    use_mapped_place_bottleneck = bool(mapped_place_bn and demo_place_bn)
    start_idx = (
        0 if use_mapped_place_bottleneck
        else max(0, int(open_idx) - max(1, pre_samples)))
    end_idx = min(len(poses), int(open_idx) + max(1, post_samples) + 1)
    runtime_start = copy.deepcopy(move_group.get_current_pose().pose)
    demo_start = poses[start_idx]["position"]
    demo_release = poses[open_idx]["position"]
    demo_place = _payload_place_xyz(payload)
    if use_mapped_place_bottleneck:
        runtime_anchor = _make_pose(
            mapped_place_bn[0], mapped_place_bn[1], mapped_place_bn[2], q)
        demo_anchor = demo_place_bn
        lock_xy = False
        rospy.loginfo(
            "Place replay anchored at mapped place bottleneck: "
            "runtime=[%.3f, %.3f, %.3f] demo=[%.3f, %.3f, %.3f]",
            runtime_anchor.position.x, runtime_anchor.position.y,
            runtime_anchor.position.z,
            float(demo_anchor[0]), float(demo_anchor[1]),
            float(demo_anchor[2]))
        rospy.loginfo(
            "Place replay preserves demo XY/Z offsets from mapped bottleneck")
    elif place_anchor_pose is not None:
        runtime_anchor = copy.deepcopy(place_anchor_pose)
        demo_anchor = demo_place if demo_place is not None else demo_release
        rospy.loginfo(
            "Place replay anchored at generalized place pose: [%.3f, %.3f, %.3f]",
            runtime_anchor.position.x, runtime_anchor.position.y,
            runtime_anchor.position.z)
        if demo_place is not None:
            rospy.loginfo(
                "Place replay release z from demo offset: demo_release_z %.3f - "
                "demo_place_z %.3f = %.3f",
                float(demo_release[2]), float(demo_place[2]),
                float(demo_release[2]) - float(demo_place[2]))
        else:
            rospy.logwarn(
                "Place replay has no demo place_xyz; falling back to release-event anchor")
        if lock_xy:
            rospy.loginfo(
                "Place replay XY locked to generalized place pose; replay keeps demo z profile")
    else:
        runtime_anchor = runtime_start
        demo_anchor = demo_start
        rospy.loginfo("Place replay anchored at segment start pose")

    before = []
    for sample in poses[start_idx:open_idx + 1:stride]:
        if use_mapped_place_bottleneck or place_anchor_pose is not None:
            before.append(_relative_replay_pose_to_anchor(
                sample, demo_anchor, runtime_anchor, q, lock_xy=lock_xy))
        else:
            before.append(_relative_replay_pose(
                sample, demo_start, runtime_start, q))
    open_pose = (
        _relative_replay_pose_to_anchor(
            poses[open_idx], demo_anchor, runtime_anchor, q,
            lock_xy=lock_xy)
        if use_mapped_place_bottleneck or place_anchor_pose is not None
        else _relative_replay_pose(poses[open_idx], demo_start, runtime_start, q))
    if before and before[-1] != open_pose:
        before.append(open_pose)

    after = []
    after_start = min(open_idx + 1, end_idx)
    for sample in poses[after_start:end_idx:stride]:
        if use_mapped_place_bottleneck or place_anchor_pose is not None:
            after.append(_relative_replay_pose_to_anchor(
                sample, demo_anchor, runtime_anchor, q, lock_xy=lock_xy))
        else:
            after.append(_relative_replay_pose(
                sample, demo_start, runtime_start, q))
    if end_idx > after_start:
        if use_mapped_place_bottleneck or place_anchor_pose is not None:
            after.append(_relative_replay_pose_to_anchor(
                poses[end_idx - 1], demo_anchor, runtime_anchor, q,
                lock_xy=lock_xy))
        else:
            after.append(_relative_replay_pose(
                poses[end_idx - 1], demo_start, runtime_start, q))

    rospy.loginfo(
        "Step F/G/H: place release replay from %s samples=[%d:%d] open=%d",
        replay_path, start_idx, end_idx, open_idx)
    if recorder is not None:
        recorder.mark_event("insert_replay_start")
        recorder.capture_diagnostic(
            "pre_step_f_replay",
            planned_pose=(before[0] if before else None))
    rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                    "insertion_step_f_replay")
    tracking_error_max_m = float(rospy.get_param(
        "/sawyer_auto_grasp/insert_replay_tracking_error_max_m", 0.020))
    rospy.set_param(
        "/sawyer_auto_grasp/insert_replay_tracking_error_max_m_active",
        tracking_error_max_m)
    rospy.loginfo(
        "Step F tracking gate: max endpoint error=%.1fmm "
        "(failure detection only; no correction/fallback)",
        tracking_error_max_m * 1000.0)

    if not _execute_waypoints(
            move_group, before, "Step F replay: descend/release approach",
            min_fraction=min_fraction, eef_step=approach_step,
            velocity_scale=approach_vel, acceleration_scale=approach_acc,
            post_sleep=approach_sleep,
            diagnostic_recorder=recorder,
            diagnostic_prefix="step_f",
            tracking_error_max_m=tracking_error_max_m,
            tracking_failure_stage="insertion_step_f_tracking"):
        current_failure_stage = str(rospy.get_param(
            '/sawyer_auto_grasp/insertion_replay_failure_stage', ""))
        if not current_failure_stage:
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_failure_stage',
                            "insertion_step_f_replay")
        return False

    if recorder is not None:
        recorder.capture_diagnostic(
            "step_f_complete", planned_pose=open_pose)
    rospy.sleep(0.5)
    rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                    "insertion_release")
    rospy.loginfo("Step G replay: open gripper at replay release event")
    if recorder is not None:
        recorder.mark_event("insert_release_open")
    gripper.open()
    rospy.sleep(1.0)
    if recorder is not None:
        recorder.capture_diagnostic(
            "post_release", planned_pose=open_pose)

    # Step F + G have completed: the insertion interaction has reached the
    # demonstrated release event. Step H is post-release retreat only.
    rospy.set_param('/sawyer_auto_grasp/insertion_interaction_success', True)
    rospy.set_param('/sawyer_auto_grasp/post_release_retreat_attempted', True)
    rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                    "insertion_step_h_replay")

    if not _execute_waypoints(
            move_group, after, "Step H replay: post-release retreat",
            min_fraction=min_fraction, eef_step=0.004,
            diagnostic_recorder=recorder,
            diagnostic_prefix="step_h"):
        rospy.set_param('/sawyer_auto_grasp/post_release_retreat_success', False)
        rospy.set_param('/sawyer_auto_grasp/insertion_replay_failure_stage',
                        "insertion_step_h_replay")
        rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                        "insertion_release_complete_retreat_failed")
        if recorder is not None:
            recorder.mark_event("insert_replay_retreat_failed")
        rospy.logwarn(
            "Step H post-release retreat failed after insertion/release completed. "
            "Task success will be decided by final Gazebo postcheck.")
        return False

    rospy.set_param('/sawyer_auto_grasp/post_release_retreat_success', True)
    if recorder is not None:
        recorder.capture_diagnostic("post_step_h")
        recorder.mark_event("insert_replay_end")
    rospy.set_param('/sawyer_auto_grasp/insertion_replay_stage',
                    "insertion_replay_complete")
    return True


def _init_robot():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("mt3_sawyer_place", anonymous=True)

    rospy.loginfo("INSERT DIAGNOSTIC BUILD: %s", INSERT_DIAG_VERSION)
    rospy.loginfo("Checking current Sawyer robot state...")
    try:
        robot_state = rospy.wait_for_message(
            "/robot/state", RobotAssemblyState, timeout=5.0)
    except rospy.ROSException as exc:
        raise RuntimeError(
            "Failed to receive /robot/state before pick-place: %s" % exc)

    rospy.loginfo(
        "Robot state: ready=%s enabled=%s error=%s stopped=%s homed=%s",
        robot_state.ready, robot_state.enabled,
        robot_state.error, robot_state.stopped,
        getattr(robot_state, "homed", "unknown"))
    if robot_state.error:
        raise RuntimeError("Sawyer reports error=True")
    if robot_state.stopped:
        raise RuntimeError("Sawyer reports stopped=True")
    if not robot_state.ready:
        raise RuntimeError("Sawyer reports ready=False")
    if hasattr(robot_state, "homed") and not robot_state.homed:
        raise RuntimeError("Sawyer is not homed; refusing startup motion")

    if robot_state.enabled:
        rospy.loginfo(
            "Sawyer is already enabled; skipping RobotEnable initialization.")
    else:
        rospy.logwarn(
            "Sawyer is not enabled; attempting RobotEnable fallback.")
        robot_enable = _create_robot_enable_with_retry(
            max_attempts=5,
            retry_delay_s=2.0)
        robot_enable.enable()

    move_group = moveit_commander.MoveGroupCommander(
        PLANNING_GROUP,
        robot_description="%s/robot_description" % ROS_NAMESPACE,
        ns=ROS_NAMESPACE,
    )
    move_group.set_end_effector_link(END_EFFECTOR_LINK)
    install_moveit_timing(move_group)
    move_group.set_pose_reference_frame("base")
    move_group.set_goal_position_tolerance(0.008)
    move_group.set_goal_orientation_tolerance(0.05)
    move_group.set_num_planning_attempts(3)

    gripper = Gripper("right_gripper")
    if not gripper.is_calibrated():
        gripper.calibrate()
        rospy.sleep(1.0)
    gripper.set_cmd_velocity(0.1)
    return move_group, gripper


def execute_pick_place():
    move_group, gripper = _init_robot()
    trajectory_record_path = rospy.get_param(
        "/sawyer_auto_grasp/trajectory_record_path", "")
    use_demo_replay = _param_bool("/sawyer_auto_grasp/use_demo_replay", False)
    use_place_replay = _param_bool(
        "/sawyer_auto_grasp/use_place_release_replay", use_demo_replay)
    demo_replay_path = rospy.get_param(
        "/sawyer_auto_grasp/demo_replay_trajectory_path", "")
    trajectory_rate = float(rospy.get_param(
        "/sawyer_auto_grasp/trajectory_record_rate_hz", 10.0))
    recorder = EndEffectorTrajectoryRecorder(
        move_group, gripper=gripper, rate_hz=trajectory_rate)
    recorder.start()

    # Clear before-close mouth diagnostic params
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_center_xy', ["", ""])
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy', ["", ""])
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_x', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_y', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_x_m', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_y_m', "")
    rospy.set_param('/sawyer_auto_grasp/before_close_mouth_error_xy_m', "")

    # Per-trial replay status. These live on the ROS parameter server so the
    # parent pipeline can log them even if this executor exits with failure.
    for _name, _value in [
            ('/sawyer_auto_grasp/grasp_replay_attempted', False),
            ('/sawyer_auto_grasp/grasp_replay_success', ""),
            ('/sawyer_auto_grasp/grasp_replay_stage', ""),
            ('/sawyer_auto_grasp/grasp_replay_failure_stage', ""),
            ('/sawyer_auto_grasp/insertion_replay_attempted', False),
            ('/sawyer_auto_grasp/insertion_replay_success', ""),
            ('/sawyer_auto_grasp/insertion_replay_stage', ""),
            ('/sawyer_auto_grasp/insertion_replay_failure_stage', ""),
            ('/sawyer_auto_grasp/insertion_interaction_success', False),
            ('/sawyer_auto_grasp/post_release_retreat_attempted', False),
            ('/sawyer_auto_grasp/post_release_retreat_success', ""),
            ('/sawyer_auto_grasp/scripted_fallback_used', False),
            ('/sawyer_auto_grasp/pure_replay_success', False),
            ('/sawyer_auto_grasp/failure_stage_detail', ""),
            ('/sawyer_auto_grasp/replay_failure_stage_detail', ""),
            ('/sawyer_auto_grasp/grasp_post_close_motion_max_m', ""),
            ('/sawyer_auto_grasp/grasp_post_close_motion_max_xy_m', ""),
            ('/sawyer_auto_grasp/grasp_post_close_motion_max_z_m', ""),
            ('/sawyer_auto_grasp/grasp_post_close_mode', ""),
            ('/sawyer_auto_grasp/grasp_post_close_dwell_s', "")]:
        rospy.set_param(_name, _value)

    success = False
    try:
        grasp_x = float(rospy.get_param("/sawyer_auto_grasp/grasp_x"))
        grasp_y = float(rospy.get_param("/sawyer_auto_grasp/grasp_y"))
        grasp_z = float(rospy.get_param("/sawyer_auto_grasp/grasp_z"))
        q = [
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qx", -1.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qy", 0.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qz", 0.0)),
            float(rospy.get_param("/sawyer_auto_grasp/grasp_qw", 0.0)),
        ]
        object_size = rospy.get_param(
            "/sawyer_auto_grasp/object_size", [0.045, 0.045, 0.045])
        object_height = float(object_size[2]) if len(object_size) >= 3 else 0.045

        place_x = float(rospy.get_param("/sawyer_auto_grasp/place_x"))
        place_y = float(rospy.get_param("/sawyer_auto_grasp/place_y"))
        place_z = float(rospy.get_param(
            "/sawyer_auto_grasp/place_z",
            grasp_z + object_height + 0.03))
        place_direction = rospy.get_param(
            "/sawyer_auto_grasp/place_direction", "right")
        insert_slow_descent = str(place_direction) == "insert_into_socket"
        insert_vel = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_descent_velocity_scale", 0.025))
        insert_acc = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_descent_acceleration_scale", 0.025))
        insert_step = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_descent_eef_step", 0.002))
        insert_fallback_step = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_descent_fallback_step_z", 0.006))
        insert_sleep = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_descent_step_sleep", 0.35))
        place_clearance = float(rospy.get_param(
            "/sawyer_auto_grasp/place_clearance", 0.030))
        insert_socket_height = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_socket_height", 0.0))
        insert_release_clearance = float(rospy.get_param(
            "/sawyer_auto_grasp/insert_release_clearance", place_clearance))
        lift_height = float(rospy.get_param(
            "/sawyer_auto_grasp/place_lift_height", 0.150))

        grasp_contact_z = grasp_z
        if str(place_direction) == "insert_into_socket":
            grasp_flange_offset = TOP_FLANGE_Z_OFFSET
            pregrasp_clearance = float(rospy.get_param(
                "/sawyer_auto_grasp/insert_pregrasp_clearance", 0.10))
        else:
            grasp_flange_offset = float(rospy.get_param(
                "/sawyer_auto_grasp/flange_grasp_z_offset",
                TOP_GRASP_FLANGE_Z_OFFSET))
            edge_mode = abs(grasp_y) >= float(rospy.get_param(
                "/sawyer_auto_grasp/edge_y_threshold", 0.08))
            pregrasp_clearance = float(rospy.get_param(
                "/sawyer_auto_grasp/pregrasp_clearance", 0.025))
            if edge_mode:
                pregrasp_clearance += float(rospy.get_param(
                    "/sawyer_auto_grasp/edge_pregrasp_extra", 0.015))
        grasp_flange_z = grasp_contact_z + grasp_flange_offset
        pregrasp_z = grasp_flange_z + pregrasp_clearance
        lift_z = grasp_flange_z + lift_height
        place_above_z = max(place_z + lift_height, lift_z)
        if str(place_direction) == "insert_into_socket" and insert_socket_height > 0.0:
            # For insertion, keep the gripper fingers above the socket rim.
            # The held cylinder extends below the fingers into the socket.
            place_release_z = (
                place_z + insert_socket_height + TOP_FLANGE_Z_OFFSET +
                insert_release_clearance)
        else:
            place_release_z = (
                place_z + object_height + TOP_FLANGE_Z_OFFSET + place_clearance)

        rospy.loginfo("=" * 60)
        rospy.loginfo("MT3 pick-place execution")
        rospy.loginfo("  grasp: [%.3f, %.3f, %.3f]", grasp_x, grasp_y, grasp_z)
        rospy.loginfo(
            "  top grasp flange_z=%.3f offset=%.3f pregrasp_clearance=%.3f",
            grasp_flange_z, grasp_flange_offset, pregrasp_clearance)
        rospy.loginfo(
            "  place: [%.3f, %.3f, %.3f] direction=%s",
            place_x, place_y, place_z, place_direction)
        rospy.loginfo(
            "  scripted-only release_z=%.3f object_height=%.3f clearance=%.3f",
            place_release_z, object_height, place_clearance)
        if str(place_direction) == "insert_into_socket":
            rospy.loginfo(
                "  insert socket height=%.3f release_clearance=%.3f",
                insert_socket_height, insert_release_clearance)
        rospy.loginfo("=" * 60)

        gripper.open()
        rospy.sleep(0.8)

        pregrasp = _make_pose(grasp_x, grasp_y, pregrasp_z, q)
        grasp_pose = _make_pose(grasp_x, grasp_y, grasp_flange_z, q)
        lift_pose = _make_pose(grasp_x, grasp_y, lift_z, q)
        place_above = _make_pose(place_x, place_y, place_above_z, q)
        place_anchor = _make_pose(place_x, place_y, place_z, q)
        place_release = _make_pose(place_x, place_y, place_release_z, q)
        retreat = _make_pose(place_x, place_y, place_above_z, q)

        if (str(place_direction) == "insert_into_socket" and
                not _param_bool("/sawyer_auto_grasp/use_grasp_replay", False)):
            if _param_bool(
                    "/sawyer_auto_grasp/insert_require_grasp_replay", True):
                rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                "grasp_replay_required_but_disabled")
                rospy.logwarn(
                    "Insert task requires grasp replay, but no grasp replay "
                    "trajectory is enabled. Refusing to use legacy scripted "
                    "grasp.")
                return False
            rospy.logwarn(
                "Insert grasp replay disabled; using legacy scripted grasp.")
            if not _go_pose(move_group, pregrasp, "Step A: pregrasp"):
                return False
            if not _cartesian_to_legacy_grasp(
                    move_group, grasp_pose, "Step B: descend to grasp"):
                return False
            rospy.loginfo("Step C: close gripper")
            recorder.mark_event("gripper_close")
            initial_gripper = None
            try:
                initial_gripper = float(gripper.get_position())
            except Exception:
                pass
            gripper.close()
            rospy.sleep(1.5)
            try:
                current_gripper = float(gripper.get_position())
                rospy.loginfo(
                    "  gripper position: %.3f -> %.3f",
                    initial_gripper if initial_gripper is not None else -1.0,
                    current_gripper)
            except Exception:
                pass
            if not _cartesian_to_legacy_grasp(
                    move_group, lift_pose, "Step D: lift object"):
                return False
        else:
            use_grasp_replay = _param_bool(
                "/sawyer_auto_grasp/use_grasp_replay", False)
            grasp_replay_path = rospy.get_param(
                "/sawyer_auto_grasp/grasp_replay_trajectory_path", "")
            if use_grasp_replay and grasp_replay_path:
                rospy.set_param('/sawyer_auto_grasp/grasp_replay_attempted', True)
                rospy.set_param('/sawyer_auto_grasp/grasp_replay_success', False)
                payload = _load_replay_payload(grasp_replay_path)
                if payload:
                    bn_xyz = _replay_position_xyz(
                        payload.get("aligned_bottleneck_pose") or {})
                    if (bn_xyz and _param_bool(
                            "/sawyer_auto_grasp/"
                            "grasp_replay_use_aligned_bottleneck_pose",
                            bool(payload.get(
                                "use_aligned_bottleneck_pose", False)))):
                        bottleneck_xyz = bn_xyz
                        rospy.loginfo(
                            "Grasp replay: using mapped demo bottleneck pose "
                            "from geometry alignment [%.3f, %.3f, %.3f]",
                            bottleneck_xyz[0], bottleneck_xyz[1],
                            bottleneck_xyz[2])
                    else:
                        bottleneck_xyz = None
                    base_xyz, close_xyz, replay_close_idx = (
                        _replay_trajectory_base_close_xyz(payload))
                    if bottleneck_xyz is not None:
                        pass
                    elif base_xyz and close_xyz:
                        desired_close_xyz = [
                            float(grasp_x),
                            float(grasp_y),
                            float(grasp_flange_z),
                        ]
                        bottleneck_xyz = [
                            desired_close_xyz[0] + (base_xyz[0] - close_xyz[0]),
                            desired_close_xyz[1] + (base_xyz[1] - close_xyz[1]),
                            desired_close_xyz[2] + (base_xyz[2] - close_xyz[2]),
                        ]
                        rospy.loginfo(
                            "Grasp replay: runtime-align saved trajectory by "
                            "demo base-close offset [%.3f, %.3f, %.3f] "
                            "close_index=%d desired_close=[%.3f, %.3f, %.3f]",
                            base_xyz[0] - close_xyz[0],
                            base_xyz[1] - close_xyz[1],
                            base_xyz[2] - close_xyz[2],
                            replay_close_idx,
                            desired_close_xyz[0],
                            desired_close_xyz[1],
                            desired_close_xyz[2])
                    else:
                        demo_grasp_xyz = _replay_position_xyz(
                            payload.get("aligned_grasp_pose") or {})
                        if bn_xyz and demo_grasp_xyz:
                            bottleneck_xyz = [
                                grasp_x + (bn_xyz[0] - demo_grasp_xyz[0]),
                                grasp_y + (bn_xyz[1] - demo_grasp_xyz[1]),
                                grasp_z + (bn_xyz[2] - demo_grasp_xyz[2]),
                            ]
                            rospy.loginfo(
                                "Grasp replay: runtime-align bottleneck by demo "
                                "bottleneck-grasp offset [%.3f, %.3f, %.3f]",
                                bn_xyz[0] - demo_grasp_xyz[0],
                                bn_xyz[1] - demo_grasp_xyz[1],
                                bn_xyz[2] - demo_grasp_xyz[2])
                        elif bn_xyz:
                            bottleneck_xyz = bn_xyz
                            rospy.logwarn(
                                "Grasp replay: aligned_grasp_pose missing; using "
                                "demo bottleneck absolute pose")
                        else:
                            bottleneck_xyz = [grasp_x, grasp_y, pregrasp_z]
                            rospy.logwarn(
                                "Grasp replay: bottleneck pose missing; using "
                                "runtime pregrasp as replay bottleneck")
                    bottleneck_pose = _make_pose(
                        bottleneck_xyz[0], bottleneck_xyz[1], bottleneck_xyz[2],
                        q)
                    close_anchor_x = grasp_x + float(rospy.get_param(
                        "/sawyer_auto_grasp/"
                        "grasp_replay_close_anchor_offset_x",
                        -0.034))
                    close_anchor_y = grasp_y + float(rospy.get_param(
                        "/sawyer_auto_grasp/"
                        "grasp_replay_close_anchor_offset_y",
                        0.002))
                    aligned_close_xyz = _replay_position_xyz(
                        payload.get("aligned_grasp_pose") or {})
                    if aligned_close_xyz:
                        close_anchor_z = float(aligned_close_xyz[2])
                        rospy.loginfo(
                            "Grasp replay: using aligned grasp close Z from "
                            "replay payload: %.6f (legacy flange_z=%.6f)",
                            close_anchor_z, grasp_flange_z)
                    else:
                        close_anchor_z = grasp_flange_z
                        rospy.logwarn(
                            "Grasp replay: aligned_grasp_pose missing; "
                            "falling back to legacy grasp_flange_z=%.6f",
                            close_anchor_z)

                    close_anchor = _make_pose(
                        close_anchor_x, close_anchor_y, close_anchor_z, q)
                    rospy.loginfo(
                        "Grasp replay: using mt3_sawyer_grasp.execute_demo_replay "
                        "bottleneck=[%.3f, %.3f, %.3f] "
                        "semantic_grasp=[%.3f, %.3f, %.3f] "
                        "close_anchor=[%.3f, %.3f, %.3f] "
                        "close_anchor_offset=[%.1f, %.1f]cm",
                        bottleneck_pose.position.x, bottleneck_pose.position.y,
                        bottleneck_pose.position.z, grasp_x, grasp_y, grasp_z,
                        close_anchor.position.x, close_anchor.position.y,
                        close_anchor.position.z,
                        (close_anchor_x - grasp_x) * 100.0,
                        (close_anchor_y - grasp_y) * 100.0)
                    target_ori = geometry_msgs.msg.Quaternion()
                    target_ori.x = float(q[0])
                    target_ori.y = float(q[1])
                    target_ori.z = float(q[2])
                    target_ori.w = float(q[3])
                    if str(place_direction).strip().lower() == "insert_into_socket":
                        replay_experiment_group = "vertical_insert"
                    else:
                        replay_experiment_group = "top_grasp"
                    rospy.set_param(
                        "/sawyer_auto_grasp/experiment_group",
                        replay_experiment_group)
                    rospy.set_param(
                        "/sawyer_auto_grasp/top_grasp_unified_execution",
                        replay_experiment_group == "top_grasp")
                    rospy.loginfo(
                        "Grasp replay experiment group: %s unified_top=%s",
                        replay_experiment_group,
                        replay_experiment_group == "top_grasp")
                    rospy.set_param(
                        "/sawyer_auto_grasp/top_replay_anchor_close_waypoint",
                        _param_bool(
                            "/sawyer_auto_grasp/"
                            "grasp_replay_anchor_close_waypoint",
                            False))
                    rospy.set_param(
                        "/sawyer_auto_grasp/top_replay_anchor_close_waypoint_z",
                        _param_bool(
                            "/sawyer_auto_grasp/"
                            "grasp_replay_anchor_close_waypoint_z",
                            True))
                    rospy.set_param(
                        "/sawyer_auto_grasp/use_top_mouth_center_final_correction",
                        _param_bool(
                            "/sawyer_auto_grasp/"
                            "grasp_replay_use_top_mouth_center_final_correction",
                            False))
                    rospy.set_param(
                        "/sawyer_auto_grasp/prefer_pose_replay",
                        _param_bool(
                            "/sawyer_auto_grasp/grasp_replay_prefer_pose_replay",
                            False))
                    rospy.set_param(
                        "/sawyer_auto_grasp/use_segmented_replay",
                        _param_bool(
                            "/sawyer_auto_grasp/grasp_replay_use_segmented_replay",
                            True))
                    rospy.set_param(
                        "/sawyer_auto_grasp/close_on_replay_blocked",
                        _param_bool(
                            "/sawyer_auto_grasp/grasp_replay_close_on_blocked",
                            True))
                    rospy.set_param(
                        "/sawyer_auto_grasp/replay_close_on_blocked_min_progress",
                        float(rospy.get_param(
                            "/sawyer_auto_grasp/"
                            "grasp_replay_close_on_blocked_min_progress",
                            0.35)))
                    grasp_ok = execute_demo_replay(
                        move_group, gripper, bottleneck_pose,
                        target_ori, grasp_replay_path,
                        trajectory_recorder=recorder,
                        trajectory_record_path=trajectory_record_path,
                        close_anchor_pose=close_anchor)
                else:
                    rospy.set_param('/sawyer_auto_grasp/grasp_replay_failure_stage',
                                    "grasp_replay_payload_load")
                    rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                    "grasp_replay_payload_load")
                    grasp_ok = False
                rospy.set_param('/sawyer_auto_grasp/grasp_replay_success',
                                bool(grasp_ok))
                if not grasp_ok:
                    grasp_failure_stage = str(rospy.get_param(
                        '/sawyer_auto_grasp/grasp_replay_failure_stage',
                        'grasp_replay_unknown'))
                    rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                    grasp_failure_stage)
                    rospy.set_param('/sawyer_auto_grasp/replay_failure_stage_detail',
                                    grasp_failure_stage)
                    rospy.logwarn(
                        "Grasp replay failed; scripted fallback disabled, "
                        "marking pick-place failed.")
                    try:
                        gripper.open()
                        rospy.sleep(0.5)
                    except Exception:
                        pass
                    return False
            else:
                grasp_ok, _ = _execute_mt3_top_grasp_core(
                    move_group, gripper, grasp_x, grasp_y, grasp_z,
                    grasp_flange_z, q, object_size,
                    trajectory_recorder=recorder)
                if not grasp_ok:
                    return False

        if recorder is not None and str(place_direction) == "insert_into_socket":
            recorder.capture_diagnostic("grasp_complete_before_transport")

        step_e_target = place_above
        step_e_label = "Step E: move above place"
        if use_place_replay and demo_replay_path:
            mapped_place_bn_pose = _mapped_place_bottleneck_pose_from_replay(
                demo_replay_path, q)
            if mapped_place_bn_pose is not None:
                step_e_target = copy.deepcopy(mapped_place_bn_pose)
                step_e_target.position.z = max(
                    float(mapped_place_bn_pose.position.z) + 0.10,
                    float(place_above_z))
                step_e_label = "Step E: move above mapped place bottleneck"

        if str(place_direction) == "insert_into_socket":
            rospy.loginfo(
                "%s will use staged collision-avoiding carried-object transport",
                step_e_label)
            if not _safe_insert_transport(
                    move_group, step_e_target, recorder=recorder):
                if not rospy.get_param(
                        '/sawyer_auto_grasp/failure_stage_detail', ""):
                    rospy.set_param(
                        '/sawyer_auto_grasp/failure_stage_detail',
                        "transport_to_insert")
                return False
        else:
            if not _go_pose(move_group, step_e_target, step_e_label):
                rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                "transport_to_place")
                return False

        replay_done = False
        if use_place_replay and demo_replay_path:
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_attempted', True)
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_success', False)
            replay_done = _execute_place_release_replay(
                move_group, gripper, demo_replay_path, q,
                place_anchor_pose=place_anchor, recorder=recorder)
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_success',
                            bool(replay_done))
            if not replay_done:
                insertion_failure_stage = str(rospy.get_param(
                    '/sawyer_auto_grasp/insertion_replay_failure_stage',
                    'insertion_replay_unknown'))
                insertion_interaction_success = bool(rospy.get_param(
                    '/sawyer_auto_grasp/insertion_interaction_success', False))
                rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                insertion_failure_stage)
                rospy.set_param('/sawyer_auto_grasp/replay_failure_stage_detail',
                                insertion_failure_stage)
                # Formal replay evaluation rule: never replace a failed replay
                # with a scripted descend/open/retreat sequence.
                rospy.set_param('/sawyer_auto_grasp/scripted_fallback_used', False)
                rospy.set_param('/sawyer_auto_grasp/pure_replay_success', False)

                if (insertion_failure_stage == "insertion_step_h_replay" and
                        insertion_interaction_success):
                    rospy.logwarn(
                        "Insertion/release completed but Step H retreat failed. "
                        "No scripted fallback will be used. Continuing to final "
                        "Gazebo postcheck; pure replay remains failed.")
                else:
                    rospy.logerr(
                        "Insertion/place replay failed before task interaction "
                        "completion; scripted fallback disabled; marking trial "
                        "failed. failure_stage=%s",
                        insertion_failure_stage)
                    return False
        elif use_place_replay:
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_attempted', False)
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_success', "")
            rospy.set_param('/sawyer_auto_grasp/insertion_replay_failure_stage',
                            "insertion_replay_path_missing")
            rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                            "insertion_replay_path_missing")
            rospy.set_param('/sawyer_auto_grasp/replay_failure_stage_detail',
                            "insertion_replay_path_missing")
            rospy.set_param('/sawyer_auto_grasp/scripted_fallback_used', False)
            rospy.set_param('/sawyer_auto_grasp/pure_replay_success', False)
            rospy.logerr(
                "Insertion/place replay requested but replay path is empty; "
                "scripted fallback disabled; marking trial failed.")
            return False

        # Scripted placement remains available only when replay was not
        # requested at all (e.g. an explicit scripted baseline).
        if not use_place_replay:
            rospy.loginfo(
                "Replay not requested; executing explicit scripted "
                "descend-open-retreat baseline.")
            rospy.loginfo("Step F: descend to table release height before opening")
            if insert_slow_descent:
                rospy.loginfo(
                    "Step F: slow insertion descent vel=%.3f acc=%.3f "
                    "eef_step=%.3f fallback_step=%.3f",
                    insert_vel, insert_acc, insert_step, insert_fallback_step)
            if not _cartesian_to(
                    move_group, place_release, "Step F: descend to place",
                    min_fraction=0.80,
                    eef_step=(insert_step if insert_slow_descent else 0.004),
                    velocity_scale=(insert_vel if insert_slow_descent else None),
                    acceleration_scale=(insert_acc if insert_slow_descent else None),
                    fallback_step_z=(
                        insert_fallback_step if insert_slow_descent else 0.012),
                    fallback_sleep=(
                        insert_sleep if insert_slow_descent else 0.15)):
                rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                "scripted_insert_descent")
                return False

            rospy.sleep(0.5)
            rospy.loginfo("Step G: open gripper only after reaching place height")
            recorder.mark_event("place_release_open")
            gripper.open()
            rospy.sleep(1.0)

            if not _cartesian_to(move_group, retreat, "Step H: retreat upward"):
                rospy.set_param('/sawyer_auto_grasp/failure_stage_detail',
                                "scripted_insert_retreat")
                return False

        success = True
        grasp_attempted = bool(rospy.get_param(
            '/sawyer_auto_grasp/grasp_replay_attempted', False))
        grasp_success = bool(rospy.get_param(
            '/sawyer_auto_grasp/grasp_replay_success', False))
        insertion_attempted = bool(rospy.get_param(
            '/sawyer_auto_grasp/insertion_replay_attempted', False))
        insertion_success = bool(rospy.get_param(
            '/sawyer_auto_grasp/insertion_replay_success', False))
        fallback_used = bool(rospy.get_param(
            '/sawyer_auto_grasp/scripted_fallback_used', False))
        pure_replay_success = bool(
            grasp_attempted and grasp_success and
            insertion_attempted and insertion_success and
            not fallback_used)
        rospy.set_param('/sawyer_auto_grasp/pure_replay_success',
                        pure_replay_success)
        if pure_replay_success:
            rospy.set_param('/sawyer_auto_grasp/failure_stage_detail', "")
        rospy.loginfo(
            "Replay status: grasp=%s insertion=%s fallback=%s pure_replay=%s",
            grasp_success, insertion_success, fallback_used, pure_replay_success)
        rospy.loginfo("MT3 pick-place completed successfully")
        return True
    finally:
        if recorder is not None:
            recorder.stop()
            saved = recorder.save(trajectory_record_path, success=success)
            if saved:
                rospy.loginfo(
                    "Pick-place rollout saved: %s success=%s samples=%d",
                    saved, success, len(recorder.samples))
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    ok = False
    try:
        ok = execute_pick_place()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted")
    except Exception as exc:
        import traceback
        rospy.logerr("Pick-place execution failed: %s", exc)
        rospy.logerr("Full traceback:\n%s", traceback.format_exc())
    if not ok:
        sys.exit(1)
