#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT3 Pick-and-Place Demo Recording
执行逻辑 = mt3_sawyer_place.py 原封不动
新增: 命令行设参 (兼容 /sawyer_auto_grasp) + Bottleneck RGB+Depth + Demo JSON + 场景包

用法:
    python3 record_place_demo.py \
        _object_x:=0.60 _object_y:=0.0 _object_z:=-0.58 \
        _object_size:="[0.045, 0.045, 0.045]" \
        _place_direction:=left \
        _demo_name:=cube_pick_place_left
"""

import copy, json, math, os, sys, threading, time
import geometry_msgs.msg, moveit_commander, rospy
import numpy as np
import tf2_ros
from intera_interface import Gripper, RobotEnable
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mt3_scene_package import save_scene_package

ROS_NAMESPACE = "/robot"
PLANNING_GROUP = "right_arm"
END_EFFECTOR_LINK = "right_hand"

ORI_VEL_SCALE = 0.25
ORI_ACC_SCALE = 0.25
DOWN_VEL_SCALE = 0.08
DOWN_ACC_SCALE = 0.08
CART_STEP = 0.006
TOP_FLANGE_Z_OFFSET = 0.050

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "demo_library", "recorded")

# ══════════════════════════════════════════════════════════════
# DemoRecorder — mt3_recorded_v2 格式
# ══════════════════════════════════════════════════════════════

class DemoRecorder(object):
    def __init__(self, move_group, gripper, rate_hz=30.0):
        self.move_group = move_group
        self.gripper = gripper
        self.rate_hz = float(rate_hz)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.gripper_command_state = None
        # TF 仅用于离散快照, 不在录制线程里用, 按需创建
        self._tf_buffer = None
        self._tf_listener = None

    def _ensure_tf(self):
        if self._tf_buffer is None:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self._stop.is_set():
            try:
                pose = self.move_group.get_current_pose().pose
                gp = None
                if self.gripper is not None:
                    try:
                        gp = float(self.gripper.get_position())
                    except Exception:
                        pass
                self.samples.append({
                    "timestamp": float(rospy.get_time()),
                    "position": [float(pose.position.x), float(pose.position.y),
                                 float(pose.position.z)],
                    "orientation": [float(pose.orientation.x), float(pose.orientation.y),
                                    float(pose.orientation.z), float(pose.orientation.w)],
                    "gripper_position": gp,
                    "gripper_state": self.gripper_command_state,
                    "gripper_next": self.gripper_command_state,
                })
            except Exception:
                pass
            rate.sleep()

    def _get_ee_pose_tf(self, retries=3, delay=0.3):
        self._ensure_tf()
        for attempt in range(retries):
            try:
                t = self._tf_buffer.lookup_transform(
                    "base", "right_hand", rospy.Time(0), rospy.Duration(2.0))
                return {
                    "position": [t.transform.translation.x, t.transform.translation.y,
                                 t.transform.translation.z],
                    "orientation": [t.transform.rotation.x, t.transform.rotation.y,
                                    t.transform.rotation.z, t.transform.rotation.w],
                    "timestamp": t.header.stamp.to_sec(),
                }
            except Exception:
                if attempt < retries - 1:
                    rospy.sleep(delay)
        return None

    def _rotate_by_quat(self, q, v):
        x, y, z, w = q[0], q[1], q[2], q[3]
        vx, vy, vz = v[0], v[1], v[2]
        return [(1-2*y*y-2*z*z)*vx + (2*x*y-2*w*z)*vy + (2*x*z+2*w*y)*vz,
                (2*x*y+2*w*z)*vx + (1-2*x*x-2*z*z)*vy + (2*y*z-2*w*x)*vz,
                (2*x*z-2*w*y)*vx + (2*y*z+2*w*x)*vy + (1-2*x*x-2*y*y)*vz]

    def _quat_delta_to_angular_velocity(self, q0, q1, dt):
        if dt <= 0:
            return [0.0, 0.0, 0.0]
        dq = [q0[0]*q1[3]+q0[3]*q1[0]+q0[1]*q1[2]-q0[2]*q1[1],
              q0[3]*q1[1]-q0[0]*q1[2]+q0[1]*q1[3]+q0[2]*q1[0],
              q0[3]*q1[2]+q0[0]*q1[1]-q0[1]*q1[0]+q0[2]*q1[3],
              q0[3]*q1[3]-q0[0]*q1[0]-q0[1]*q1[1]-q0[2]*q1[2]]
        norm = math.sqrt(sum(float(v)*float(v) for v in dq))
        if norm <= 1e-8:
            return [0.0, 0.0, 0.0]
        dq = [float(v)/norm for v in dq]
        if dq[3] < 0:
            dq = [-dq[0], -dq[1], -dq[2], -dq[3]]
        axis_norm = math.sqrt(dq[0]*dq[0]+dq[1]*dq[1]+dq[2]*dq[2])
        if axis_norm <= 1e-8:
            return [0.0, 0.0, 0.0]
        angle = 2.0*math.atan2(axis_norm, max(-1.0, min(1.0, dq[3])))
        axis = [dq[0]/axis_norm, dq[1]/axis_norm, dq[2]/axis_norm]
        return [float(axis[j]*angle/dt) for j in range(3)]

    def to_demo(self, demo_name, object_pos, object_size, place_dir,
                place_x, place_y, place_z, bottleneck_ee, grasp_ee):
        seen = set()
        unique = []
        for s in self.samples:
            ts = round(s.get("timestamp", 0), 4)
            if ts not in seen:
                seen.add(ts)
                unique.append(s)

        velocities = []
        for i in range(1, len(unique)):
            dt = unique[i]["timestamp"] - unique[i-1]["timestamp"]
            if dt <= 0:
                continue
            p0 = np.array(unique[i-1]["position"])
            p1 = np.array(unique[i]["position"])
            dp = (p1 - p0) / dt
            q = unique[i-1]["orientation"]
            v_ee = self._rotate_by_quat([-q[0], -q[1], -q[2], q[3]], dp.tolist())
            gn = unique[i].get("gripper_next", unique[i-1].get("gripper_state"))
            q0 = unique[i-1]["orientation"]
            q1 = unique[i]["orientation"]
            w_ee = self._quat_delta_to_angular_velocity(q0, q1, dt)
            w_world = self._rotate_by_quat(q0, w_ee)
            velocities.append({
                "timestamp": unique[i]["timestamp"], "dt": float(dt),
                "position": unique[i]["position"],
                "orientation": unique[i]["orientation"],
                "linear_ee": v_ee, "linear_world": dp.tolist(),
                "angular_ee": w_ee, "angular_world": w_world,
                "gripper_position": unique[i].get("gripper_position"),
                "gripper_state": unique[i].get("gripper_state"),
                "gripper_next": gn,
            })

        return {
            "id": demo_name, "format": "mt3_recorded_v2",
            "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": "pick_place",
            "task": f"pick and place to {place_dir}",
            "object_info": {
                "position_base": object_pos, "size_m": object_size,
                "category": "cube", "color": "green",
            },
            "place_info": {
                "direction": place_dir,
                "place_pose_base_frame": {"position": [place_x, place_y, place_z]},
            },
            "bottleneck_pose_base_frame": {
                "position_m": {"x": bottleneck_ee["position"][0],
                               "y": bottleneck_ee["position"][1],
                               "z": bottleneck_ee["position"][2]},
                "orientation_xyzw": {"x": bottleneck_ee["orientation"][0],
                                     "y": bottleneck_ee["orientation"][1],
                                     "z": bottleneck_ee["orientation"][2],
                                     "w": bottleneck_ee["orientation"][3]},
                "timestamp": bottleneck_ee.get("timestamp", 0),
            },
            "grasp_pose_base_frame": {
                "position_m": {"x": grasp_ee["position"][0] if grasp_ee else 0,
                               "y": grasp_ee["position"][1] if grasp_ee else 0,
                               "z": grasp_ee["position"][2] if grasp_ee else 0},
            },
            "trajectory": {
                "format": "end_effector_pose_twist_gripper_v2",
                "frame": "end_effector", "pose_frame": "base",
                "sample_rate_hz": self.rate_hz,
                "num_waypoints": len(velocities),
                "poses": unique, "velocities": velocities,
                "gripper_convention": "gripper_next: 1=close, 0=open, null=unknown",
            },
            "language_tags": ["pick and place", place_dir,
                            "cube", "green cube", "top-down grasp", "抓取", "放置"],
            "language_description": f"Pick cube and place to {place_dir}",
            "approach_direction": [0.0, 0.0, -1.0],
            "retract_direction": [0.0, 0.0, 1.0],
            "gripper_opening_m": 0.07,
        }

    def save_demo(self, path, demo_dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(demo_dict, f, indent=2, ensure_ascii=False)
        return path


# ══════════════════════════════════════════════════════════════
# mt3_sawyer_place.py 原版函数 (一字不改)
# ══════════════════════════════════════════════════════════════

def _make_pose(x, y, z, q):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
    pose.orientation.x = float(q[0]); pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2]); pose.orientation.w = float(q[3])
    return pose


def _go_pose(move_group, pose, label, velocity=ORI_VEL_SCALE,
             acceleration=ORI_ACC_SCALE, attempts=3, planning_time=8.0):
    rospy.loginfo("%s: [%.3f, %.3f, %.3f]", label,
                  pose.position.x, pose.position.y, pose.position.z)
    move_group.set_max_velocity_scaling_factor(float(velocity))
    move_group.set_max_acceleration_scaling_factor(float(acceleration))
    move_group.set_planning_time(float(planning_time))
    for attempt in range(attempts):
        move_group.set_pose_target(pose)
        plan_result = move_group.plan()
        if bool(plan_result[0]):
            move_group.execute(plan_result[1], wait=True)
            rospy.sleep(0.4)
            move_group.stop(); move_group.clear_pose_targets()
            return True
        rospy.logwarn("%s retry %d/%d", label, attempt + 1, attempts)
    move_group.clear_pose_targets()
    rospy.logerr("%s failed", label)
    return False


def _cartesian_to(move_group, pose, label, min_fraction=0.90, eef_step=CART_STEP):
    start = move_group.get_current_pose().pose
    plan, fraction = move_group.compute_cartesian_path(
        [start, pose], float(eef_step), True)
    rospy.loginfo("%s cartesian: %.1f%%", label, fraction * 100.0)
    if fraction < min_fraction or not plan.joint_trajectory.points:
        rospy.logwarn("%s cartesian insufficient; pose fallback", label)
        return _go_pose(move_group, pose, label + "-fallback",
                        velocity=DOWN_VEL_SCALE, acceleration=DOWN_ACC_SCALE,
                        attempts=2, planning_time=6.0)
    move_group.execute(plan, wait=True)
    rospy.sleep(0.4)
    return True


def _green_mask_from_bgr(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 55, 55), (80, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask.astype(bool)


# ══════════════════════════════════════════════════════════════
# 主流程: mt3_sawyer_place.py 执行 + Demo 录制
# ══════════════════════════════════════════════════════════════

def _load_langsam_mask(mask_path, rgb_shape=None):
    if not mask_path or not os.path.exists(mask_path):
        return None
    try:
        mask = np.load(mask_path).astype(bool)
        if rgb_shape is not None and mask.shape != rgb_shape:
            rospy.logwarn("  LangSAM mask shape %s != RGB %s", mask.shape, rgb_shape)
            return None
        rospy.loginfo("  LangSAM mask loaded: %s pixels=%d", mask_path, int(np.count_nonzero(mask)))
        return mask
    except Exception as exc:
        rospy.logwarn("  LangSAM mask load failed: %s", exc)
        return None


def execute_and_record(obj_x, obj_y, obj_z, obj_size, place_dir, demo_name,
                       mask_path=""):
    # ── 初始化 (同 mt3_sawyer_place._init_robot) ──
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("mt3_record_place_demo", anonymous=True)

    try:
        rs = RobotEnable(); rs.enable()
    except Exception:
        pass

    move_group = moveit_commander.MoveGroupCommander(
        PLANNING_GROUP,
        robot_description="%s/robot_description" % ROS_NAMESPACE, ns=ROS_NAMESPACE)
    move_group.set_end_effector_link(END_EFFECTOR_LINK)
    move_group.set_pose_reference_frame("base")
    move_group.set_goal_position_tolerance(0.008)
    move_group.set_goal_orientation_tolerance(0.05)
    move_group.set_num_planning_attempts(3)

    gripper = Gripper("right_gripper")
    if not gripper.is_calibrated():
        gripper.calibrate(); rospy.sleep(1.0)
    gripper.set_cmd_velocity(0.1)

    bridge = CvBridge()

    # ── 计算目标 (同 mt3_sawyer_place.execute_pick_place) ──
    grasp_x = obj_x; grasp_y = obj_y
    object_height = float(obj_size[2]) if len(obj_size) >= 3 else 0.045
    # grasp_z = 物体顶面 + 5mm (指尖接触高度, 同 pipeline 设的 /sawyer_auto_grasp/grasp_z)
    grasp_z = obj_z + object_height + 0.005
    q = [-1.0, 0.0, 0.0, 0.0]

    # 放置目标
    PLACE_OFFSETS = {
        "left": (0.0, +0.18), "right": (0.0, -0.18),
        "front": (+0.15, 0.0), "back": (-0.15, 0.0),
    }
    off = PLACE_OFFSETS.get(place_dir, PLACE_OFFSETS["left"])
    place_x = obj_x + off[0]
    place_y = obj_y + off[1]
    place_z = obj_z  # 放置面 = 桌面
    place_clearance = 0.030
    lift_height = 0.150

    grasp_contact_z = grasp_z
    grasp_flange_z = grasp_contact_z + TOP_FLANGE_Z_OFFSET
    pregrasp_z = grasp_flange_z + 0.10
    lift_z = grasp_flange_z + lift_height
    place_above_z = max(place_z + lift_height, lift_z)
    place_release_z = place_z + object_height + TOP_FLANGE_Z_OFFSET + place_clearance

    rospy.loginfo("=" * 60)
    rospy.loginfo("MT3 Pick-Place Demo: %s", demo_name)
    rospy.loginfo("  grasp: [%.3f, %.3f, %.3f]", grasp_x, grasp_y, grasp_z)
    rospy.loginfo("  place: [%.3f, %.3f, %.3f] direction=%s",
                  place_x, place_y, place_z, place_dir)
    rospy.loginfo("  release_z=%.3f object_height=%.3f clearance=%.3f",
                  place_release_z, object_height, place_clearance)
    rospy.loginfo("=" * 60)

    # ── 录制器 ──
    recorder = DemoRecorder(move_group, gripper, rate_hz=30.0)
    recorder.start()

    success = False
    bottleneck_ee = None
    grasp_ee = None
    bottleneck_rgb = None
    bottleneck_depth = None

    try:
        gripper.open()
        rospy.sleep(0.8)
        recorder.gripper_command_state = 0

        pregrasp = _make_pose(grasp_x, grasp_y, pregrasp_z, q)
        grasp_pose = _make_pose(grasp_x, grasp_y, grasp_flange_z, q)
        lift_pose = _make_pose(grasp_x, grasp_y, lift_z, q)
        place_above = _make_pose(place_x, place_y, place_above_z, q)
        place_release = _make_pose(place_x, place_y, place_release_z, q)
        retreat = _make_pose(place_x, place_y, place_above_z, q)

        # Step A: pregrasp
        if not _go_pose(move_group, pregrasp, "Step A: pregrasp"):
            return False

        # ── Bottleneck capture ──
        rospy.loginfo("Bottleneck capture (RGB + Depth)...")
        try:
            rgb_msg = rospy.wait_for_message(
                "/io/internal_camera/head_camera/image_raw", Image, timeout=5.0)
            bottleneck_rgb = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rospy.loginfo("  RGB: %s", bottleneck_rgb.shape)
        except Exception:
            rospy.logwarn("  RGB capture failed")
        try:
            depth_msg = rospy.wait_for_message(
                "/io/internal_camera/head_camera/depth/image_raw", Image, timeout=5.0)
            bottleneck_depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            rospy.loginfo("  Depth: %s", bottleneck_depth.shape)
        except Exception:
            rospy.logwarn("  Depth capture failed")
        bottleneck_ee = recorder._get_ee_pose_tf()
        if bottleneck_ee is not None:
            rospy.loginfo("  Bottleneck EE: [%.4f, %.4f, %.4f]",
                          bottleneck_ee["position"][0],
                          bottleneck_ee["position"][1],
                          bottleneck_ee["position"][2])

        # Step B: descend to grasp
        if not _cartesian_to(move_group, grasp_pose, "Step B: descend to grasp"):
            return False

        # Step C: close gripper
        rospy.loginfo("Step C: close gripper")
        recorder.gripper_command_state = 1
        gripper.close()
        rospy.sleep(1.5)
        grasp_ee = recorder._get_ee_pose_tf()

        # Step D: lift
        if not _cartesian_to(move_group, lift_pose, "Step D: lift object"):
            return False

        # Step E: transport
        if not _go_pose(move_group, place_above, "Step E: move above place"):
            return False

        # Step F: descend to place
        rospy.loginfo("Step F: descend to table release height")
        if not _cartesian_to(move_group, place_release, "Step F: descend to place",
                             min_fraction=0.80, eef_step=0.004):
            return False

        rospy.sleep(0.5)

        # Step G: open gripper
        rospy.loginfo("Step G: open gripper")
        recorder.gripper_command_state = 0
        gripper.open()
        rospy.sleep(1.0)

        # Step H: retreat
        if not _cartesian_to(move_group, retreat, "Step H: retreat upward"):
            return False

        # Restore normal velocity/acceleration scaling
        move_group.set_max_velocity_scaling_factor(0.6)
        move_group.set_max_acceleration_scaling_factor(0.6)

        success = True
        rospy.loginfo("Pick-place completed!")
        return True

    finally:
        recorder.stop()
        rospy.loginfo("Recording stopped: %d samples", len(recorder.samples))

        if bottleneck_ee is not None:
            demo = recorder.to_demo(
                demo_name,
                [obj_x, obj_y, obj_z], obj_size, place_dir,
                place_x, place_y, place_z,
                bottleneck_ee, grasp_ee)
            json_path = os.path.join(OUTPUT_DIR, f"{demo_name}.json")
            recorder.save_demo(json_path, demo)
            rospy.loginfo("Demo saved: %s (success=%s, %d samples)",
                          json_path, success, len(recorder.samples))

            if bottleneck_rgb is not None:
                import cv2
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{demo_name}_bottleneck_rgb.png"),
                            bottleneck_rgb)

            if bottleneck_rgb is not None and bottleneck_depth is not None:
                try:
                    import cv2
                    rgb = cv2.cvtColor(bottleneck_rgb, cv2.COLOR_BGR2RGB)
                    # Prefer LangSAM mask, fall back to HSV green filter
                    mask = _load_langsam_mask(mask_path, rgb_shape=rgb.shape[:2])
                    if mask is None:
                        mask = _green_mask_from_bgr(bottleneck_rgb)
                        rospy.logwarn("  Falling back to HSV mask: pixels=%d",
                                      int(np.count_nonzero(mask)) if mask is not None else 0)
                    scene_data = {
                        "rgb": rgb, "depth": bottleneck_depth,
                        "segmap": mask,
                        "intrinsics": np.array([
                            [407.391526, 0.0, 640.5],
                            [0.0, 407.391526, 400.5],
                            [0.0, 0.0, 1.0]], dtype=np.float64),
                        "pose": {"position": bottleneck_ee["position"],
                                 "orientation": bottleneck_ee["orientation"],
                                 "method": "recorded_bottleneck_pose", "confidence": 1.0},
                    }
                    pkg_root = os.path.join(os.path.dirname(OUTPUT_DIR), "scene_packages")
                    pkg = save_scene_package(scene_data, pkg_root,
                                          name=f"demo_{demo_name}", role="recorded_demo",
                                          extra_metadata={"demo_id": demo_name,
                                                         "object_position_base": [obj_x, obj_y, obj_z],
                                                         "object_size": obj_size,
                                                         "object_shape": "cube",
                                                         "object_label": "green_cube",
                                                         "mask_source": mask_path or "hsv_fallback"})
                    rospy.loginfo("  Scene package: %s", pkg["package_dir"])
                    rospy.loginfo("  mask_px=%d  points=%d",
                                  pkg["stats"]["segmap_pixels"],
                                  pkg["stats"]["pointcloud_points"])
                except Exception as exc:
                    rospy.logwarn("  Scene package save skipped: %s", exc)
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    obj_x = rospy.get_param("~object_x", 0.60)
    obj_y = rospy.get_param("~object_y", 0.00)
    obj_z = rospy.get_param("~object_z", -0.58)
    obj_size = rospy.get_param("~object_size", [0.045, 0.045, 0.045])
    demo_name = rospy.get_param("~demo_name", "cube_pick_place_left")
    place_dir = rospy.get_param("~place_direction", "left")
    mask_path = rospy.get_param("~langsam_mask_path", "/mnt/hgfs2/tmp_vision/current_mask.npy")

    ok = False
    try:
        ok = execute_and_record(obj_x, obj_y, obj_z, obj_size, place_dir, demo_name,
                               mask_path=mask_path)
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted")
    except Exception as exc:
        rospy.logerr("Failed: %s", exc)
        import traceback; traceback.print_exc()
    if not ok:
        sys.exit(1)
