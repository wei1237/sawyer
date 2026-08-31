#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import copy
import moveit_commander
import geometry_msgs.msg
from intera_interface import Gripper, RobotEnable
import moveit_msgs.msg
import subprocess
from moveit_msgs.msg import OrientationConstraint, Constraints

# ===== 录制用 imports =====
import os
import json
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

OBJECT_BASE_X = 0.6
OBJECT_BASE_Y = 0.0
OBJECT_BASE_Z = -0.58

ROS_NAMESPACE = "/robot"
PLANNING_GROUP = "right_arm"
END_EFFECTOR_LINK = "right_hand"

OVERHEAD_BASE_Z = OBJECT_BASE_Z + 0.045 + 0.15   # 物块顶部 + 15cm
GRASP_CONTACT_BASE_Z = OBJECT_BASE_Z + 0.045 + 0.005  # 物块顶部 + 5mm（抓取/指尖语义高度）
CONTACT_TO_FLANGE_Z_OFFSET = 0.040  # right_hand法兰相对抓取接触点的固定几何补偿
GRASP_TARGET_BASE_Z = GRASP_CONTACT_BASE_Z + CONTACT_TO_FLANGE_Z_OFFSET
TRANSITION_X = 0.5
TRANSITION_BASE_Z = OBJECT_BASE_Z + 0.6


FINGER_LENGTH = 0.03  # 夹爪指尖长度
ALLOWED_ERROR = 0.002 # XY对齐误差阈值
GRASP_XY_TOLERANCE = 0.012  # 夹爪可容忍的抓取XY误差，避免微调过多触发控制失败
MAX_RETRY = 10        # 对齐最大重试次数
CART_STEP = 0.01      # 笛卡尔步长
CART_VEL_SCALE = 0.1  # 笛卡尔执行速度
DEFAULT_CAMERA_INTRINSICS = np.array([
    [554.254691, 0.0, 640.0],
    [0.0, 554.254691, 400.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ===== 录制配置 =====
RECORD_RATE = 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..",
                                           "code", "learning_thousand_tasks",
                                           "demo_library", "recorded"))

# Sawyer官方关节物理限位
JOINT_LIMITS = {
    'right_j0': (-3.05, 3.05),
    'right_j1': (-1.92, 1.396),
    'right_j2': (-3.05, 3.05),
    'right_j3': (-3.05, 3.05),
    'right_j4': (-3.05, 3.05),
    'right_j5': (-3.05, 3.05),
    'right_j6': (-5.23, 5.23)
}

# 关节限位修正函数
def clamp_joint_value(joint_name, value):
    if joint_name not in JOINT_LIMITS:
        return value
    min_val, max_val = JOINT_LIMITS[joint_name]
    clamped = max(min_val, min(value, max_val))
    if abs(value - clamped) > 0.01:
        rospy.logwarn(f"关节{joint_name}超界修正：{value:.3f}→{clamped:.3f}")
    return clamped


# ===== 录制辅助函数 =====
def _rotate_vector_by_quat(q, v):
    x, y, z, w = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    rx = (1-2*y*y-2*z*z)*vx + (2*x*y-2*w*z)*vy + (2*x*z+2*w*y)*vz
    ry = (2*x*y+2*w*z)*vx + (1-2*x*x-2*z*z)*vy + (2*y*z-2*w*x)*vz
    rz = (2*x*z-2*w*y)*vx + (2*y*z+2*w*x)*vy + (1-2*x*x-2*y*y)*vz
    return [rx, ry, rz]


def _poses_to_velocities(poses):
    """将末端位姿序列转换为 (T,7) twists：[vx,vy,vz, wx,wy,wz, gripper]"""
    velocities = []
    for i in range(1, len(poses)):
        dt = poses[i]["timestamp"] - poses[i-1]["timestamp"]
        if dt <= 0:
            continue
        p0 = np.array(poses[i-1]["position"])
        p1 = np.array(poses[i]["position"])
        dp_world = (p1 - p0) / dt
        q = poses[i-1]["orientation"]
        q_conj = [-q[0], -q[1], -q[2], q[3]]
        v_ee = _rotate_vector_by_quat(q_conj, dp_world.tolist())
        q0 = R.from_quat(poses[i-1]["orientation"])
        q1 = R.from_quat(poses[i]["orientation"])
        w_ee = (q0.inv() * q1).as_rotvec() / dt
        w_ee = w_ee.tolist()
        # 夹爪状态：位置>5mm视为打开(1)，否则闭合(0)
        gp = poses[i].get("gripper_position", 0.07)
        gripper_state = 1.0 if gp > 0.005 else 0.0
        velocities.append({
            "timestamp": poses[i]["timestamp"],
            "twist": v_ee + w_ee + [gripper_state],  # 7维
        })
    return velocities


def _save_recorded_demo(demo_name, recorded_poses, bottleneck_rgb, bottleneck_depth,
                        bottleneck_ee, camera_intrinsics, object_pos, object_size):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 去重
    seen_ts = set()
    unique_poses = []
    for p in recorded_poses:
        ts = round(p.get("timestamp", 0), 4)
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique_poses.append(p)

    velocities = _poses_to_velocities(unique_poses)

    # ---- 构建 4×4 bottleneck_pose (SE3) ----
    pos = bottleneck_ee["position"]
    ori = bottleneck_ee["orientation"]  # [x,y,z,w]
    from scipy.spatial.transform import Rotation as R
    bn_pose = np.eye(4)
    bn_pose[:3, :3] = R.from_quat(ori).as_matrix()
    bn_pose[:3, 3] = pos
    bottleneck_pose_44 = bn_pose

    # ---- 构建 (T,7) twists 数组 ----
    twists_list = [v["twist"] for v in velocities]
    twists_array = np.array(twists_list, dtype=np.float64) if twists_list else np.empty((0, 7))

    # ---- 保存 JSON（兼容现有 DemoLibrary） ----
    demo = {
        "id": demo_name,
        "format": "mt3_recorded_v1",
        "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object_info": {
            "position_base": list(object_pos),
            "size_m": list(object_size),
            "category": "cube",
            "color": "green",
        },
        "bottleneck_pose_base_frame": {
            "position_m": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation_xyzw": {"x": ori[0], "y": ori[1], "z": ori[2], "w": ori[3]},
            "timestamp": bottleneck_ee.get("timestamp", 0),
        },
        "trajectory": {
            "format": "end_effector_twist",
            "frame": "end_effector",
            "num_waypoints": len(velocities),
            "velocities": velocities,  # 每个含 twist:[vx,vy,vz,wx,wy,wz,gripper]
        },
        "language_tags": ["grasp", "pick up", "cube", "green cube",
                         "top-down grasp", "抓取", "正方体", "绿色方块"],
        "language_description": "Pick up the green cube from above",
        "approach_direction": [0.0, 0.0, -1.0],
        "retract_direction": [0.0, 0.0, 1.0],
        "gripper_opening_m": 0.07,
    }
    json_path = os.path.join(OUTPUT_DIR, f"{demo_name}.json")
    with open(json_path, "w") as f:
        json.dump(demo, f, indent=2)
    rospy.loginfo(f"[Record] JSON saved: {json_path}")

    # ---- 保存官方 MT3 格式文件 ----
    # bottleneck_pose.npy (4×4 SE3)
    np.save(os.path.join(OUTPUT_DIR, f"{demo_name}_bottleneck_pose.npy"), bottleneck_pose_44)
    rospy.loginfo(f"[Record] bottleneck_pose.npy (4×4): {bottleneck_pose_44.shape}")

    # demo_eef_twists.npy (T, 7)
    np.save(os.path.join(OUTPUT_DIR, f"{demo_name}_demo_eef_twists.npy"), twists_array)
    rospy.loginfo(f"[Record] demo_eef_twists.npy: {twists_array.shape} (T,7)")

    # task_name.txt
    task_path = os.path.join(OUTPUT_DIR, f"{demo_name}_task_name.txt")
    with open(task_path, "w") as f:
        f.write("pick_up_green_cube")
    rospy.loginfo(f"[Record] task_name.txt saved")

    # bottleneck RGB
    if bottleneck_rgb is not None:
        rgb_path = os.path.join(OUTPUT_DIR, f"{demo_name}_head_camera_ws_rgb.png")
        cv2.imwrite(rgb_path, bottleneck_rgb)
        rospy.loginfo(f"[Record] ws_rgb.png: {bottleneck_rgb.shape}")

    # bottleneck depth (毫米, uint16)
    if bottleneck_depth is not None:
        depth_path = os.path.join(OUTPUT_DIR, f"{demo_name}_head_camera_ws_depth_to_rgb.png")
        cv2.imwrite(depth_path, bottleneck_depth)
        rospy.loginfo(f"[Record] ws_depth.png: {bottleneck_depth.shape} {bottleneck_depth.dtype}")

    # camera intrinsics (3×3)
    if camera_intrinsics is not None:
        intrinsics_path = os.path.join(OUTPUT_DIR, f"{demo_name}_head_camera_rgb_intrinsic_matrix.npy")
        np.save(intrinsics_path, camera_intrinsics)
        rospy.loginfo(f"[Record] intrinsics.npy (3×3):\n{camera_intrinsics}")

    rospy.loginfo(f"[Record] Trajectory: {len(velocities)} waypoints × 7 dims")


def wait_for_robot_move_group(timeout=60):
    rospy.loginfo(f"等待{ROS_NAMESPACE}/move_group节点启动...")
    start_time = rospy.get_time()
    while rospy.get_time() - start_time < timeout:
        try:
            result = subprocess.check_output(['rosnode', 'list'], stderr=subprocess.STDOUT)
            if f'{ROS_NAMESPACE}/move_group' in result.decode('utf-8'):
                rospy.loginfo(f" {ROS_NAMESPACE}/move_group节点已启动")
                return True
        except subprocess.CalledProcessError as e:
            rospy.logwarn(f"查询节点失败：{e.output}")
        rospy.sleep(1)
    rospy.logerr(f"未找到{ROS_NAMESPACE}/move_group节点，请先启动：\nroslaunch sawyer_moveit_config demo.launch electric_gripper:=true use_sim_time:=true")
    return False

# 核心抓取主函数
def auto_grasp_with_moveit():
    # 初始化MoveIt和ROS节点
    moveit_commander.roscpp_initialize([])
    rospy.init_node('sawyer_auto_grasp', anonymous=True)
    gripper = None
    robot_enabled = False
    # 定义全局速度缩放因子
    ORI_VEL_SCALE = 0.3
    ORI_ACC_SCALE = 0.3
    DOWN_VEL_SCALE = 0.1
    DOWN_ACC_SCALE = 0.1

    # ===== 录制参数 =====
    demo_name = rospy.get_param("~demo_name", "cube_top_grasp_recorded")
    rospy.loginfo(f"[Record] Demo name: {demo_name}")
    rospy.loginfo(f"[Record] Output dir: {OUTPUT_DIR}")

    # ===== 录制状态 =====
    recording = False
    recorded_poses = []
    bottleneck_rgb = None
    bottleneck_depth = None
    bottleneck_ee = None
    camera_intrinsics = None
    bridge = CvBridge()

    # 初始化 TF
    try:
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.sleep(0.5)
        rospy.loginfo("[Record] TF listener initialized")
    except Exception as e:
        rospy.logwarn(f"[Record] TF init failed: {e}")
        tf_buffer = None

    def _get_ee_pose():
        try:
            # 尝试多个 frame 名（兼容带/不带 namespace）
            if tf_buffer is not None:
                for (src, dst) in [("base", "right_hand"),
                                  ("robot/base", "robot/right_hand"),
                                  ("world", "right_hand"),
                                  ("base", "right_gripper_tip"),
                                  ("robot/base", "robot/right_gripper_tip")]:
                    try:
                        tf = tf_buffer.lookup_transform(src, dst, rospy.Time(0), rospy.Duration(0.3))
                        pos = tf.transform.translation
                        ori = tf.transform.rotation
                        return {
                            "position": [pos.x, pos.y, pos.z],
                            "orientation": [ori.x, ori.y, ori.z, ori.w],
                            "timestamp": tf.header.stamp.to_sec()
                        }
                    except Exception:
                        continue
            try:
                pose_msg = move_group.get_current_pose().pose
                pos = pose_msg.position
                ori = pose_msg.orientation
                return {
                    "position": [pos.x, pos.y, pos.z],
                    "orientation": [ori.x, ori.y, ori.z, ori.w],
                    "timestamp": rospy.get_time()
                }
            except Exception:
                pass
            return None
        except Exception:
            return None

    def _record_thread():
        rate = rospy.Rate(RECORD_RATE)
        while recording:
            pose = _get_ee_pose()
            if pose:
                # 同时记录夹爪位置
                try:
                    pose["gripper_position"] = gripper.get_position() if gripper else 0.07
                except Exception:
                    pose["gripper_position"] = 0.07
                recorded_poses.append(pose)
            rate.sleep()

    # ====================== 1. 初始化======================
    rospy.loginfo("=== 初始化MoveGroup连接 ===")
    if not wait_for_robot_move_group(timeout=60):
        return gripper, robot_enabled

    # 等待规划场景服务就绪
    service_name = f'{ROS_NAMESPACE}/get_planning_scene'
    rospy.loginfo(f"等待服务{service_name}就绪...")
    try:
        rospy.wait_for_service(service_name, timeout=60)
        rospy.loginfo(f"服务{service_name}已就绪")
    except rospy.ROSException:
        rospy.logerr(f" 超时！未找到服务{service_name}")
        return gripper, robot_enabled

    # 校验SRDF参数
    semantic_param = f"{ROS_NAMESPACE}/robot_description_semantic"
    if not rospy.has_param(semantic_param):
        rospy.logwarn(f"未找到{semantic_param}，可忽略，继续执行...")
    else:
        rospy.loginfo(" SRDF参数加载成功")

    # 机器人使能
    try:
        rs = RobotEnable()
        if not rs.state().enabled:
            rs.enable()
        robot_enabled = True
        rospy.loginfo(" Robot enabled")
    except Exception as e:
        rospy.logwarn(f" RobotEnable警告: {e}，仿真环境可忽略")

    # ====================== 2. MoveGroup核心配置（Noetic专属适配） ======================
    # 初始化机器人对象
    robot = moveit_commander.RobotCommander(
        robot_description=f"{ROS_NAMESPACE}/robot_description",
        ns=ROS_NAMESPACE
    )
    # 验证可用规划组
    available_groups = robot.get_group_names()
    rospy.loginfo(f" MoveIt可用规划组：{available_groups}")
    if PLANNING_GROUP not in available_groups:
        rospy.logerr(f" 规划组{PLANNING_GROUP}不存在！可用组：{available_groups}")
        return gripper, robot_enabled

    # 初始化规划组
    move_group = moveit_commander.MoveGroupCommander(
        PLANNING_GROUP,
        robot_description=f"{ROS_NAMESPACE}/robot_description",
        ns=ROS_NAMESPACE
    )
    move_group.set_end_effector_link(END_EFFECTOR_LINK)
    move_group.set_pose_reference_frame("base")  # 全程base坐标系
    # 打印当前配置
    rospy.loginfo(f"当前规划帧：{move_group.get_planning_frame()}")
    rospy.loginfo(f" 当前位姿参考帧：{move_group.get_pose_reference_frame()}")
    rospy.loginfo(f" 末端执行器link：{move_group.get_end_effector_link()}")

    # 规划核心参数优化
    move_group.allow_replanning(True)
    move_group.set_planning_time(30.0)
    move_group.set_num_planning_attempts(15)
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
    move_group.set_goal_position_tolerance(0.001)
    move_group.set_goal_orientation_tolerance(0.01)
    move_group.set_workspace([0.1, -0.2, -0.8, 0.9, 0.2, 0.5])

    # 获取运动关节列表
    joint_names = [jn for jn in robot.get_joint_names(PLANNING_GROUP) if jn in JOINT_LIMITS.keys()]
    rospy.loginfo(f"运动关节：{joint_names}")

    # ====================== 3. 初始姿态修正 ======================
    rospy.loginfo("=== 修正初始姿态 ===")
    current_joints = move_group.get_current_joint_values()
    min_len = min(len(current_joints), len(joint_names))
    safe_joints = dict(zip(joint_names[:min_len], current_joints[:min_len]))
    # 初始姿态优化：J1向下偏，为低位抓取预留空间
    safe_joints['right_j0'] = clamp_joint_value('right_j0', 0.0)
    safe_joints['right_j1'] = clamp_joint_value('right_j1', -0.8)
    safe_joints['right_j2'] = clamp_joint_value('right_j2', 0.0)
    safe_joints['right_j3'] = clamp_joint_value('right_j3', 1.8)
    safe_joints['right_j4'] = clamp_joint_value('right_j4', 0.0)
    safe_joints['right_j5'] = clamp_joint_value('right_j5', 0.0)
    safe_joints['right_j6'] = clamp_joint_value('right_j6', 0.0)

    move_group.set_joint_value_target(safe_joints)
    plan_success = False
    retry_count = 0
    while not plan_success and retry_count < 3:
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        if not plan_success:
            rospy.logwarn(f"初始姿态规划重试({retry_count+1}/3)...")
            safe_joints['right_j1'] += 0.1
            move_group.set_joint_value_target(safe_joints)
            retry_count += 1
    if plan_success:
        move_group.execute(plan, wait=True)
        rospy.sleep(2)
        rospy.loginfo(" 初始姿态修正完成")
    else:
        rospy.logwarn(" 初始姿态修正失败，使用备用安全态...")
        backup_safe_joints = {
            'right_j0':0.2, 'right_j1':-0.6, 'right_j2':0.3,
            'right_j3':1.5, 'right_j4':0.1, 'right_j5':0.2, 'right_j6':0.0
        }
        for jn in backup_safe_joints.keys():
            backup_safe_joints[jn] = clamp_joint_value(jn, backup_safe_joints[jn])
        move_group.set_joint_value_target(backup_safe_joints)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        if plan_success:
            move_group.execute(plan, wait=True)
            rospy.sleep(2)
            rospy.loginfo(" 备用安全态启动成功")
        else:
            rospy.logerr(" 安全态规划失败，程序退出")
            return gripper, robot_enabled

    # ====================== 4. 夹爪初始化 ======================
    try:
        gripper = Gripper('right_gripper')
        rospy.loginfo("=== 初始化夹爪（朝下姿态）===")
        if not gripper.is_calibrated():
            gripper.calibrate()
            rospy.sleep(2)
        gripper.set_cmd_velocity(0.03)
        gripper.open()
        rospy.loginfo(f" 夹爪已打开（指尖沿Z轴朝下伸出{FINGER_LENGTH}m）")
        rospy.sleep(1)
    except Exception as e:
        rospy.logerr(f"夹爪初始化失败：{e}")
        return gripper, robot_enabled

    # ====================== 5. 夹爪朝下标准姿态 ======================
    target_pose = geometry_msgs.msg.Pose()
    target_pose.orientation.x = 1.0
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 0.0

    # 轨迹显示发布器
    display_pub = rospy.Publisher(
        '/move_group/display_planned_path',
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=10
    )
    display_traj = moveit_msgs.msg.DisplayTrajectory()
    display_traj.trajectory_start = robot.get_current_state()

    # ===== 启动录制线程 =====
    rospy.loginfo("[Record] Starting recording thread...")
    recording = True
    record_thread = threading.Thread(target=_record_thread, daemon=True)
    record_thread.start()
    init_pose = _get_ee_pose()
    if init_pose:
        recorded_poses.append(init_pose)

    # ====================== Step1: 初始态 → 远距过渡点 ======================
    rospy.loginfo(f"=== Step1: 初始态 → 远距过渡点（基座系z={TRANSITION_BASE_Z:.3f}）===")
    transition_pose = copy.deepcopy(target_pose)
    transition_pose.position.x = TRANSITION_X
    transition_pose.position.y = OBJECT_BASE_Y
    transition_pose.position.z = TRANSITION_BASE_Z
    move_group.set_pose_target(transition_pose)

    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    planning_time = plan_result[2]
    error_msg = plan_result[3]
    retry_count = 0
    while not plan_success and retry_count < 3:
        transition_pose.position.z += 0.05
        move_group.set_pose_target(transition_pose)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        planning_time = plan_result[2]
        error_msg = plan_result[3]
        retry_count += 1
    if not plan_success:
        rospy.logerr(f" 过渡点规划失败: {error_msg}")
        recording = False
        return gripper, robot_enabled

    rospy.loginfo(f" 过渡点规划成功（耗时：{planning_time:.2f}s）")
    display_traj.trajectory.append(plan)
    display_pub.publish(display_traj)
    move_group.execute(plan, wait=True)
    rospy.sleep(len(plan.joint_trajectory.points) / 50 + 2)

    # ====================== Step2: 移动到物块上方安全高度 ======================
    rospy.loginfo(f"=== Step2: 移动到物块上方安全高度（基座系z={OVERHEAD_BASE_Z:.3f}）===")
    overhead_pose = copy.deepcopy(transition_pose)
    overhead_pose.position.x = OBJECT_BASE_X
    overhead_pose.position.y = OBJECT_BASE_Y
    overhead_pose.position.z = OVERHEAD_BASE_Z
    move_group.set_pose_target(overhead_pose)

    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    planning_time = plan_result[2]
    error_msg = plan_result[3]
    retry_count = 0
    while not plan_success and retry_count < 5:
        rospy.logwarn(f"安全高度规划重试({retry_count+1}/5)，抬高目标高度...")
        overhead_pose.position.z += 0.05
        move_group.set_pose_target(overhead_pose)
        plan_result = move_group.plan()
        plan_success = plan_result[0]
        plan = plan_result[1]
        planning_time = plan_result[2]
        error_msg = plan_result[3]
        retry_count += 1

    if plan_success:
        display_traj.trajectory[0] = plan
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        rospy.sleep(2)
        rospy.loginfo(" 到达物块上方安全高度")
    else:
        rospy.logwarn(f" 安全高度移动失败，直接水平对齐：{error_msg}")
        overhead_pose = transition_pose

    # ===== 拍摄 bottleneck =====
    rospy.loginfo("[Record] Capturing bottleneck observation...")
    bottleneck_ee = _get_ee_pose()
    if bottleneck_ee:
        rospy.loginfo(f"[Record] Bottleneck EE pose: {bottleneck_ee['position']}")
    # RGB
    try:
        rgb_msg = rospy.wait_for_message(
            "/io/internal_camera/head_camera/image_raw", Image, timeout=3.0)
        bottleneck_rgb = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rospy.loginfo(f"[Record] Bottleneck RGB: {bottleneck_rgb.shape}")
    except Exception as e:
        rospy.logwarn(f"[Record] RGB capture failed: {e}")
    # Depth（毫米 uint16）
    try:
        depth_msg = rospy.wait_for_message(
            "/io/internal_camera/head_camera/depth/image_raw", Image, timeout=3.0)
        bottleneck_depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        rospy.loginfo(f"[Record] Bottleneck Depth: {bottleneck_depth.shape} {bottleneck_depth.dtype}")
    except Exception as e:
        rospy.logwarn(f"[Record] Depth capture failed: {e}")
    # Camera intrinsics
    try:
        cinfo_msg = rospy.wait_for_message(
            "/io/internal_camera/head_camera/camera_info", CameraInfo, timeout=3.0)
        camera_intrinsics = np.array(cinfo_msg.K).reshape(3, 3)
        rospy.loginfo(f"[Record] Intrinsics (3×3):\n{camera_intrinsics}")
    except Exception as e:
        rospy.logwarn(f"[Record] CameraInfo capture failed: {e}")
        camera_intrinsics = DEFAULT_CAMERA_INTRINSICS.copy()
        rospy.logwarn(f"[Record] Using fallback intrinsics (3×3):\n{camera_intrinsics}")

    # ====================== Step3: 笛卡尔直线规划 ======================
    rospy.loginfo(f"=== Step3: XY精准对齐（目标x={OBJECT_BASE_X}, y={OBJECT_BASE_Y}，误差阈值±{ALLOWED_ERROR}m）===")
    # 获取当前位姿作为起点
    start_pose = move_group.get_current_pose().pose
    # 目标对齐位姿：Z/姿态完全锁死
    target_align_pose = copy.deepcopy(start_pose)
    target_align_pose.position.x = OBJECT_BASE_X
    target_align_pose.position.y = OBJECT_BASE_Y
    target_align_pose.position.z = OVERHEAD_BASE_Z
    target_align_pose.orientation = copy.deepcopy(target_pose.orientation)

    # 笛卡尔路径路点
    waypoints = []
    waypoints.append(copy.deepcopy(start_pose))
    waypoints.append(copy.deepcopy(target_align_pose))

    # 闭环对齐核心逻辑
    retry_count = 0
    x_error = 100.0
    y_error = 100.0
    final_align = None
    current_x = 0.0
    current_y = 0.0
    cartesian_success_threshold = 0.95

    # 笛卡尔规划前临时降速
    move_group.set_max_velocity_scaling_factor(CART_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(CART_VEL_SCALE)

    while (x_error > ALLOWED_ERROR or y_error > ALLOWED_ERROR) and retry_count < MAX_RETRY:
        rospy.loginfo(f"--- 对齐重试{retry_count+1}/{MAX_RETRY} ---")
        # 笛卡尔直线规划
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints,
            CART_STEP,
            True
        )
        # 校验规划成功率
        if fraction < cartesian_success_threshold:
            rospy.logwarn(f"笛卡尔路径规划成功率仅{fraction*100:.1f}%，抬高Z轴重试...")
            waypoints[1].position.z += 0.005
            target_align_pose.position.z += 0.005
            retry_count += 1
            continue

        rospy.loginfo(f" 笛卡尔路径规划成功，成功率{fraction*100:.1f}%")
        # 执行轨迹
        display_traj.trajectory.append(plan)
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        # 等待机械臂完全停稳
        rospy.sleep(len(plan.joint_trajectory.points) / 50 + 1.5)

        # 误差计算与校验
        final_align = move_group.get_current_pose().pose
        current_x = final_align.position.x
        current_y = final_align.position.y
        x_error = abs(current_x - OBJECT_BASE_X)
        y_error = abs(current_y - OBJECT_BASE_Y)
        rospy.loginfo(f"当前误差：x={x_error:.4f}m，y={y_error:.4f}m")
        if x_error <= GRASP_XY_TOLERANCE and y_error <= GRASP_XY_TOLERANCE:
            if x_error > ALLOWED_ERROR or y_error > ALLOWED_ERROR:
                rospy.logwarn(
                    f"已达到抓取容忍范围（±{GRASP_XY_TOLERANCE:.3f}m），停止微调以避免控制失败"
                )
            break

        # 更新路点
        waypoints[0] = copy.deepcopy(final_align)
        waypoints[1] = copy.deepcopy(target_align_pose)
        retry_count += 1

    # 恢复原始速度缩放因子
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)

    # 最终误差校验
    if x_error > GRASP_XY_TOLERANCE or y_error > GRASP_XY_TOLERANCE:
        rospy.logerr(f" 误差超出阈值：x={x_error:.4f}m，y={y_error:.4f}m")
        lift_pose = move_group.get_current_pose().pose
        lift_pose.position.z += 0.1
        move_group.set_pose_target(lift_pose)
        move_group.go(wait=True)
        recording = False
        # 保存已录制的数据（即使失败）
        _save_recorded_demo(demo_name, recorded_poses, bottleneck_rgb, bottleneck_depth,
                           bottleneck_ee or (recorded_poses[0] if recorded_poses else
                           {"position": [OBJECT_BASE_X, OBJECT_BASE_Y, OVERHEAD_BASE_Z],
                            "orientation": [1.0, 0.0, 0.0, 0.0], "timestamp": rospy.get_time()}),
                           camera_intrinsics,
                           (OBJECT_BASE_X, OBJECT_BASE_Y, OBJECT_BASE_Z),
                           (0.045, 0.045, 0.045))
        return gripper, robot_enabled

    rospy.loginfo(f"X/Y闭环对齐完成：x={current_x:.3f}（误差{x_error:.4f}m）, y={current_y:.3f}（误差{y_error:.4f}m）")
    rospy.loginfo(f" 当前基座系高度：{final_align.position.z:.3f}m，物块基座系高度：{OBJECT_BASE_Z:.3f}m")

    # ====================== Step4: 垂直下降到抓取位置 ======================
    rospy.loginfo(
        f"=== Step4: 垂直下降到抓取位置 "
        f"(contact_z={GRASP_CONTACT_BASE_Z:.3f}, flange_z={GRASP_TARGET_BASE_Z:.3f}) ==="
    )
    # 复用对齐姿态，仅修改Z轴
    grasp_pose = copy.deepcopy(final_align)
    grasp_pose.position.z = GRASP_TARGET_BASE_Z
    grasp_pose.orientation = copy.deepcopy(final_align.orientation)

    # 创建姿态路径约束
    rospy.loginfo("=== 设置姿态锁死路径约束 ===")
    ori_constraint = OrientationConstraint()
    ori_constraint.link_name = END_EFFECTOR_LINK
    ori_constraint.header.frame_id = "base"
    ori_constraint.orientation = copy.deepcopy(final_align.orientation)
    ori_constraint.absolute_x_axis_tolerance = 0.01
    ori_constraint.absolute_y_axis_tolerance = 0.01
    ori_constraint.absolute_z_axis_tolerance = 0.01
    ori_constraint.weight = 1.0

    # 组合约束
    path_constraints = Constraints()
    path_constraints.orientation_constraints.append(ori_constraint)
    move_group.set_path_constraints(path_constraints)

    # 下降阶段低速执行
    move_group.set_max_velocity_scaling_factor(DOWN_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(DOWN_ACC_SCALE)

    # 笛卡尔直线下降：只允许末端从当前点沿 base Z 方向下移
    plan_success = False
    planning_time = 0.0
    error_msg = ""
    retry_count = 0
    while not plan_success and retry_count < 5:
        descend_start = move_group.get_current_pose().pose
        descend_goal = copy.deepcopy(descend_start)
        descend_goal.position.x = OBJECT_BASE_X
        descend_goal.position.y = OBJECT_BASE_Y
        descend_goal.position.z = grasp_pose.position.z
        descend_goal.orientation = copy.deepcopy(final_align.orientation)

        descend_waypoints = [copy.deepcopy(descend_start), copy.deepcopy(descend_goal)]
        plan, fraction = move_group.compute_cartesian_path(
            descend_waypoints,
            0.005,
            True
        )
        plan_success = fraction >= 0.98 and len(plan.joint_trajectory.points) > 0
        if not plan_success:
            error_msg = f"Cartesian descent fraction={fraction:.3f}"
            rospy.logwarn(f"垂直下降笛卡尔规划重试({retry_count+1}/5)，成功率{fraction*100:.1f}%，抬高目标高度...")
            grasp_pose.position.z += 0.02
            retry_count += 1

    if not plan_success:
        rospy.logerr(f" 垂直下降路径规划失败: {error_msg}")
        # 失败后安全处理：清除约束+恢复速度+抬升
        move_group.clear_path_constraints()
        move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
        move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)
        lift_pose = move_group.get_current_pose().pose
        lift_pose.position.z += 0.1
        move_group.set_pose_target(lift_pose)
        move_group.go(wait=True)
        recording = False
        _save_recorded_demo(demo_name, recorded_poses, bottleneck_rgb, bottleneck_depth,
                           bottleneck_ee or (recorded_poses[0] if recorded_poses else
                           {"position": [OBJECT_BASE_X, OBJECT_BASE_Y, OVERHEAD_BASE_Z],
                            "orientation": [1.0, 0.0, 0.0, 0.0], "timestamp": rospy.get_time()}),
                           camera_intrinsics,
                           (OBJECT_BASE_X, OBJECT_BASE_Y, OBJECT_BASE_Z),
                           (0.045, 0.045, 0.045))
        return gripper, robot_enabled

    # 执行下降轨迹
    rospy.loginfo(" 垂直下降笛卡尔规划成功")
    display_traj.trajectory[0] = plan
    display_pub.publish(display_traj)
    move_group.execute(plan, wait=True)
    rospy.sleep(len(plan.joint_trajectory.points) / 100 + 1)

    # 清除约束+恢复原始速度
    move_group.clear_path_constraints()
    move_group.set_max_velocity_scaling_factor(ORI_VEL_SCALE)
    move_group.set_max_acceleration_scaling_factor(ORI_ACC_SCALE)

    # 下降后高度验证
    final_pose = move_group.get_current_pose().pose
    actual_base_z = final_pose.position.z
    height_diff = abs(actual_base_z - GRASP_TARGET_BASE_Z)
    rospy.loginfo(f" 下降后高度验证：")
    rospy.loginfo(f"  - 法兰实际基座Z：{actual_base_z:.3f}m（法兰目标：{GRASP_TARGET_BASE_Z:.3f}m）")
    rospy.loginfo(f"  - 抓取接触语义Z：{GRASP_CONTACT_BASE_Z:.3f}m")
    rospy.loginfo(f"  - 物块中心基座Z：{OBJECT_BASE_Z:.3f}m")
    rospy.loginfo(f"  - 高度差：{height_diff:.3f}m（≤0.01m为合格）")

    # 高度微调
    if height_diff > 0.005:
        rospy.loginfo(f" 高度差超出阈值，向下微调...")
        fine_tune_pose = copy.deepcopy(final_pose)
        fine_tune_pose.position.z = GRASP_TARGET_BASE_Z
        fine_tune_pose.orientation = copy.deepcopy(final_pose.orientation)
        plan, fraction = move_group.compute_cartesian_path(
            [copy.deepcopy(final_pose), copy.deepcopy(fine_tune_pose)],
            0.003,
            True
        )
        plan_success = fraction >= 0.98 and len(plan.joint_trajectory.points) > 0
        if plan_success:
            move_group.execute(plan, wait=True)
            rospy.sleep(2)
            final_pose = move_group.get_current_pose().pose
            actual_base_z = final_pose.position.z
            height_diff = abs(actual_base_z - GRASP_TARGET_BASE_Z)
            rospy.loginfo(f" 高度微调完成：法兰实际基座Z={actual_base_z:.3f}m（误差{height_diff:.3f}m）")
        else:
            rospy.logwarn(f" 高度微调失败（笛卡尔成功率{fraction*100:.1f}%），沿用原位置")

    # ====================== Step5: 抓取 ======================
    rospy.loginfo("=== Step5: 轻柔抓取 ===")
    grasp_success = False
    try:
        gripper.close()
        rospy.sleep(3)
        is_gripped = gripper.is_gripping()
        gripper_pos = gripper.get_position()
        if is_gripped and gripper_pos > 0.005:
            rospy.loginfo(" 抓取成功")
            grasp_success = True
        else:
            rospy.logwarn(" 夹爪未检测到抓取，执行二次闭合...")
            gripper.open()
            rospy.sleep(1)
            gripper.close()
            rospy.sleep(2)
            is_gripped = gripper.is_gripping()
            gripper_pos = gripper.get_position()
            if is_gripped and gripper_pos > 0.005:
                rospy.loginfo(" 二次抓取成功")
                grasp_success = True
            else:
                rospy.logerr(" 抓取失败")
                # 安全抬升+打开夹爪
                lift_pose = move_group.get_current_pose().pose
                lift_pose.position.z += 0.15
                move_group.set_pose_target(lift_pose)
                move_group.go(wait=True)
                gripper.open()
    except Exception as e:
        rospy.logwarn(f" 夹爪SDK调用异常：{e}")

    # ====================== Step6: 垂直抬起 ======================
    rospy.loginfo("=== Step6: 垂直抬起 ===")
    lift_pose_final = copy.deepcopy(final_pose)
    lift_pose_final.position.z = OBJECT_BASE_Z + 0.2
    move_group.set_pose_target(lift_pose_final)
    plan_result = move_group.plan()
    plan_success = plan_result[0]
    plan = plan_result[1]
    if plan_success:
        display_traj.trajectory[0] = plan
        display_pub.publish(display_traj)
        move_group.execute(plan, wait=True)
        rospy.sleep(2)
        rospy.loginfo(f"抬起成功，当前基座系高度Z={lift_pose_final.position.z:.3f}m")
    else:
        rospy.logerr(" 抬起规划失败")

    # ===== 停止录制 + 保存（无论成败都保存） =====
    rospy.loginfo("[Record] Stopping recording...")
    recording = False
    rospy.sleep(0.5)

    rospy.loginfo(f"[Record] Total recorded poses: {len(recorded_poses)}")
    object_pos = (OBJECT_BASE_X, OBJECT_BASE_Y, OBJECT_BASE_Z)
    object_size = (0.045, 0.045, 0.045)
    if bottleneck_ee is None:
        bottleneck_ee = recorded_poses[0] if recorded_poses else {
            "position": [OBJECT_BASE_X, OBJECT_BASE_Y, OVERHEAD_BASE_Z],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": rospy.get_time(),
        }
    _save_recorded_demo(demo_name, recorded_poses, bottleneck_rgb, bottleneck_depth,
                        bottleneck_ee, camera_intrinsics, object_pos, object_size)

    rospy.loginfo("抓取完成")
    return gripper, robot_enabled

# ====================== 主程序入口 ======================
if __name__ == '__main__':
    gripper = None
    try:
        gripper, robot_enabled = auto_grasp_with_moveit()
    except rospy.ROSInterruptException:
        rospy.loginfo(" 程序被用户手动中断")
    except Exception as e:
        rospy.logerr(f" 程序运行异常：{e}")
    # 资源清理
    finally:
        rospy.loginfo("=== 资源清理 ===")
        if gripper:
            try:
                gripper.open()
                rospy.loginfo(" 夹爪已打开")
            except:
                pass
        moveit_commander.roscpp_shutdown()
        rospy.sleep(1)
        rospy.loginfo(" 清理完成")
