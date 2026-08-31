#!/usr/bin/env python
import rospy
import moveit_commander
import geometry_msgs.msg
from intera_interface import Gripper, Limb, CHECK_VERSION
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive

def plan_and_execute(group, target_pose, desc):
    """规划并执行轨迹，最多重试3次"""
    max_attempts = 3
    for attempt in range(max_attempts):
        rospy.loginfo(f"规划{desc}（第{attempt+1}/{max_attempts}次）...")
        group.set_pose_target(target_pose)
        plan = group.plan()
        if plan[0]:  # plan[0]为规划成功标志
            rospy.loginfo(f"执行{desc}...")
            group.execute(plan[1], wait=True)
            return True
        rospy.logwarn(f"{desc}规划失败，重试...")
        rospy.sleep(1)
    rospy.logerr(f"{desc}超出{max_attempts}次规划失败！")
    return False

def add_grasp_object(scene, group, grasp_pose):
    """添加带摩擦系数的待抓取物体到规划场景（适配/robot命名空间）"""
    # 1. 创建CollisionObject消息
    collision_object = CollisionObject()
    collision_object.id = "grasp_object"
    collision_object.header.frame_id = group.get_planning_frame()

    # 2. 设置物体形状（3cm×3cm×3cm，适配夹爪开合范围）
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [0.03, 0.03, 0.03]
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(grasp_pose)
    collision_object.operation = CollisionObject.ADD

    # 3. 设置摩擦系数（防止物体滑落）
    collision_object.surface_friction.mu = 1.0
    collision_object.surface_friction.mu2 = 1.0
    collision_object.surface_friction.fdir1.x = 0.0
    collision_object.surface_friction.fdir1.y = 0.0
    collision_object.surface_friction.fdir1.z = 1.0

    # 4. 发布到/robot命名空间下的规划场景（关键：适配Gazebo命名空间）
    planning_scene_pub = rospy.Publisher("/robot/planning_scene", PlanningScene, queue_size=10)
    planning_scene = PlanningScene()
    planning_scene.world.collision_objects.append(collision_object)
    planning_scene.is_diff = True
    planning_scene_pub.publish(planning_scene)

    # 5. 确认物体添加成功
    rospy.sleep(1)
    if "grasp_object" not in scene.get_known_object_names():
        rospy.logerr("物体添加失败，MoveIt未识别到！")
        return False
    rospy.loginfo("物体已成功添加到规划场景")
    return True

def main():
    # 核心修正1：去掉namespace参数（rospy.init_node不支持该参数）
    rospy.init_node("sawyer_auto_grasp_urdf")
    moveit_commander.roscpp_initialize([])
    rospy.loginfo("等待MoveIt与Gazebo接口初始化...")
    rospy.sleep(3)  # 适配虚拟机初始化延迟

    try:
        # 核心修正2：MoveIt接口通过ns参数指定/robot命名空间（该参数有效）
        robot = moveit_commander.RobotCommander(ns='/robot')
        scene = moveit_commander.PlanningSceneInterface(ns='/robot')
        group = moveit_commander.MoveGroupCommander("right_arm", ns='/robot')
        
        # 核心修正3：夹爪、机械臂接口通过namespace参数指定/robot（该参数有效）
        gripper = Gripper("right", namespace='/robot')
        limb = Limb("right", namespace='/robot')

        # 验证夹爪接口连接
        if not gripper.connected():
            rospy.logerr("错误：夹爪接口未连接！请启动Gazebo时添加electric_gripper:=true")
            return
        rospy.loginfo("所有接口初始化完成")

        # 2. 夹爪标定与参数配置（参考文档3.3.3节）
        rospy.loginfo("标定夹爪...")
        gripper.calibrate()
        rospy.sleep(2)
        gripper.set_velocity(gripper.MAX_VELOCITY * 0.8)  # 80%最大速度
        gripper.set_holding_force(50.0)  # 50N抓取力，防止滑落
        rospy.loginfo("夹爪参数配置完成")

        # 3. MoveIt规划参数（参考文档2.2.2节）
        group.set_planning_time(8.0)
        group.set_goal_position_tolerance(0.005)
        group.set_goal_orientation_tolerance(0.01)  # 补充姿态精度
        rospy.loginfo("MoveIt规划参数配置完成")

        # 4. 定义物体位置与关键位姿（适配URDF工作范围）
        object_x = 0.6  # 机械臂工作范围内
        object_y = 0.0
        object_z = 0.05  # 贴近地面
        rospy.loginfo(f"物体位置：X={object_x}, Y={object_y}, Z={object_z}")

        # 4.1 抓取前位姿（物体上方10cm）
        pre_grasp_pose = geometry_msgs.msg.Pose()
        pre_grasp_pose.orientation.x = 0.0
        pre_grasp_pose.orientation.y = 0.0
        pre_grasp_pose.orientation.z = 1.0
        pre_grasp_pose.orientation.w = 0.0  # 与URDF姿态一致（RPY=0 0 1.5708）
        pre_grasp_pose.position.x = object_x
        pre_grasp_pose.position.y = object_y
        pre_grasp_pose.position.z = object_z + 0.1

        # 4.2 抓取位姿（贴近物体上表面）
        grasp_pose = geometry_msgs.msg.Pose()
        grasp_pose.orientation = pre_grasp_pose.orientation
        grasp_pose.position.x = object_x
        grasp_pose.position.y = object_y
        grasp_pose.position.z = object_z + 0.01  # 更贴近物体（1cm间距）

        # 4.3 提升位姿（上升10cm）
        lift_pose = geometry_msgs.msg.Pose()
        lift_pose.orientation = pre_grasp_pose.orientation
        lift_pose.position.x = object_x
        lift_pose.position.y = object_y
        lift_pose.position.z = object_z + 0.11  # 高于抓取前位姿1cm

        # 5. 添加待抓取物体（带摩擦系数）
        if not add_grasp_object(scene, group, grasp_pose):
            moveit_commander.roscpp_shutdown()
            return

        # 6. 执行抓取流程
        # 6.1 移动到物体上方
        if not plan_and_execute(group, pre_grasp_pose, "移动到物体上方"):
            gripper.open()
            moveit_commander.roscpp_shutdown()
            return
        rospy.sleep(1)

        # 6.2 打开夹爪
        rospy.loginfo("打开夹爪...")
        gripper.open()
        rospy.sleep(1)
        if not gripper.is_open():
            rospy.logerr("夹爪未成功打开！")
            moveit_commander.roscpp_shutdown()
            return

        # 6.3 移动到抓取位置
        if not plan_and_execute(group, grasp_pose, "移动到抓取位置"):
            gripper.open()
            moveit_commander.roscpp_shutdown()
            return
        rospy.sleep(1)

        # 6.4 闭合夹爪并确认抓取
        rospy.loginfo("闭合夹爪抓取物体...")
        grip_success = False
        for _ in range(3):
            gripper.close()
            rospy.sleep(1.5)
            if gripper.is_gripping():
                grip_success = True
                rospy.loginfo("成功抓住物体！")
                break
            rospy.logwarn("未抓住物体，重试闭合...")
            gripper.open()
            rospy.sleep(1)
        if not grip_success:
            rospy.logerr("3次尝试均未抓住物体，抓取失败！")
            gripper.open()
            moveit_commander.roscpp_shutdown()
            return

        # 6.5 提升机械臂
        if not plan_and_execute(group, lift_pose, "提升机械臂"):
            gripper.open()
            moveit_commander.roscpp_shutdown()
            return
        rospy.loginfo("抓取流程全部完成！")

    except Exception as e:
        rospy.logerr(f"抓取过程异常：{str(e)}")
        if 'gripper' in locals():
            gripper.open()
        moveit_commander.roscpp_shutdown()
        return

if __name__ == "__main__":
    main()
