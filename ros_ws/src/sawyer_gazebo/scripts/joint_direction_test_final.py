#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import moveit_commander
from intera_interface import Gripper, RobotEnable, CHECK_VERSION  # 导入CHECK_VERSION
from std_msgs.msg import Bool  # 用于监听机器人状态

def joint_direction_test_final():
    # 复用你能正常运行的抓取代码配置
    moveit_commander.roscpp_initialize([])
    rospy.init_node('joint_direction_test_final', anonymous=True)

    # 🔴 关键修正1：等待/robot/state话题可用（避免超时）
    rospy.loginfo("⌛ 等待机器人状态话题...")
    try:
        rospy.wait_for_message("/robot/state", Bool, timeout=15.0)
    except rospy.ROSException:
        rospy.logerr("❌ 未找到/robot/state话题！请确保：1. Gazebo已启动 2. 已执行 rosrun intera_interface enable_robot.py -e")
        return

    # 加载SRDF参数（和抓取代码一致）
    semantic_param = "/robot_description_semantic"
    if not rospy.has_param(semantic_param):
        rospy.logerr(f"❌ 未找到参数 {semantic_param}，请先启动MoveIt的demo.launch")
        return
    rospy.set_param('/robot_description_semantic', rospy.get_param(semantic_param))
    rospy.loginfo("✅ SRDF参数加载成功")

    # 🔴 关键修正2：RobotEnable初始化加命名空间+版本校验
    rs = RobotEnable(CHECK_VERSION, ns="/robot")  # 指定ns+传CHECK_VERSION
    if not rs.state().enabled:
        rospy.logerr("❌ 机器人未使能！请先在终端执行：rosrun intera_interface enable_robot.py -e")
        return
    rospy.loginfo("✅ 机械臂已使能")

    # 初始化MoveIt规划组（正确配置）
    move_group = moveit_commander.MoveGroupCommander("right_arm", ns="/robot")
    move_group.set_planning_time(5.0)
    move_group.set_max_velocity_scaling_factor(0.2)
    move_group.clear_pose_targets()

    # 初始化夹爪（修正夹爪名称：Sawyer夹爪名称是'right'，不是'right_gripper'！）
    try:
        gripper = Gripper('right', ns="/robot")  # 夹爪也加命名空间
        gripper.calibrate()
        rospy.sleep(2)
        rospy.loginfo("✅ 夹爪标定完成")
    except Exception as e:
        rospy.logerr(f"❌ 夹爪初始化失败：{str(e)}")
        return

    # 关节列表+预期效果（正面观察）
    joints_info = [
        ("right_j0", "底座向【右侧】转动"),
        ("right_j1", "肩关节向【下方】转动（已怀疑反向）"),
        ("right_j2", "上臂向【外侧】转动（远离身体）"),
        ("right_j3", "肘关节向【下方】弯曲（手臂下垂）"),
        ("right_j4", "下臂向【外侧】转动"),
        ("right_j5", "腕关节向【下方】转动"),
        ("right_j6", "夹爪向【顺时针】转动（俯视）")
    ]

    # 获取当前规划组的活跃关节名称
    active_joints = move_group.get_active_joints()
    rospy.loginfo(f"当前活跃关节：{active_joints}")

    # 逐关节测试（小角度0.2rad，安全无碰撞）
    for joint_name, expected_effect in joints_info:
        choice = input(f"\n👉 测试关节：{joint_name}\n预期效果：{expected_effect}\n是否测试？（y=测试，n=跳过）：")
        if choice.lower() != 'y':
            continue

        # 检查关节是否在活跃列表中
        if joint_name not in active_joints:
            rospy.logwarn(f"⚠️ {joint_name} 不是当前规划组的活跃关节，跳过")
            continue
        joint_index = active_joints.index(joint_name)

        # 获取当前关节角度
        current_joint_values = move_group.get_current_joint_values()
        original_angle = current_joint_values[joint_index]
        target_angle = original_angle + 0.2  # 正向转动0.2rad

        # 设置目标角度并规划执行
        current_joint_values[joint_index] = target_angle
        move_group.set_joint_value_target(current_joint_values)
        rospy.loginfo(f"🔄 转动 {joint_name}：{original_angle:.2f} → {target_angle:.2f} rad")
        
        plan_success, plan, _, _ = move_group.plan()
        if not plan_success:
            rospy.logwarn(f"⚠️ {joint_name} 规划失败，跳过")
            continue
        move_group.execute(plan, wait=True)
        rospy.sleep(2)

        # 判断方向是否正确
        correct = input(f"❓ 实际转动符合预期？（y=正确，n=反向）：")
        if correct.lower() == 'n':
            rospy.loginfo(f"⚠️  标记 {joint_name} 为反向关节，后续需乘-1修正")

        # 转回原角度
        current_joint_values[joint_index] = original_angle
        move_group.set_joint_value_target(current_joint_values)
        plan_success, plan, _, _ = move_group.plan()
        if plan_success:
            move_group.execute(plan, wait=True)
        rospy.sleep(1)

    # 机械臂回中
    move_group.set_named_target("zero_pose")
    plan_success, plan, _, _ = move_group.plan()
    if plan_success:
        move_group.execute(plan, wait=True)
    rospy.loginfo("🎉 所有关节测试完成！")

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        joint_direction_test_final()
    except rospy.ROSInterruptException:
        rospy.loginfo("❌ 测试被中断")
    finally:
        # 异常时打开夹爪
        try:
            Gripper('right', ns="/robot").open()
        except:
            pass
