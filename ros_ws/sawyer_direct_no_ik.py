#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def main():
    rospy.init_node('sawyer_direct_no_ik')
    
    # 直接对接控制器话题（Gazebo启动时已加载，无需额外服务）
    arm_pub = rospy.Publisher(
        '/robot/right_joint_position_controller/command',
        JointTrajectory,
        queue_size=10
    )
    rospy.sleep(2)  # 等待控制器订阅话题

    # 关节名称（严格匹配控制器配置，不能错）
    joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    
    # 构造轨迹消息（适配Sawyer定制控制器：必须包含速度/加速度）
    msg = JointTrajectory()
    msg.joint_names = joint_names
    
    # 初始点（Gazebo默认中立位置，避免碰撞）
    point1 = JointTrajectoryPoint()
    point1.positions = [0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.16]
    point1.velocities = [0.0]*7  # 定制控制器要求必须有速度字段
    point1.accelerations = [0.0]*7  # 必须有加速度字段
    point1.time_from_start = rospy.Duration(0.0)
    msg.points.append(point1)
    
    # 目标点（仅转动right_j0关节，其他关节不变，肉眼可见底座旋转）
    point2 = JointTrajectoryPoint()
    point2.positions = [0.8, -1.18, 0.0, 2.18, 0.0, 0.57, 3.16]  # right_j0转到0.8弧度（约46度）
    point2.velocities = [0.3]*7  # 速度0.3 rad/s（避免控制器拒收）
    point2.accelerations = [0.1]*7  # 加速度0.1 rad/s²
    point2.time_from_start = rospy.Duration(2.5)  # 2.5秒完成动作
    msg.points.append(point2)
    
    # 连续发布6次，确保控制器接收（解决可能的消息丢失）
    rospy.loginfo("→ 发送关节控制指令（只动right_j0）")
    for _ in range(6):
        arm_pub.publish(msg)
        rospy.sleep(0.4)
    
    # 等待动作完成，观察Gazebo机械臂
    rospy.sleep(3)
    rospy.loginfo("✅ 指令发送完成！查看Gazebo中机械臂底座是否旋转")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
