#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def main():
    rospy.init_node('sawyer_test_move', anonymous=True)
    
    # 关键：Sawyer定制控制器的话题（与日志中一致）
    arm_pub = rospy.Publisher(
        '/robot/right_joint_position_controller/command',
        JointTrajectory,
        queue_size=10
    )
    rospy.sleep(2)  # 等待控制器订阅话题

    # 关节名称（严格按控制器配置顺序）
    joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    
    # 构造轨迹消息（适配Sawyer定制控制器：必须包含速度/加速度）
    msg = JointTrajectory()
    msg.joint_names = joint_names
    
    # 初始点（当前位置）
    point1 = JointTrajectoryPoint()
    point1.positions = [0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.16]  # 初始关节角度
    point1.velocities = [0.0]*7  # 必须包含速度（定制控制器要求）
    point1.accelerations = [0.0]*7  # 必须包含加速度
    point1.time_from_start = rospy.Duration(0.0)
    msg.points.append(point1)
    
    # 目标点（只动right_j0关节，其他不变，方便观察）
    point2 = JointTrajectoryPoint()
    point2.positions = [0.5, -1.18, 0.0, 2.18, 0.0, 0.57, 3.16]  # 仅right_j0转到0.5弧度
    point2.velocities = [0.2]*7  # 速度0.2 rad/s（定制控制器要求非零）
    point2.accelerations = [0.1]*7  # 加速度0.1 rad/s²
    point2.time_from_start = rospy.Duration(2.0)  # 2秒完成动作
    msg.points.append(point2)
    
    # 连续发布5次，确保控制器接收（解决参数冲突导致的接收不稳定）
    rospy.loginfo("→ 发布关节运动指令（只动right_j0）")
    for _ in range(5):
        arm_pub.publish(msg)
        rospy.sleep(0.5)
    
    # 等待动作完成
    rospy.sleep(3)
    rospy.loginfo("→ 测试结束，查看机械臂是否转动")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
