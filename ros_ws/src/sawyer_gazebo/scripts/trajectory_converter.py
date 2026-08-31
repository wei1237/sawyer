#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from moveit_msgs.msg import DisplayTrajectory
from intera_core_msgs.msg import JointCommand

class TrajectoryConverter:
    def __init__(self):
        rospy.init_node('trajectory_converter', anonymous=True)
        # 订阅MoveIt规划的轨迹话题
        self.trajectory_sub = rospy.Subscriber(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            self.trajectory_callback
        )
        # 发布给Gazebo控制器的话题（已验证有效）
        self.joint_cmd_pub = rospy.Publisher(
            '/robot/limb/right/joint_command',
            JointCommand,
            queue_size=10
        )
        # 初始化JointCommand消息（位置模式，关节顺序与SRDF一致）
        self.cmd = JointCommand()
        self.cmd.mode = 1  # 位置模式（开发指南3.3.6节）
        self.cmd.names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.rate = rospy.Rate(50)  # 发布频率匹配控制器

    def trajectory_callback(self, msg):
        # 解析MoveIt轨迹中的关节角度
        if len(msg.trajectory) == 0:
            rospy.logwarn("未收到有效轨迹")
            return
        points = msg.trajectory[0].joint_trajectory.points
        rospy.loginfo(f"收到轨迹，共{len(points)}个关键点，执行中...")
        # 逐个关键点发布关节命令
        for point in points:
            if rospy.is_shutdown():
                break
            self.cmd.position = point.positions
            self.joint_cmd_pub.publish(self.cmd)
            self.rate.sleep()
        rospy.loginfo("轨迹执行完成！")

if __name__ == '__main__':
    try:
        converter = TrajectoryConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("轨迹转换节点退出")
