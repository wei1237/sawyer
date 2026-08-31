#!/usr/bin/env python
import rospy
from rospy.msg import AnyMsg

def wait_robot_state():
    rospy.init_node('wait_robot_state', anonymous=True)
    rospy.loginfo("⌛ 等待/robot/state话题就绪...")
    try:
        # 等待话题，超时15秒
        rospy.wait_for_message("/robot/state", AnyMsg, timeout=15.0)
        rospy.loginfo("✅ /robot/state话题已就绪！")
    except rospy.ROSException:
        rospy.logerr("❌ 超时15秒，未找到/robot/state话题，请检查Gazebo是否正常启动")
        exit(1)

if __name__ == '__main__':
    wait_robot_state()
