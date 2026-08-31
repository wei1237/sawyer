#!/usr/bin/env python3
import rospy
import intera_interface

def main():
    rospy.init_node('sawyer_raw_control')
    
    # 直接连接右臂（Sawyer原生接口，绕开所有控制器冲突）
    limb = intera_interface.Limb('right')
    rospy.sleep(1)
    
    # 使能机械臂（必须步骤，激活关节控制权限）
    rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
    rs.enable()
    
    # 获取当前关节角度（打印出来，方便确认是否连接成功）
    current_angles = limb.joint_angles()
    rospy.loginfo("当前关节角度：%s", current_angles)
    
    # 只修改right_j0关节（转到0.5弧度，肉眼可见转动）
    target_angles = current_angles
    target_angles['right_j0'] = 0.5
    
    # 发布目标位置（原生接口无格式兼容问题）
    rospy.loginfo("→ 控制right_j0转动，肉眼可见机械臂底座旋转")
    limb.move_to_joint_positions(target_angles, timeout=5.0)
    
    rospy.loginfo("→ 动作完成！如果没动，检查Gazebo是否已启动")

if __name__ == '__main__':
    main()
