#!/bin/bash
cd ~/ros_ws && source devel/setup.bash
export ROS_USE_SIM_TIME=true
export ROS_NAMESPACE=/robot

# 直接运行抓取代码（跳过所有依赖验证）
echo "🚀 直接启动抓取代码..."
rosrun sawyer_gazebo auto_grasp_final.py

# 抓取完成后自动释放夹爪（保留核心收尾步骤）
unset ROS_NAMESPACE
rosrun intera_interface gripper_keyboard.py -l right -o 2>/dev/null
echo "🎉 抓取任务执行完成"
