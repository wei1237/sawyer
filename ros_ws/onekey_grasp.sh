#!/bin/bash
echo "🔥 Sawyer 抓取任务一键启动脚本（贴合开发指南规范）"
echo "================================================"

# ================================================
# 第一步：彻底清理残留（解决进程冲突、参数缓存问题）
# ================================================
echo -e "\n🔧 清理残留进程和缓存..."
killall -9 rosmaster rosnode gzserver gzclient controller_manager move_group joint_state_publisher robot_state_publisher python3 python 2>/dev/null
rm -rf ~/.ros/* 2>/dev/null
echo "✅ 残留清理完成"

# ================================================
# 第二步：启动核心组件（按开发指南2.2.1启动顺序）
# ================================================
cd ~/ros_ws && source devel/setup.bash

# 1. 启动roscore（后台运行）
echo -e "\n🚀 启动ROS核心..."
roscore &
ROS_MASTER_PID=$!
sleep 5  # 等待master就绪
if ps -p $ROS_MASTER_PID > /dev/null; then
  echo "✅ ROS核心启动成功"
else
  echo "❌ ROS核心启动失败，退出脚本！"
  exit 1
fi

# 2. 启动Gazebo+电动夹爪（开发指南1.5.2 Gazebo启动规范）
echo -e "\n🌍 启动Gazebo仿真环境..."
roslaunch sawyer_gazebo sawyer_world.launch electric_gripper:=true use_sim_time:=true &
sleep 12  # 给足时间加载机器人模型
if rosnode list | grep -q "gzserver"; then
  echo "✅ Gazebo启动成功"
else
  echo "❌ Gazebo启动失败，退出脚本！"
  kill $ROS_MASTER_PID
  exit 1
fi

# 3. 启动物理引擎（开发指南1.5.2.1必做步骤）
echo -e "\n⚡ 启动物理引擎..."
rosservice call /gazebo/unpause_physics "{}" 2>/dev/null
echo "✅ 物理引擎已启动"

# 4. 启动关节+TF发布节点（开发指南2.1.2坐标变换依赖）
echo -e "\n🤖 启动关节状态+TF发布..."
rosrun joint_state_publisher joint_state_publisher __name:=joint_pub &
rosrun robot_state_publisher robot_state_publisher __name:=tf_pub &
sleep 4
if rosnode list | grep -q "joint_pub" && rosnode list | grep -q "tf_pub"; then
  echo "✅ 关节+TF节点启动成功"
else
  echo "❌ 关节+TF节点启动失败，退出脚本！"
  kill $ROS_MASTER_PID
  exit 1
fi

# 5. 切换位置控制器（开发指南3.3.6关节控制规范）
echo -e "\n🎮 切换位置控制器..."
rosservice call /robot/controller_manager/switch_controller "{
  start_controllers: ['right_joint_position_controller', 'joint_state_controller', 'electric_gripper_controller'],
  stop_controllers: [],
  strictness: 1
}" 2>/dev/null
sleep 3
if rosservice call /robot/controller_manager/list_controllers 2>/dev/null | grep -q "right_joint_position_controller.*running"; then
  echo "✅ 控制器切换成功"
else
  echo "❌ 控制器切换失败，退出脚本！"
  kill $ROS_MASTER_PID
  exit 1
fi

# 6. 使能机器人（开发指南3.1.3标准流程）
echo -e "\n⚡ 使能机器人..."
rosrun intera_interface enable_robot.py -e 2>/dev/null
sleep 3
if rostopic echo /robot/state -n 1 2>/dev/null | grep -q "enabled: True"; then
  echo "✅ 机器人已使能"
else
  echo "⚠️ 首次使能超时，重试..."
  rosrun intera_interface enable_robot.py -e 2>/dev/null
  sleep 2
  if rostopic echo /robot/state -n 1 2>/dev/null | grep -q "enabled: True"; then
    echo "✅ 机器人已使能"
  else
    echo "❌ 机器人使能失败，退出脚本！"
    kill $ROS_MASTER_PID
    exit 1
  fi
fi

# 7. 启动MoveIt（开发指南2.2.2 move_group接口规范，绑定/robot命名空间）
echo -e "\n🧠 启动MoveIt运动规划..."
roslaunch sawyer_moveit_config demo.launch ns:=/robot electric_gripper:=true use_sim_time:=true moveit_manage_controllers:=true &
sleep 10
if rosnode list | grep -q "/robot/move_group"; then
  echo "✅ MoveIt启动成功"
else
  echo "❌ MoveIt启动失败，退出脚本！"
  kill $ROS_MASTER_PID
  exit 1
fi

# 8. 启动轨迹转换节点（适配控制器通信）
echo -e "\n📡 启动轨迹转换节点..."
rosrun sawyer_gazebo trajectory_converter.py __name:=traj_conv ns:=/robot &
sleep 2
if rosnode list | grep -q "traj_conv"; then
  echo "✅ 轨迹转换节点启动成功"
else
  echo "❌ 轨迹转换节点启动失败，退出脚本！"
  kill $ROS_MASTER_PID
  exit 1
fi

# ================================================
# 第三步：验证核心参数（解决“参数未找到”问题）
# ================================================
echo -e "\n🔍 验证核心参数..."
if rosparam list | grep -q "/robot/robot_description_semantic" && rosparam list | grep -q "/robot/robot_description"; then
  echo "✅ MoveIt核心参数已就绪"
else
  echo "⚠️ 参数未加载，手动补充..."
  rosparam set /robot/robot_description_semantic "$(rospack find sawyer_moveit_config)/config/sawyer.srdf"
  echo "✅ 参数补充完成"
fi

# ================================================
# 第四步：运行抓取代码（显式绑定命名空间）
# ================================================
echo -e "\n🚀 启动抓取代码..."
rosrun sawyer_gazebo auto_grasp_final.py __ns:=/robot

# ================================================
# 异常兜底：关闭所有进程
# ================================================
echo -e "\n📌 抓取任务结束，清理资源..."
killall -9 rosmaster rosnode gzserver gzclient controller_manager move_group 2>/dev/null
echo "✅ 所有资源已清理"
echo -e "\n🎉 一键启动流程完成！"
