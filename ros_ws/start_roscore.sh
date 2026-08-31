#!/bin/bash
# 🔴 TF保障1：彻底清理所有可能导致冲突的残留进程/缓存
killall -9 roscore rosmaster rosnode gzserver gzclient python3 python joint_state_publisher robot_state_publisher tf static_transform_publisher 2>/dev/null
rm -rf ~/.ros/log ~/.ros/cache ~/.ros/param 2>/dev/null

# 启动ROS核心
echo "🚀 启动ROS核心（已清理残留，无冲突）"
roscore &
sleep 3

# 验证核心启动
if rosnode list | grep -q "/rosout"; then
  echo "✅ ROS核心启动成功，TF基础环境就绪"
else
  echo "❌ ROS核心启动失败，检查系统环境"
  exit 1
fi

exec bash
