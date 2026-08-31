#!/bin/bash
cd ~/ros_ws && source devel/setup.bash
unset ROS_NAMESPACE

# 🔧 强化残留清理（解决节点重名问题）
echo "🔧 清理残留进程和缓存..."
# 强制杀死所有相关节点，包括重名残留
killall -9 tf static_transform_publisher robot_state_publisher joint_state_publisher intera_interface_node gazebo_ros_control gzserver gzclient controller_manager joint_state_pub_only robot_state_pub_only 2>/dev/null
# 彻底删除ROS缓存，避免参数残留
rm -rf ~/.ros/log ~/.ros/cache ~/.ros/param ~/.ros/rosout 2>/dev/null
echo "✅ 残留清理完成"

# 1. 加载官方URDF（忽略xacro警告，不影响使用）
URDF_PATH=$(rospack find sawyer_description)/urdf/sawyer.urdf.xacro
if [ -f "$URDF_PATH" ]; then
  # 传递参数触发内置ros_control，忽略xacro include警告（官方文件正常现象）
  rosparam set /robot_description "`xacro $URDF_PATH electric_gripper:=true gazebo:=true`" 2>/dev/null
  echo "📥 加载官方URDF完成（含xacro内置ros_control）"
else
  echo "❌ 未找到URDF文件，路径：$URDF_PATH"
  exit 1
fi

# 2. 启动Gazebo（修复语法错误：反斜杠后不能加注释/空格）
echo "🚀 启动Gazebo（动态检测就绪状态）"
roslaunch sawyer_gazebo sawyer_world.launch \
  electric_gripper:=true \
  use_sim_time:=true \
  joint_state_publisher:=false \
  robot_state_publisher:=false \
  ns:="/robot" \
  gazebo_tf_publish:=false \
  tf_prefix:="" &

# 🔴 核心：动态检测Gazebo就绪（替代固定sleep）
echo "⌛ 等待Gazebo核心服务就绪..."
timeout 25 bash -c 'until rosservice list | grep -q "/gazebo/get_physics_properties"; do sleep 1; done'
if [ $? -eq 0 ]; then
  echo "✅ Gazebo已就绪（模型显示+核心服务可用）"
else
  echo "⚠️ Gazebo启动稍慢，手动等待3秒补全..."
  sleep 3
fi

# 3. 启动物理引擎
rosservice call /gazebo/unpause_physics "{}" 2>/dev/null
echo "⚡ 物理引擎已启动"

# 4. 启动唯一TF发布节点（确保节点名唯一，无残留）
echo "🤖 启动关节+TF发布节点..."
rosrun joint_state_publisher joint_state_publisher __name:=joint_state_pub_only &
sleep 2
rosrun robot_state_publisher robot_state_publisher __name:=robot_state_pub_only _publish_frequency:=50 &
sleep 3

# 5. 验证controller_manager和节点状态
if rosnode list | grep -q "/robot/controller_manager"; then
  echo "✅ controller_manager已启动，终端二启动完成！"
else
  echo "⚠️ controller_manager未启动，执行以下命令后重试："
  echo "rosrun controller_manager controller_manager __name:=controller_manager robot_description:=/robot_description ns:=/robot"
fi

exec bash
