#!/bin/bash
set -e

# 1. 彻底清理残留进程（修复：找不到进程也不会报错终止）
echo "=== 清理残留进程 ==="
killall -9 gzserver gzclient rosmaster roscore python3 python 2>/dev/null || true
rm -rf ~/.ros/log 2>/dev/null || true
echo "✅ 残留进程清理完成"

# 2. 进入工作空间，加载ROS基础环境
echo "=== 加载ROS工作空间环境 ==="
cd ~/ros_ws
source devel/setup.bash
# 检查intera.sh是否存在，加载Sawyer SDK环境
if [ -f "./intera.sh" ]; then
    source ./intera.sh
    echo "✅ intera.sh SDK环境加载完成"
else
    echo "⚠️  警告：当前目录未找到intera.sh，跳过SDK环境加载"
fi

# 3. 全局统一仿真时间（根治时间不同步，开发指南强制要求）
rosparam set /use_sim_time true
echo "✅ 全局仿真时间设置完成"

# 4. 启动Gazebo仿真环境（官方原生sawyer_world.launch）
echo "=== 启动Sawyer Gazebo仿真环境 ==="
roslaunch sawyer_gazebo sawyer_world.launch electric_gripper:=true use_sim_time:=true &
GAZEBO_PID=$!

# 等待Gazebo真实关节状态话题就绪（避免提前启动后续节点）
echo "等待Gazebo仿真启动..."
until rostopic list | grep -q "/robot/joint_states"; do
    # 检查Gazebo进程是否还在运行
    if ! kill -0 $GAZEBO_PID 2>/dev/null; then
        echo "❌ Gazebo进程异常退出，启动失败"
        exit 1
    fi
    sleep 2
done
echo "✅ Gazebo仿真启动完成，真实关节状态话题就绪"

# 5. 启动物理引擎（开发指南仿真环境强制要求）
rosservice call /gazebo/unpause_physics "{}" 2>/dev/null || true
echo "✅ 物理引擎已启动"

# 6. 机器人使能（官方规范步骤）
rosrun intera_interface enable_robot.py -e
echo "✅ 机器人已使能"

# 7. 启动关节轨迹动作服务器（开发指南3.3.8节强制要求，MoveIt必需）
echo "=== 启动关节轨迹动作服务器 ==="
rosrun intera_interface joint_trajectory_action_server.py &
TRAJECTORY_PID=$!
sleep 3
# 检查进程是否正常运行
if kill -0 $TRAJECTORY_PID 2>/dev/null; then
    echo "✅ 关节轨迹动作服务器启动完成"
else
    echo "⚠️  关节轨迹服务器启动异常，继续执行..."
fi

# 8. 启动Sawyer官方MoveIt全组件（开发指南规范，自动加载SRDF/URDF）
echo "=== 启动Sawyer官方MoveIt组件 ==="
roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true use_sim_time:=true &
MOVEIT_PID=$!

# 等待MoveIt核心节点就绪
echo "等待MoveIt节点启动..."
until rosnode list | grep -q "/robot/move_group"; do
    # 检查MoveIt进程是否还在运行
    if ! kill -0 $MOVEIT_PID 2>/dev/null; then
        echo "❌ MoveIt进程异常退出，启动失败"
        exit 1
    fi
    sleep 2
done
echo "✅ MoveIt启动完成，全组件就绪"

# 9. 等待2秒确保所有节点完全同步
sleep 2

# 10. 运行抓取代码（替换成你的代码文件名即可）
echo "=== 启动抓取程序 ==="
rosrun sawyer_gazebo auto_grasp_final.py

# 脚本结束后清理后台进程
trap 'kill $GAZEBO_PID $TRAJECTORY_PID $MOVEIT_PID 2>/dev/null' EXIT
