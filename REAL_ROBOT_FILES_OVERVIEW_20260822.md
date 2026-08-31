# 真实 Sawyer + ASC60C + MT3 文件梳理 - 2026-08-22

最新更新：2026-08-27

这份文档用于新聊天窗口快速接手当前真实机器人进展。重点是把真实机器人相关文件和仿真文件分开，避免为了调真机误改 Gazebo/MT3 仿真链路。

## 0. 当前结论

当前真实机器人 top grasp 链路已经完成第一阶段：

```text
ASC60C RGB-D
  -> LangSAM mask
  -> depth + camera_info
  -> camera frame object geometry
  -> TF 转 base
  -> /mt3/current_object_* ROS 参数
  -> mt3_sawyer_real_grasp.py
  -> MoveIt right_arm 规划
  -> /robot/limb/right/follow_joint_trajectory
```

已经确认过的状态：

```text
ASC60C RGB / depth / camera_info 正常
LangSAM mask 正常
base <- ascamera_hp60c_color_0 TF 正常
物体中心、尺寸、top_z 已能进入 MT3 参数
demo mouth_center_xyz 已作为最高优先级抓取 anchor
MoveIt group right_arm 能加载
Cartesian path fraction 可到 100%
/robot/move_group/controller_list 指向 /robot/limb/right/follow_joint_trajectory
```

早先这个结论已经过期：

```text
MoveIt 规划成功后执行阶段 ABORTED: CONTROL_FAILED
机器人只轻微动一下
```

该问题后续已经通过真实执行链路、起点、bottleneck、执行器和感知定义逐步排掉。当前主要任务已经转为：

```text
在已验证 top grasp 基础上实现真实放置任务。
```

当前真实 top grasp 主入口：

```text
~/code/learning_thousand_tasks/mt3_sawyer_real_grasp.py
D:\ubuntu20\code\learning_thousand_tasks\mt3_sawyer_real_grasp.py
```

当前放置任务缺口：

```text
缺 mt3_sawyer_place_real.py
缺真实 place demo JSON
缺真实 place launch
缺真实 place CSV schema 和成功判定
```

## 1. 真实机器人推荐启动顺序

### 一键启动脚本

```bash
cd ~/ros_ws
./start_mt3_real.sh 192.168.137.100 192.168.137.2 noetic
```

当前 `start_mt3_real.sh` 已经按下面顺序串起来：

```text
1. 检查 /robot/state，必要时 enable Sawyer
2. 启动 joint_trajectory_action_server.py -m position
3. 等待 /robot/limb/right/follow_joint_trajectory/goal
4. 启动 ASC60C: roslaunch ascamera hp60c.launch
5. 等待 /ascamera_hp60c/rgb0/image、depth0/image_raw、rgb0/camera_info
6. 启动相机外参 TF: roslaunch sawyer_description ascamera_eye_to_hand_tf.launch
7. 等待 TF: base <- ascamera_hp60c_color_0
8. 启动 MoveIt: roslaunch sawyer_moveit_config demo_real.launch use_sim_time:=false use_rviz:=false
9. 等待 /robot/move_group/goal
10. 启动 MT3: roslaunch sawyer_gazebo mt3_real_grasp.launch
```

### 手动调试顺序

如果一键脚本出问题，用多个终端拆开查：

终端 1：

```bash
cd ~/ros_ws
./intera.sh
rosrun intera_interface joint_trajectory_action_server.py -m position
```

终端 2：

```bash
roslaunch ascamera hp60c.launch
```

终端 3：

```bash
roslaunch sawyer_description ascamera_eye_to_hand_tf.launch
```

终端 4：

```bash
roslaunch sawyer_moveit_config demo_real.launch use_sim_time:=false use_rviz:=false
```

终端 5，真实执行只在确认安全后使用：

```bash
cd ~/code/learning_thousand_tasks
python3 mt3_sawyer_real_grasp.py \
  --execute \
  --move_to_start_pose \
  --replay_velocity_scale 1.0 \
  --disable_vision_y_linear_calibration \
  --disable_vision_y_piecewise_compensation \
  --update_perception \
  --demo_path ~/code/learning_thousand_tasks/demo_library/real/recorded/cube_green_top_grasp_real.json \
  --trial_id real_grasp_01
```

## 2. MoveIt / Sawyer 真机启动文件

### `~/ros_ws/src/sawyer_moveit_config/launch/demo_real.launch`

真实 MoveIt 主入口。它把 `move_group_real.launch` 放到：

```text
/robot
```

所以当前真实系统的关键 endpoint 是：

```text
/robot/move_group
/robot/robot_description
/robot/robot_description_semantic
```

代码里不要再改回根 namespace 的 `/move_group`、`/robot_description`，除非你换了另一套启动文件。

### `~/ros_ws/src/sawyer_moveit_config/launch/move_group_real.launch`

真实 MoveIt 核心 launch。当前关键点：

```xml
<param name="robot_description"
       command="$(find xacro)/xacro --inorder $(find sawyer_description)/urdf/sawyer.urdf.xacro gazebo:=false pedestal:=true electric_gripper:=true" />
```

这解决了早先的 `right_j6` 限位问题。真机应走 `gazebo:=false`，这样：

```text
right_j6 lower=-4.7124 upper=4.7124
```

不要改厂商 `sawyer_base.urdf.xacro` 的限位数字。

它还会 include：

```text
planning_context.launch
trajectory_execution_real.launch.xml
```

并启动：

```text
/robot/move_group
```

### `~/ros_ws/src/sawyer_moveit_config/launch/trajectory_execution_real.launch.xml`

真实轨迹执行参数。被 `move_group_real.launch` 以 `ns="move_group"` include，所以参数实际落在：

```text
/robot/move_group/*
```

关键参数：

```text
moveit_manage_controllers=false
moveit_controller_manager=moveit_simple_controller_manager/MoveItSimpleControllerManager
controllers_file=config/controllers_real.yaml
```

### `~/ros_ws/src/sawyer_moveit_config/config/controllers_real.yaml`

真实 Sawyer 手臂 controller 映射：

```yaml
controller_list:
  - name: /robot/limb/right
    action_ns: follow_joint_trajectory
    type: FollowJointTrajectory
    default: true
    joints:
      - right_j0
      - right_j1
      - right_j2
      - right_j3
      - right_j4
      - right_j5
      - right_j6
```

对应 action：

```text
/robot/limb/right/follow_joint_trajectory
```

真机夹爪暂时不通过 MoveIt controller 执行，而是在 MT3 脚本里通过：

```python
intera_interface.Gripper("right_gripper")
```

直接开合。

### `~/ros_ws/src/sawyer_moveit_config/config/controllers.yaml`

旧的混合 controller 文件，里面包含类似 Gazebo/电夹爪 controller 的配置。当前真机 top grasp 不优先用它。

### `~/ros_ws/src/sawyer_moveit_config/launch/moveit_simple_controller_manager_moveit_controller_manager.launch.xml`

旧的 simple controller manager helper，会加载 `controllers.yaml`。当前 `demo_real.launch -> move_group_real.launch -> trajectory_execution_real.launch.xml` 这条链路使用的是 `controllers_real.yaml`。

## 3. 真实执行主文件

### `~/code/learning_thousand_tasks/mt3_sawyer_real_grasp.py`

当前真实 Sawyer 抓取执行入口。职责：

```text
可选调用 ASC60C 感知更新
读取 /mt3/current_object_* 参数
读取真实 demo json
用 demo mouth_center_xyz 作为嘴中心 anchor
计算 object -> mouth -> TCP
生成 object-relative replay waypoints
应用 close-tail XY correction
通过 MoveIt 执行笛卡尔轨迹
通过 Intera 控制真实夹爪
```

当前几个重要实现点：

```python
moveit_ns = "/robot"
robot_description = "/robot/robot_description"
move_group = "right_arm"
```

不要改回：

```text
ns=""
robot_description="/robot_description"
```

因为当前 `demo_real.launch` 的 MoveIt 确实运行在 `/robot` 下。

脚本里还加了 SRDF 镜像保护：

```text
/robot/robot_description_semantic -> /robot_description_semantic
```

这是为了兼容 MoveIt Python 有时在根 namespace 查 semantic 参数的问题。

真机状态消息类型是：

```python
intera_core_msgs.msg.RobotAssemblyState
```

当前 top grasp 关键行为：

```text
1. 可选 update_perception，刷新 /mt3/current_object_*。
2. 可选 move_to_start_pose，回固定 Sawyer joint 起点；该时间单独记录。
3. MoveIt 到 mapped bottleneck。
4. 执行 before-close replay。
5. 到 recorded close event 后闭合真实夹爪。
6. 不再执行 recorded after-close replay。
7. 从当前实际 pose 做纯竖直 lift，默认 0.100 m。
8. 写真实 CSV；共享目录无权限时 fallback 到本地日志目录。
```

当前软件急停：

```bash
rostopic pub -1 /mt3/emergency_stop std_msgs/Bool "data: true"
```

不是：

```python
AssemblyState
```

脚本开头还把 Sawyer 工作空间的 Python 消息路径加入 `sys.path`：

```text
/home/wei/ros_ws/devel/lib/python3/dist-packages
```

否则 `python3` 可能 import 不到 `intera_core_msgs`。

### 当前实验日志位置

默认实验日志已经改到 Windows 共享目录：

```text
/mnt/hgfs2/learning_thousand_tasks_logs
```

对应 Windows 端：

```text
D:\ubuntu20\learning_thousand_tasks_logs
```

脚本会写：

```text
mt3_real_top_grasp_trials.csv
mt3_real_top_grasp_trials.jsonl
```

直接运行脚本时不需要额外设置路径。代码里仍保留 `~real_top_grasp_log_dir` 私有参数入口，适合以后用 launch 文件启动时覆盖。

### 当前 mouth / TCP 逻辑

现在统一使用真实记录里的人工 mouth 标定：

```json
top_grasp_mouth_center_calibration.mouth_center_xyz
```

`mouth_offset_xyz` 只作为 legacy/diagnostic，不再作为主依据。

当前 TCP-mouth offset 计算方式：

```python
tcp_to_mouth_offset = demo_mouth_center - demo_grasp_tcp
```

这样 dry run 里应看到 TCP-mouth residual 接近：

```text
0 mm
```

### 当前相机等待逻辑

`mt3_sawyer_real_grasp.py` 的 `wait_for_camera_ready()` 只检查：

```text
/ascamera_hp60c/rgb0/image
/ascamera_hp60c/depth0/image_raw
```

不在 grasp 层强行等待 `camera_info`。真正需要 `camera_info` 的地方是 perception/bridge 内部。

## 4. 真实 ASC60C 感知文件

### `~/code/learning_thousand_tasks/real_perception_patch/mt3_perception_real.py`

真实 RGB-D 感知入口。默认 topic：

```text
/ascamera_hp60c/rgb0/image
/ascamera_hp60c/depth0/image_raw
/ascamera_hp60c/rgb0/camera_info
/ascamera_hp60c/depth0/points
```

默认相机 frame：

```text
ascamera_hp60c_color_0
```

默认 LangSAM mask 文件：

```text
/mnt/hgfs2/ascamera_data/current_mask.npy
```

它会输出 pose dict，关键字段：

```python
"position"
"object_points"
"object_size_m"
"visible_spread_camera_m"
"source_frame"
"mask_bbox_2d"
"mask_center_2d"
```

其中：

```python
"object_size_m": spread.tolist()
"visible_spread_camera_m": spread.tolist()
```

这两个字段当前是后续真实 geometry-aware grasp 的尺寸来源。

### `~/code/learning_thousand_tasks/real_perception_patch/mt3_alignment_real.py`

真实相机到 Sawyer base 的 TF 对齐文件。它要求 TF 存在：

```text
base <- ascamera_hp60c_color_0
```

真实模式不要退回 Gazebo 里的硬编码相机外参。

### `~/code/learning_thousand_tasks/real_perception_patch/mt3_real_object_param_bridge.py`

真实感知到 MT3 ROS 参数的桥接文件。作用：

```text
mt3_perception_real.py 输出
  -> TF 转 base
  -> /mt3/current_object_* 参数
  -> mt3_sawyer_real_grasp.py 读取
```

它发布的关键参数：

```text
/mt3/current_object_x
/mt3/current_object_y
/mt3/current_object_z
/mt3/current_object_size_m
/mt3/current_object_top_z_base
/mt3/current_object_top_z_raw_base
/mt3/current_object_top_z_offset_m
/mt3/current_object_z_semantics
/mt3/current_object_source_frame
/mt3/current_object_surface_center_base
/mt3/current_object_points_base_count
/mt3/current_object_size_source
/mt3/current_object_size_base_visible_m
/mt3/current_object_size_camera_m
```

当前尺寸来源优先级：

```text
1. /sawyer_auto_grasp/live_object_size_override_m
2. pose["object_size_m"]
3. pose["visible_spread_camera_m"]
4. base visible points percentile fallback
```

这个改动很重要：之前 bridge 用 base-frame 顶部点云厚度算 Z，可能得到 `size_z=0.004m`。但 perception 阶段的 `spread` 可能已经有约 `0.05m`，所以当前应优先使用 perception 输出的 `object_size_m`。

### `~/code/learning_thousand_tasks/real_perception_patch/mt3_real_params_updated.yaml`

真实参数文件，包含：

```text
workspace
table_surface_z
safe_joints
ASC60C topics
camera_frame
target_frame
real_top_z_offset_m
dry_run / allow_real_execution 默认值
```

这些值后续要按真实桌面高度、相机外参、夹爪 TCP 等现场标定结果继续调。

## 5. 真实示教录制文件

### `~/code/learning_thousand_tasks/real_kinesthetic_demo_patch/real_kinesthetic_recorder.py`

底层真实示教记录器，使用 Intera/TF 记录真机轨迹和末端状态。

### `~/code/learning_thousand_tasks/real_kinesthetic_demo_patch/record_demo_real.py`

当前真实 top grasp demo 录制主入口。生成的 demo JSON 包含：

```text
object_info.position_base
object_info.size_m
object_info.top_z_base
grasp_pose_base_frame
trajectory.poses
top_grasp_mouth_center_calibration.mouth_center_xyz
top_grasp_mouth_center_calibration.mouth_offset_xyz
```

replay 时应信任：

```text
top_grasp_mouth_center_calibration.mouth_center_xyz
```

不要让 `mouth_offset_xyz` 覆盖它。

其他真实录制脚本：

```text
record_anchor_place_demo_real.py
record_cuboid_yaw_demo_real.py
record_cylinder_insert_demo_real.py
```

这些不是当前第一阶段 cube -> cylinder top grasp 的主入口。

## 6. 真实 pipeline wrapper 文件

### `~/code/learning_thousand_tasks/real_pipeline_patch/mt3_pipeline_real.py`

较完整 MT3 pipeline 的真实机器人包装层。它的作用是保持仿真代码不动，把感知、对齐、安全参数换成真实版本。

当前最新 top grasp 测试不是直接从它进，而是从：

```text
mt3_sawyer_real_grasp.py
```

进。

### `~/code/learning_thousand_tasks/real_pipeline_patch/mt3_cylinder_insert_pipeline_real.py`

真实 cylinder insertion 包装文件，不属于当前 top grasp 第一阶段。

### `~/code/learning_thousand_tasks/real_pipeline_patch/check_real_demo_ready.py`

检查真实 demo 包是否完整的工具。

## 7. 真实 demo / 数据位置

当前常用 demo：

```text
~/code/learning_thousand_tasks/demo_library/real/recorded/cube_green_top_grasp_real.json
```

Windows 侧曾出现的备份位置：

```text
D:\ubuntu20\mt3_real_demo_backup\cube_green_top_grasp_real\
D:\ubuntu20\新建文件夹\cube_green_top_grasp_real.json
```

之前检查过两个 demo 版本，mouth-object XY offset 大致是：

```text
[-2.6, -1.8] mm
[-9.1,  1.9] mm
```

这说明人工示教 mouth 标定本身不是 30 mm 偏差来源。早先 30 mm 偏差来自 replay 取错 anchor 或 TCP-mouth offset 方向/来源不一致。

## 8. Windows 到 Ubuntu 复制命令

普通真实抓取代码：

```bash
cp -v /mnt/hgfs2/code/learning_thousand_tasks/mt3_sawyer_real_grasp.py \
  ~/code/learning_thousand_tasks/
```

真实感知补丁：

```bash
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_perception_real.py \
  ~/code/learning_thousand_tasks/

cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_alignment_real.py \
  ~/code/learning_thousand_tasks/

cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_real_object_param_bridge.py \
  ~/code/learning_thousand_tasks/

cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_real_params_updated.yaml \
  ~/code/learning_thousand_tasks/
```

MoveIt 文件只有在 Windows 侧改过后才复制：

```bash
cp -v /mnt/hgfs2/ros_ws/src/sawyer_moveit_config/launch/demo_real.launch \
  ~/ros_ws/src/sawyer_moveit_config/launch/

cp -v /mnt/hgfs2/ros_ws/src/sawyer_moveit_config/launch/move_group_real.launch \
  ~/ros_ws/src/sawyer_moveit_config/launch/

cp -v /mnt/hgfs2/ros_ws/src/sawyer_moveit_config/launch/trajectory_execution_real.launch.xml \
  ~/ros_ws/src/sawyer_moveit_config/launch/

cp -v /mnt/hgfs2/ros_ws/src/sawyer_moveit_config/config/controllers_real.yaml \
  ~/ros_ws/src/sawyer_moveit_config/config/
```

## 9. 过期记录：`CONTROL_FAILED` 排查方向

本节保留为历史记录。当前真实 top grasp 已经越过该阶段，不应把它当作当前主要问题。

当前现象：

```text
MoveIt ready for right_arm
real before-close replay cartesian fraction 100.0%
waypoints=28
ABORTED: CONTROL_FAILED
real before-close replay execution failed
机器人只轻微动一下
```

并且：

```bash
rosparam get /robot/move_group/controller_list
```

能看到：

```text
name: /robot/limb/right
action_ns: follow_joint_trajectory
type: FollowJointTrajectory
```

如果以后又出现同类问题，再查 action 是否收到 goal：

```bash
rostopic echo /robot/limb/right/follow_joint_trajectory/goal
```

另一个终端查 result：

```bash
rostopic echo /robot/limb/right/follow_joint_trajectory/result
```

然后再执行一次 MT3。如果 `/goal` 有消息，说明 MoveIt controller manager 基本连通，失败原因在 `joint_trajectory_action_server.py` 或 Intera 实际执行层。

如果 `/goal` 没消息，再查：

```bash
rosparam get /robot/move_group/controller_list
rosparam get /robot/move_group/moveit_controller_manager
rosnode info /robot/move_group
```

还可以看 action server 是否存在：

```bash
rostopic list | grep follow_joint_trajectory
```

## 10. 当前问题不要动这些

进入真实放置任务时，不要优先改：

```text
已验证 top grasp 的 LangSAM prompt/mask 逻辑
ASC60C 驱动
camera_info topic
base <- camera TF
demo mouth_center_xyz
MoveIt SRDF group
Sawyer URDF joint limit
Gazebo 仿真文件
```

这些不是当前放置任务的第一缺口。当前第一缺口是缺真实放置执行器和真实放置 demo。

## 11. 新窗口接手优先顺序

建议新窗口按这个顺序继续：

```text
1. 以 mt3_sawyer_real_grasp.py 作为真实 top grasp 已验证基线。
2. 查看 mt3_sawyer_place.py 中可借鉴的 transport/place/release/retreat 逻辑。
3. 新建或派生 mt3_sawyer_real_place.py，不要直接照搬 Gazebo postcheck。
4. 先实现抓取后 move above place -> descend -> gripper.open -> retreat。
5. 再接 anchor/target 双 mask 感知和真实 place demo。
6. 最后补真实 place CSV 与 manual success/failure reason。
```

当前阶段目标已经从：

```text
真实 geometry-aware top grasp
cube demo -> cylinder test
```

推进到：

```text
真实 pick-place：
已验证 top grasp
  -> 搬运到 place target
  -> 下降
  -> 打开夹爪
  -> retreat
```

仍然不要在第一版 place 中混入完整 cylinder insertion、复杂 release replay 或仿真重构。

## 12. 当前真实放置任务已有/缺失文件

已有：

```text
D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place.py
D:\ubuntu20\code\learning_thousand_tasks\mt3_place_generalization.py
D:\ubuntu20\code\learning_thousand_tasks\mt3_anchor_place_generalization.py
D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_anchor_perception_real.py
D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_anchor_place_demo_real.py
```

缺失：

```text
D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place_real.py
D:\ubuntu20\code\learning_thousand_tasks\mt3_sawyer_real_place.py
D:\ubuntu20\code\learning_thousand_tasks\demo_library\real\recorded\真实放置 demo JSON
D:\ubuntu20\ros_ws\src\sawyer_gazebo\launch\mt3_real_place.launch
D:\ubuntu20\learning_thousand_tasks_logs\真实 place CSV
```

建议第一版真实 place 以 `mt3_sawyer_real_grasp.py` 为基线扩展，而不是以旧 Gazebo `mt3_sawyer_place.py` 为基线整体搬运。
