# MT3 项目与真实 Sawyer 机器人进展交接

日期：2026-08-17  
最新更新：2026-08-27

本文档用于把当前 MT3 仿真项目、真实 Sawyer 控制链路、ASC60C/HP60C RGB-D 相机、LangSAM 感知、eye-to-hand 标定进展整理到一处，方便新聊天或新 AI 继续接手。

---

## 1. 当前总目标

项目从 Sawyer + Gazebo 的 MT3 风格任务泛化复现开始，现在正在迁移到真实 Sawyer 机器人。

核心目标不是重写一套真实机代码，而是尽量复用现有 MT3 pipeline：

```text
RGB-D 感知
  -> LangSAM mask
  -> mask + depth/PointCloud2 得到物体/锚点 3D 几何
  -> 相机坐标转换到 Sawyer base
  -> 检索/选择 demo
  -> 映射 demo bottleneck
  -> MoveIt 到达 bottleneck
  -> replay demo 末端轨迹和夹爪状态
```

真实机当前最关键的未完成项已经从“相机外参/真实 top grasp”推进到“真实放置任务”。早期目标是：

```text
ASC60C/HP60C 相机外参标定
  -> 求出 base <- ascamera_hp60c_color_0
  -> 验证真实物体 3D 定位
  -> 再接入 MT3 真实抓取/放置/插入 pipeline
```

---

## 2. 重要路径

Windows 侧项目根目录：

```text
D:\ubuntu20
```

MT3 代码：

```text
D:\ubuntu20\code\learning_thousand_tasks
~/code/learning_thousand_tasks
```

Sawyer ROS 工作空间：

```text
D:\ubuntu20\ros_ws
~/ros_ws
```

ASC60C/HP60C 相机资料：

```text
D:\ubuntu20\HP60C
```

Windows LangSAM 测试脚本：

```text
D:\ubuntu20\vision_models\test_asc60c_langsam.py
```

ASC60C 数据交换目录：

```text
D:\ubuntu20\ascamera_data
```

里面当前已有：

```text
current_rgb.png
current_depth.npy
current_depth.png
current_camera_info.json
current_mask.npy
current_mask.png
current_overlay.png
current_object_points.npy
snapshot_meta.json
```

虚拟机 ASC60C 工作空间：

```text
~/ascam_ws
```

真实机 eye-to-hand 标定数据目录：

```text
~/ascam_ws/eye_to_hand_corner_calibration
```

当前出现过的 session：

```text
~/ascam_ws/eye_to_hand_corner_calibration/20260817_011253
~/ascam_ws/eye_to_hand_corner_calibration/20260817_011818
```

---

## 3. MT3 仿真项目状态

项目实现的是 Sawyer + Gazebo 上的 MT3 风格任务泛化，技术路线是：

```text
示教库
  -> 语言/任务检索
  -> LangSAM 分割
  -> RGB-D 几何估计
  -> demo bottleneck 映射
  -> MoveIt 规划
  -> replay 末端轨迹
  -> 实验日志
```

重要代码文件：

```text
mt3_generalize.py
mt3_pipeline.py
mt3_pipeline_top_lift.py
mt3_perception.py
mt3_alignment.py
mt3_anchor_perception.py
mt3_anchor_place_pipeline.py
mt3_cylinder_insert_pipeline.py
mt3_demo_library.py
mt3_scene_package.py
mt3_relation_scene_package.py
record_demo.py
record_place_demo.py
record_anchor_place_demo.py
record_cylinder_insert_demo.py
langsam_depth_localization.py
```

---

## 4. 已有仿真任务

### 4.1 top grasp

主流程：

```text
mt3_generalize.py _task:=top_grasp
mt3_pipeline.py
```

核心逻辑：

```text
目标物体 mask
  -> PointCloud2
  -> 估计物体位置/尺寸/顶面
  -> demo bottleneck 映射到当前物体
  -> MoveIt 到 bottleneck
  -> replay grasp/contact segment
```

这是后续 placement/insertion 需要对齐的参考实现。

### 4.2 rotated top grasp

主流程：

```text
mt3_generalize.py _task:=rotated_top_grasp
mt3_pipeline.py
```

重点：

```text
PCA/OBB/yaw 估计
demo yaw 与当前 yaw 对齐
调整 gripper yaw 和 bottleneck orientation
```

示例命令：

```bash
cd ~/code/learning_thousand_tasks
python3 mt3_generalize.py \
  _task:=rotated_top_grasp \
  _yaw_deg:=15 \
  _condition_id:=x07_y-008_yaw15 \
  _repeat_id:=3
```

### 4.3 anchor placement

主流程：

```text
mt3_generalize.py _task:=anchor_place
mt3_anchor_place_pipeline.py
```

当前 demo：

```text
cube_place_on_blue_platform_10cm
```

对象：

```text
target: green cube
anchor: blue placement platform
```

当前思路：

```text
target mask 得到方块几何
anchor mask 得到平台中心/尺寸
保存 target-anchor scene package
映射 demo place bottleneck
replay place/release segment
记录稳定性、中心误差、yaw 误差、平台尺寸等字段
```

典型运行：

```bash
cd ~/code/learning_thousand_tasks
python3 mt3_generalize.py \
  _task:=anchor_place \
  _demo_id:=cube_place_on_blue_platform_10cm \
  _use_demo_replay:=true \
  _condition_id:=x065_y0_6cm \
  _repeat_id:=1
```

### 4.4 cylinder insertion

主流程：

```text
mt3_generalize.py _task:=cylinder_insert_socket
mt3_cylinder_insert_pipeline.py
```

当前 demo：

```text
green_cylinder_insert_blue_socket
```

对象：

```text
target: green vertical cylinder
anchor/socket: blue circular socket
```

常用尺寸：

```text
socket_size=[0.085, 0.085, 0.100]
socket_opening=[0.055, 0.055]
cylinder_size=[0.045, 0.045, 0.100]
```

核心映射：

```text
current_bottleneck = current_insert + (demo_bottleneck - demo_insert)
```

已做过的增强：

```text
target/socket perception error logging
mapped insertion bottleneck
structured grasp_trajectory replay
rim contact / insertion relation diagnostics
socket top-band / circle-fit / mask-plane 等方法尝试
```

注意：插入任务对真实相机外参和 socket 几何精度要求最高，真实机上应最后接。

---

## 5. 真实 Sawyer 控制链路状态

真实 Sawyer 网络、ROS Master、MoveIt、TF、夹爪、joint trajectory action server 已经打通过。

当前真实机 ROS Master：

```text
http://192.168.137.100:11311
```

真实机 namespace：

```text
/robot
```

已确认过：

```text
/robot/joint_states
/robot/limb/right/endpoint_state
/robot/move_group
FollowJointTrajectory controller connected
Planning Group = right_arm
```

真实 MoveIt 启动相关文件：

```text
~/ros_ws/src/sawyer_moveit_config/launch/demo_real.launch
~/ros_ws/src/sawyer_moveit_config/launch/move_group_real.launch
~/ros_ws/src/sawyer_moveit_config/launch/trajectory_execution_real.launch.xml
~/ros_ws/src/sawyer_moveit_config/config/controllers_real.yaml
~/ros_ws/src/sawyer_gazebo/launch/mt3_real_grasp.launch
~/ros_ws/src/sawyer_gazebo/config/mt3_real_params.yaml
~/ros_ws/start_mt3_real.sh
```

### 5.1 已解决的 MoveIt 限位问题

之前 Plan 失败：

```text
Joint 'right_j6' from the starting state is outside bounds:
4.71227 should be in [-3.14, 3.14]
Start state violates joint limits
ABORTED: INVALID_ROBOT_STATE
```

根因：

```text
真实机启动链误加载了 sawyer_with_gripper.urdf
该静态 URDF 使用 gazebo 分支 right_j6=[-3.14, 3.14]
真实 Sawyer right_j6 合法范围应接近 [-4.7124, 4.7124]
```

已修复思路：

```text
move_group_real.launch 直接通过 sawyer.urdf.xacro 生成 robot_description
gazebo:=false
pedestal:=true
electric_gripper:=true
planning_context.launch 中 load_robot_description:=false
```

期望 `/robot/robot_description`：

```text
right_j6 lower=-4.7124 upper=4.7124
```

### 5.2 已解决/规避的自碰撞问题

之前还有 `head-right_l2` self collision 问题，真实机器人 MoveIt 规划后来已经能成功：

```text
Using planning pipeline 'ompl'
Planner configuration 'right_arm[RRTConnect]'
ParallelPlan::solve(): Solution found
```

### 5.3 仍存在但非阻塞的问题

Sawyer 机载会发布 reference TF tree：

```text
base -> right_l0 -> ... -> right_gripper_*
reference/base -> reference/right_l0 -> ... -> reference/right_gripper_*
```

MoveIt Plan 前曾警告：

```text
Unable to transform object from frame
'reference/right_gripper_l_finger_tip'
to planning frame 'base'
```

静态搜索 `~/ros_ws/src` 没找到本地代码写死 `reference/`。当前判断更像 Sawyer legacy environment / robot_ref_publisher / RViz planning scene 的运行时来源。

处理原则：

```text
不要删 reference/*
不要杀 robot_ref_publisher
不要全局改 TF
如需处理，优先只在 *_real.launch 中隔离
```

---

## 6. 真实机与仿真的文件隔离原则

真实机不要改仿真 launch 和 Gazebo controller 文件。

不要随便改：

```text
sawyer_world.launch
sawyer_world_place.launch
sawyer_sim_cameras.launch
controllers.yaml
gazebo_controllers.yaml
start_mt3_sim.sh
mt3_batch_runner.py
trajectory_converter.py
```

真实机相关参数集中在：

```text
mt3_real_params.yaml
*_real.launch
controllers_real.yaml
start_mt3_real.sh
```

---

## 7. ASC60C/HP60C 相机接入状态

相机实体铭牌可见：

```text
ASC60C
E43 000729
```

Ubuntu `lsusb`：

```text
ID 3482:6723 NOVATEK ASJ ZNX_NVT
```

资料包路径：

```text
D:\ubuntu20\HP60C\NUWA HP60C
```

静态分析结论：

```text
ascam_ws.zip 是 ROS1 catkin workspace
核心包是 ascamera
支持 Ubuntu 20.04 + ROS Noetic + x86_64
内含 x86_64-linux-gnu ELF64 厂商 .so
udev rules 明确覆盖 3482:6723
```

`angstrong-camera.rules` 中：

```text
# HP60C CAMERA VIDEO
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3482", ATTRS{idProduct}=="6723", MODE="0666"
```

注意：

```text
资料中没有明确出现 ASC60C 字样
但 USB ID 3482:6723 与 HP60C rules 完全对应
```

### 7.1 当前推荐相机工作空间

已经按保守方案只编译核心驱动：

```text
~/ascam_ws
└── src
    └── ascamera
```

不要第一阶段编译：

```text
ascam_visual
astra_tracker
yahboomcar_mediapipe
yahboomcar_msgs
```

原因：这些只是教学 demo，会引入 OpenCV/MediaPipe 等额外变量。

### 7.2 编译状态

环境检查通过：

```text
ROS_DISTRO=noetic
ROS_MASTER_URI=http://localhost:11311
Target: x86_64-linux-gnu
rospack find pcl_ros -> /opt/ros/noetic/share/pcl_ros
rospack find pcl_conversions -> /opt/ros/noetic/share/pcl_conversions
```

`catkin_make` 已成功：

```text
[100%] Built target ascamera_node
```

CMake 有一个 warning：

```text
libjpeg.so in /usr/lib/x86_64-linux-gnu may be hidden by
~/ascam_ws/src/ascamera/libs/lib/x86_64-linux-gnu
```

当前判断：先不用管，普通 `catkin_make` 不会把库复制进 `/usr/lib`。

建议每次启动前先确认：

```bash
ldd ~/ascam_ws/devel/lib/ascamera/ascamera_node | grep "not found"
```

如果无输出，先不要手动设置 `LD_LIBRARY_PATH`。

### 7.3 相机 ROS topic

启动：

```bash
source /opt/ros/noetic/setup.bash
source ~/ascam_ws/devel/setup.bash
roslaunch ascamera hp60c.launch
```

实际 topic：

```text
/ascamera_hp60c/rgb0/image
/ascamera_hp60c/rgb0/camera_info
/ascamera_hp60c/depth0/image_raw
/ascamera_hp60c/depth0/camera_info
/ascamera_hp60c/depth0/points
/ascamera_hp60c/mjpeg0/compressed
```

实际 frame：

```text
ascamera_hp60c_camera_link_0
ascamera_hp60c_color_0
ascamera_hp60c_depth_0
```

当前 CameraInfo：

```text
resolution = 640 x 480
frame_id = ascamera_hp60c_color_0
fx = 588.943359375
fy = 589.1879272460938
cx = 323.456787109375
cy = 235.81532287597656
D = [0, 0, 0, 0, 0]
```

---

## 8. 相机独立测试注意事项

第一次单独测相机时：

```text
不要运行 ~/ros_ws/intera.sh
```

原因：

```text
intera.sh 会把 ROS_MASTER_URI 指向真实 Sawyer:
http://192.168.137.100:11311
```

相机独立测试应使用本地 ROS master：

```text
ROS_MASTER_URI=http://localhost:11311
```

只有在需要同时读取 Sawyer TF 或做真实 hand-eye 标定时，才进入 `intera.sh` 环境。

---

## 9. Windows LangSAM 真实相机进展

当前脚本：

```text
D:\ubuntu20\vision_models\test_asc60c_langsam.py
```

当前输入：

```text
D:\ubuntu20\ascamera_data\current_rgb.png
```

当前输出：

```text
D:\ubuntu20\ascamera_data\current_mask.npy
D:\ubuntu20\ascamera_data\current_mask.png
D:\ubuntu20\ascamera_data\current_overlay.png
```

当前 prompt：

```text
small wooden block
```

已经修正过一个关键 bug：

旧逻辑：

```python
combined_mask = np.any(masks.astype(bool), axis=0)
```

问题：

```text
LangSAM 同时检出整个工作台和小木块时，会把两个 mask 合并
导致整个桌面变红
```

当前逻辑：

```text
读取 masks / boxes / scores
过滤 area_ratio >= 0.20 的巨大框
在小目标候选里按 score 选一个
只保存 selected_idx 对应的 mask
```

真实 ASC60C RGB 输入和 LangSAM 检测链路已经能工作。下一步重点是把 mask 与 ASC60C depth/PointCloud2 对齐，生成小木块自己的 3D 点云。

---

## 10. mask + depth / PointCloud2 接入 MT3 的关键点

MT3 当前最接近真实相机入口的代码：

```text
mt3_perception.py
PerceptionNode.estimate_pose_with_pointcloud_mask()
```

该函数逻辑：

```text
读取 current_mask.npy
读取 PointCloud2
按 mask 像素索引从 PointCloud2 中取 x/y/z
计算 masked points 的 median center
估计 spread / bbox / confidence
返回 source_frame 下的 pose
```

对 ASC60C 最小参数：

```text
_langsam_mask_path:=/mnt/hgfs2/ascamera_data/current_mask.npy
_pointcloud_topic:=/ascamera_hp60c/depth0/points
```

风险：

```text
要求 PointCloud2 与 RGB/mask 是像素对齐的 organized cloud
如果点云未对齐 RGB，mask 索引会错
如果点云是无序 unorganized cloud，当前函数不可靠
```

当前下一步建议：

```text
先做 camera-frame-only 测试
只打印 ascamera_hp60c_color_0 / depth frame 下的小木块 XYZ 和点数
不要先转 base，也不要直接执行机器人
```

---

## 11. Eye-to-hand 标定当前进展

当前做的是：

```text
Sawyer base <- ASC60C/HP60C RGB camera
```

也就是求：

```text
base <- ascamera_hp60c_color_0
```

标定方法已经从 ArUco 改为白色实体板外轮廓四角：

```text
640x480 RGB
  -> 白板外轮廓检测
  -> 四条边直线拟合
  -> 四个亚像素外角
  -> CameraInfo + 板实际尺寸
  -> PnP 得到 camera <- board
  -> 同步查询 base <- right_l5
  -> 多姿态联合求 base <- camera 和 right_l5 <- board
```

实体板尺寸：

```text
外框总宽 = 200 mm
外框总高 = 150 mm
```

关键 frame：

```text
BASE_FRAME = base
ROBOT_FRAME = right_l5
CAMERA_FRAME = ascamera_hp60c_color_0
```

正式程序在虚拟机：

```text
~/ascam_ws/calibrate_sawyer_hp60c_corners.py
```

该程序启动命令：

```bash
cd ~/ros_ws
./intera.sh
source ~/ascam_ws/devel/setup.bash

python3 ~/ascam_ws/calibrate_sawyer_hp60c_corners.py
```

启动后显示：

```text
Sawyer + HP60C OUTER-CORNER calibration
RGB topic: /ascamera_hp60c/rgb0/image
Robot transform: base <- right_l5
Physical board outer size: 200.0 x 150.0 mm

Keys:
  S : save current pose
  U : undo last sample
  C : solve calibration
  Q : quit
```

保存条件：

```text
FOUND
refine=YES
STABLE
max_std < 1.0 px
PnP RMS < 1.5 px
board not too close to image edge
robot pose not too similar to existing sample
```

按键：

```text
S 保存当前姿态
U 撤销最后一个样本
C 求解标定
Q 退出
```

---

## 12. 当前标定样本状态

当前已知两个 session：

```text
20260817_011253
  sample 1:
    max_std = 0.281 px
    PnP RMS = 0.783 px

20260817_011818
  sample 1:
    max_std = 0.376 px
    PnP RMS = 0.747 px
  sample 2:
    max_std = 0.641 px
    PnP RMS = 0.913 px
  sample 3:
    max_std = 0.420 px
    PnP RMS = 0.021 px
  sample 4:
    max_std = 0.242 px
    PnP RMS = 1.180 px
```

判断：

```text
这些数值本身都合格
但是否能混用取决于相机是否在两个 session 之间移动过
相机移动前后的样本绝对不能混用
```

如果顺序是：

```text
旧 sample
  -> 相机移动
  -> 011818 采了 4 个
```

则：

```text
011253 旧 sample 1 不用
011818 的 4 个保留
从当前相机位置继续采到 15-20 个
```

如果采完 `011818` 后相机又移动过，则所有旧样本失效，需要从当前最终相机位置重新采。

---

## 13. 机械臂采样姿态要求

不要只平移，也不要只转 `right_j6`。

原因：

```text
标定板固定在 right_l5 附近
只转 J6 对 right_l5/board 姿态基本没有贡献
hand-eye 需要 base <- right_l5 有充分平移和旋转变化
```

推荐采样：

```text
15-20 个不同姿态
左 / 中 / 右
前 / 中 / 后
稍高 / 稍低
左右倾 15-25 deg
前后俯仰 15-25 deg
少量组合旋转
```

不要让 15 个样本都只是：

```text
位置变化 2-3 cm
姿态几乎不变
```

每次采样流程：

```text
移动机械臂
  -> 完全停住
  -> 等 2-5 秒
  -> 看 OpenCV 窗口
  -> FOUND / refine=YES / STABLE
  -> max_std < 1 px
  -> PnP RMS < 1.5 px
  -> 四个红点在白板最外角
  -> 紫色 PnP 重投影点和红点基本重合
  -> 按 S
```

如果出现：

```text
REJECT: board too close to image edge
```

就把板移回画面中央，四边最好留 20-30 px。

如果出现：

```text
REJECT: robot pose too similar to an existing sample
```

说明机器人姿态变化不够，换更明显的位置/角度。

---

## 14. PnP RMS 与 max_std 的理解

`max_std`：

```text
机器人静止时四个检测角点是否抖动
```

`PnP RMS`：

```text
检测到的四个角是否符合 200 x 150 mm 矩形
在当前 CameraInfo 针孔模型下的投影
```

所以：

```text
max_std 很低但 PnP RMS 高
不一定是相机不稳定
可能是角点系统偏差、过曝、畸变 D=0、板太正视或靠边
```

经验门槛：

```text
max_std < 0.5 px      很好
0.5-1.0 px            可用
PnP RMS < 1.0 px      很好
1.0-1.5 px            可用
>1.5 px               不建议保存
```

如果板在画面中央 RMS 低、靠边 RMS 高，优先怀疑：

```text
CameraInfo D=[0,0,0,0,0] 是否真的代表已去畸变
```

---

## 15. 光照与标定板检测经验

黑布背景过黑时会让自动曝光抬高，导致白板过曝。

当前经验：

```text
开均匀顶灯/日光灯通常有帮助
不要用直射强灯照板
深灰/黑灰哑光背景可能比纯黑布更好
只需要白板和背景有清晰对比，不需要整张桌子纯黑
```

理想画面：

```text
白板不过曝
黑色格子清楚
白板外边缘清晰
背景不反光
```

---

## 16. 真实机器人接下来建议顺序

### 阶段 A：完成 eye-to-hand

1. 确认相机固定不再移动。
2. 使用当前最终相机位置重新/继续采样。
3. 只使用相机固定后的同一批 samples。
4. 采 15-20 个多样姿态。
5. 按 `C` 求解：

```text
base <- ascamera_hp60c_color_0
right_l5 <- board
```

6. 查看 SHAH/LI 结果、闭环误差、leave-one-out。
7. 保存最终外参。

### 阶段 B：发布相机 TF

根据标定结果发布：

```text
base -> ascamera_hp60c_color_0
```

或按程序输出的方向写 static_transform_publisher。

必须用 `tf_echo` 验证：

```bash
rosrun tf tf_echo base ascamera_hp60c_color_0
```

### 阶段 C：验证真实物体 3D 定位

先不动机器人，只做：

```text
ASC60C RGB
  -> Windows LangSAM current_mask.npy
  -> Ubuntu mask + /ascamera_hp60c/depth0/points
  -> 输出 camera frame 下 object XYZ
  -> 经 TF 转 base
  -> 与尺子测量对比
```

误差先做到 1-2 cm 内，再接入抓取。

### 阶段 D：接入真实 MT3 单次抓取

只做一个安全、低速、单目标 top grasp。

先 dry run / plan-only：

```text
不闭合夹爪
不执行下降
只看目标点和规划轨迹是否合理
```

然后再允许低速执行。

### 阶段 E：再做真实 demo 和泛化

真实 demo 推荐顺序：

```text
1. top grasp
2. rotated top grasp
3. anchor placement
4. cylinder insertion
```

插入任务最后做，因为对外参、点云、夹爪、物体几何误差最敏感。

---

## 17. 真实机安全原则

1. 相机独立测试不要进 `intera.sh`。
2. 需要 Sawyer TF/MoveIt 时才进 `intera.sh`。
3. 第一次真实执行速度保持低：

```text
normal_velocity_scale = 0.10
normal_acceleration_scale = 0.10
descent_velocity_scale = 0.05
descent_acceleration_scale = 0.05
```

4. 首次执行前必须确认：

```text
规划目标在工作台上方
不会撞相机支架
不会撞桌面
不会扯 USB 线
急停可用
```

5. 不要把仿真里的桌面高度、workspace、TCP 偏移直接用于真实机。
6. 真实桌面高度可以和仿真不同，必须实测并写入真实参数。
7. 真实 demo 可以手动拖动/示教，也可以代码控制，关键是记录同一套 base-frame 末端轨迹和感知 scene package。

---

## 18. 下一步最短行动清单

当前最应该继续的是标定，不是改 MT3 大代码。

1. 确认相机没有再动。
2. 在 `~/ascam_ws/eye_to_hand_corner_calibration` 里确认哪些 session 是相机固定后采的。
3. 如果 `011818` 是相机固定后采的，则继续从这 4 个样本往后采到 15-20 个。
4. 如果采完 `011818` 后相机又动了，则重新开一个 session，从当前相机位置采 15-20 个。
5. 求解外参并保存。
6. 发布 `base -> ascamera_hp60c_color_0`。
7. 跑 `mask + PointCloud2 -> object XYZ` 的 camera/base frame 定位验证。
8. 再回到 MT3 pipeline 做真实机低速 dry run。

---

## 19. 给新 AI 的一句话总结

这个项目已经不是“从零接真实机器人”。Sawyer 真机 MoveIt/FollowJointTrajectory/TF/夹爪链路已经打通，ASC60C/HP60C ROS1 相机驱动和 LangSAM mask 也已经跑通。当前关键卡点是完成 `base <- ascamera_hp60c_color_0` 的 eye-to-hand 外参标定，并验证真实 RGB mask 与 ASC60C PointCloud2 的像素对齐。完成这一步后，才能安全把真实相机感知结果接回 MT3 的 demo bottleneck 映射与 replay pipeline。

---

## 20. 2026-08-19 后续进展补充

这一节记录 8 月 17 之后在当前聊天窗口里继续确认的新状态，后续新窗口应优先以本节和 `PROJECT_CODE_AUDIT.md` 为准。

### 20.1 当前新增/更新的总览文档

已经重新扫描并更新：

```text
D:\ubuntu20\code\learning_thousand_tasks\repo_tree.txt
D:\ubuntu20\code\learning_thousand_tasks\PROJECT_CODE_AUDIT.md
```

其中 `PROJECT_CODE_AUDIT.md` 已经修正几个旧结论：

```text
1. 真实拖动/Zero-G 示教 recorder 已经存在，不再是缺失状态。
2. 真实 RGB-D 感知 patch 已经存在，不再是完全未适配状态。
3. 真实 pipeline patch 已经存在，但还要确认是否已合并到 Ubuntu runtime。
4. ros_ws/src/sawyer_gazebo/config/mt3_real_params.yaml 仍是旧相机 topic，占位配置需要后续更新。
```

另建了一个当前窗口交接草稿：

```text
D:\ubuntu20\code\learning_thousand_tasks\CODEX_HANDOFF_REAL_ROBOT_CURRENT_20260819.md
```

但后续主交接文档仍以当前这个 `MT3_REAL_ROBOT_PROGRESS_20260817.md` 为主。

---

### 20.2 虚拟机网卡确认

当前 Ubuntu 虚拟机网卡关系已经确认：

```text
ens33 = 虚拟机上网
ens34 = 连接 Sawyer 真实机器人
```

注意：Sawyer 的 `192.168.137.*` 地址应配置到连接 Sawyer 的网卡，也就是 `ens34`，不要加到 `ens33`。

---

### 20.3 真实工作台与示教原则确认

真实工作台高度不需要和 Gazebo 仿真桌面高度一致。

真实机必须单独测量并配置：

```text
真实桌面高度
真实 workspace
安全初始关节
TCP 偏移
夹爪参数
相机 topic
相机外参
是否使用 TF 外参
```

真实机器人示教推荐使用手动拖动 / Zero-G 风格示教，而不是完全像仿真里靠代码调位姿。关键是记录真实 `base` 坐标系下的末端轨迹、夹爪事件、bottleneck 和 RGB-D scene package，然后再让 MT3 做泛化。

当前已经存在真实示教 patch：

```text
D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch
```

关键文件：

```text
real_kinesthetic_recorder.py
record_demo_real.py
record_anchor_place_demo_real.py
record_cylinder_insert_demo_real.py
record_cuboid_yaw_demo_real.py
```

`real_kinesthetic_recorder.py` 当前默认采样：

```text
TF: base -> right_hand
RGB: /ascamera_hp60c/rgb0/image
Depth: /ascamera_hp60c/depth0/image_raw
CameraInfo: /ascamera_hp60c/rgb0/camera_info
```

键盘事件：

```text
c = 夹爪闭合并记录 gripper_close
o = 夹爪打开，闭合后打开可作为 release/open 事件
t = 标记 terminal bottleneck
s = 停止并保存
x = 中止并丢弃
```

---

### 20.4 RGB-D 相机选型和当前相机确认

老师说的是 RGB-D 相机，不是单独深度相机。ASC60C/HP60C 这类相机提供 RGB、Depth、CameraInfo、PointCloud2；LangSAM 只用 RGB 做分割，深度来自相机。

当前相机信息：

```text
铭牌：ASC60C
lsusb: ID 3482:6723 NOVATEK ASJ ZNX_NVT
USB: 2.0 / 480M
```

480M 不再作为主要故障判断。资料显示 HP60C/ASC60C 很可能本来就是 USB2.0 规格。

---

### 20.5 ASC60C/HP60C 驱动静态核对和编译状态

相机资料目录：

```text
D:\ubuntu20\HP60C
```

资料包：

```text
ascam_ws.zip
demo.zip
说明文档.zip
环境搭建.pdf
readme.txt
```

已经核对：

```text
ascam_ws.zip 是 ROS1 catkin 包。
核心包是 ascamera。
包内 x86_64 .so 与 Ubuntu 20.04 x86_64 匹配。
普通 catkin_make 不会把库复制到 /usr/lib。
udev rule 明确包含 3482:6723。
```

udev rule 中有：

```text
# HP60C CAMERA VIDEO
SUBSYSTEMS=="usb",
ATTRS{idVendor}=="3482",
ATTRS{idProduct}=="6723",
MODE="0666"
```

用户已在 Ubuntu 中确认：

```text
ROS_DISTRO=noetic
ROS_MASTER_URI=http://localhost:11311
gcc target=x86_64-linux-gnu
pcl_ros found
pcl_conversions found
```

用户已执行：

```bash
cd ~/ascam_ws
catkin_make
```

结果：

```text
ascamera_node 编译成功。
唯一 CMake warning 是 libjpeg.so 搜索路径可能被包内库隐藏。
这不是编译失败，也不是相机深度问题。
```

下一步相机侧应先检查运行时依赖：

```bash
ldd ~/ascam_ws/devel/lib/ascamera/ascamera_node | grep "not found"
```

如果无输出，再安装/确认 udev rule、重插相机、启动驱动并检查 topic。

重要原则：

```text
相机独立测试时不要运行 intera.sh。
intera.sh 会把 ROS_MASTER_URI 指到真实 Sawyer。
只有需要 Sawyer TF/MoveIt 时才进入 intera.sh。
```

---

### 20.6 ASC60C/HP60C 实际 topic 和 frame

当前以源码和实测资料为准，实际 ROS topic 是：

```text
/ascamera_hp60c/depth0/camera_info
/ascamera_hp60c/depth0/image_raw
/ascamera_hp60c/depth0/points

/ascamera_hp60c/rgb0/camera_info
/ascamera_hp60c/rgb0/image

/ascamera_hp60c/mjpeg0/compressed
```

frame 约为：

```text
ascamera_hp60c_camera_link_0
ascamera_hp60c_color_0
ascamera_hp60c_depth_0
```

如果旧 PDF 或旧说明里写：

```text
/ascamera_hp60c/depth/image_raw
```

不要优先相信旧写法。当前源码版本使用 `depth0/rgb0`。

---

### 20.7 Windows LangSAM 实例选择逻辑修正

Windows 端脚本：

```text
D:\ubuntu20\vision_models\test_asc60c_langsam.py
```

输入：

```text
D:\ubuntu20\ascamera_data\current_rgb.png
```

输出：

```text
D:\ubuntu20\ascamera_data\current_mask.npy
D:\ubuntu20\ascamera_data\current_mask.png
D:\ubuntu20\ascamera_data\current_overlay.png
```

当前 prompt：

```text
small wooden block
```

曾经的问题：

```text
LangSAM 检出两个候选：
  大黄框 = 整个工作台
  小黄框 = 真实木色物块
旧脚本把所有 masks 用 np.any 合并，导致整个桌面变红。
```

修正原则：

```text
不要合并所有 mask。
打印 boxes/scores/area_ratio。
过滤占图像面积过大的候选，例如 area_ratio >= 0.20。
在剩余小目标候选中选 score 最高的实例。
没有 score 时选最小 bbox。
```

当前审计中已记录该逻辑。后续如果 mask 又覆盖桌面，应优先检查实例选择，不要先怀疑深度或相机。

---

### 20.8 mask + depth 到 Sawyer base 坐标的调试脚本

目录：

```text
D:\ubuntu20\ascamera_data
```

关键脚本：

```text
save_hp60c_rgb_once.py
mask_to_base_xyz.py
mask_to_base_xyz_top.py
mask_to_base_xyz_top_z44.py
calibrate_sawyer_hp60c_corners_latest_tf.py
click_tip_compare_tf.py
click_tip_compare_tf_v2.py
sawyer_hover_test_safe.py
sawyer_hover_test_safe_v2.py
sawyer_hover_test_safe_v3.py
```

`mask_to_base_xyz_top.py` / `mask_to_base_xyz_top_z44.py` 当前特点：

```text
mask 默认：/mnt/hgfs2/ascamera_data/current_mask.npy
depth topic：/ascamera_hp60c/depth0/image_raw
camera_info topic：/ascamera_hp60c/depth0/camera_info
使用硬编码 T_BASE_CAMERA，即 base <- ascamera_hp60c_color_0
```

`mask_to_base_xyz_top_z44.py` 里还有经验补偿：

```text
Z_TOP_OFFSET_M = 0.044
```

这说明现在已经能把 LangSAM mask + depth 变成 base 坐标调试点，但还没有完全工程化。接入 MT3 前应把硬编码外参和 Z 补偿挪到 yaml/ROS 参数/TF。

安全 hover 测试脚本：

```text
sawyer_hover_test_safe_v3.py
```

该脚本会检查 MoveIt 当前 `right_hand` pose 和 TF `base->right_hand` 是否一致，阻止 stale state。使用 `--execute` 时还需要终端输入 `EXECUTE`，比直接执行脚本安全。

---

### 20.9 真实 MT3 patch 状态

当前 `learning_thousand_tasks` 根目录下有三类真实机 patch：

```text
real_kinesthetic_demo_patch
real_perception_patch
real_pipeline_patch
```

真实 RGB-D 感知 patch：

```text
real_perception_patch/mt3_perception_real.py
real_perception_patch/mt3_anchor_perception_real.py
real_perception_patch/mt3_alignment_real.py
real_perception_patch/mt3_real_params_updated.yaml
```

真实执行 pipeline patch：

```text
real_pipeline_patch/mt3_pipeline_real.py
real_pipeline_patch/mt3_cylinder_insert_pipeline_real.py
real_pipeline_patch/mt3_real_params_pipeline.yaml
```

重要提醒：

```text
这些 patch 文件在 Windows 项目镜像中存在。
不等于已经合并进 Ubuntu runtime。
后续要先确认复制/合并策略，避免新旧代码混用。
```

---

### 20.10 当前 `mt3_real_params.yaml` 仍需更新

文件：

```text
D:\ubuntu20\ros_ws\src\sawyer_gazebo\config\mt3_real_params.yaml
```

当前审计发现它仍含旧/占位相机 topic，例如：

```text
/io/internal_camera/head_camera/image_raw
/camera/depth/image_raw
/camera/color/camera_info
/camera/depth_registered/points
camera_color_optical_frame
```

这不适合当前 ASC60C/HP60C。后续应改成类似：

```text
rgb_topic: /ascamera_hp60c/rgb0/image
depth_topic: /ascamera_hp60c/depth0/image_raw
rgb_camera_info_topic: /ascamera_hp60c/rgb0/camera_info
depth_camera_info_topic: /ascamera_hp60c/depth0/camera_info
pointcloud_topic: /ascamera_hp60c/depth0/points
camera_frame: ascamera_hp60c_color_0
target_frame: base
```

但不要盲改。先确认到底哪个 launch / python 文件实际读取这个 yaml。

---

### 20.11 新窗口优先阅读文件

如果新开聊天窗口，先让新 AI 读：

```text
D:\ubuntu20\code\learning_thousand_tasks\MT3_REAL_ROBOT_PROGRESS_20260817.md
D:\ubuntu20\code\learning_thousand_tasks\PROJECT_CODE_AUDIT.md
D:\ubuntu20\code\learning_thousand_tasks\repo_tree.txt
```

然后重点读：

```text
D:\ubuntu20\vision_models\test_asc60c_langsam.py
D:\ubuntu20\ascamera_data\mask_to_base_xyz_top_z44.py
D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_perception_real.py
D:\ubuntu20\code\learning_thousand_tasks\real_pipeline_patch\mt3_pipeline_real.py
D:\ubuntu20\ros_ws\src\sawyer_gazebo\config\mt3_real_params.yaml
```

目标不是重写全项目，而是确认：

```text
1. 当前真实相机 topic 是否全部一致。
2. 当前外参是 TF 还是硬编码。
3. 当前 mask/depth/base 坐标输出如何传入 MT3。
4. 最小应该改哪个文件。
5. 如何保持 Gazebo 仿真不被破坏。
```

---

### 20.12 后续最短行动清单

当前不要直接跑真实抓取。建议顺序：

```text
1. 检查 ascamera_node 依赖：
   ldd ~/ascam_ws/devel/lib/ascamera/ascamera_node | grep "not found"

2. 确认/安装 HP60C udev rule，重插相机。

3. 启动相机驱动，确认：
   /ascamera_hp60c/rgb0/image
   /ascamera_hp60c/depth0/image_raw
   /ascamera_hp60c/depth0/points
   /ascamera_hp60c/rgb0/camera_info
   /ascamera_hp60c/depth0/camera_info

4. Windows LangSAM 生成 current_mask.npy，确认 mask 只覆盖目标物。

5. Ubuntu 用 mask + depth/camera_info 输出 camera/base 坐标目标点。

6. 确认或完成 base <- ascamera_hp60c_color_0 外参。

7. 把硬编码外参和 Z_TOP_OFFSET_M 参数化。

8. 更新 mt3_real_params.yaml 的相机 topic/frame。

9. 合并 real_*_patch 到 Ubuntu runtime，并做语法检查。

10. 先做 sawyer_hover_test_safe_v3.py hover/dry-run。

11. 最后再接 MT3 real pipeline，继续保持 dry_run=true、allow_real_execution=false。
```

---

### 20.13 关于“标准竖直 top-down 姿态”的定义

真实示教时如果提示：

```text
按住 cuff，把夹爪先调整成标准竖直 top-down 姿态
```

这里不能理解成“随便调一个看起来差不多竖直的姿态”。更准确的要求应该是：

```text
使用之前标定 / hover 测试时验证过的那个参考夹爪姿态。
```

也就是：

```text
1. 夹爪 TCP / approach axis 竖直朝向桌面。
2. 夹爪左右手指相对桌面方向保持标准对称，不随意绕竖直轴旋转。
3. yaw 使用之前验证过的 neutral/reference yaw。
4. right_hand/TCP 的 quaternion 应记录为真实 reference，不只靠肉眼描述。
```

需要区分两个概念：

```text
竖直 top-down：
  约束的是夹爪从上往下接近物体，主要是 roll/pitch 或 tool approach axis。

完全没旋转：
  还额外约束绕竖直方向的 yaw。这个 yaw 应该固定为之前标定/hover 测试用过的参考 yaw。
```

所以后续真实 top grasp / MT3 demo 的标准起始姿态，应优先采用：

```text
之前测试标定时那个“完全竖直、yaw 不乱转”的夹爪姿态
```

而不是每次示教时重新凭感觉调整一个新姿态。

工程上建议：

```text
1. 用 TF 或 MoveIt current pose 记录该参考 right_hand/TCP quaternion。
2. 在真实 demo recorder metadata 中保存 top_down_reference_pose。
3. 后续 replay / hover / bottleneck 映射都使用这个 reference orientation。
4. 如果任务需要旋转抓取，再在这个 reference yaw 上叠加任务需要的 yaw offset。
```

这能把真实示教、相机标定验证、hover 安全测试和 MT3 bottleneck 映射统一到同一套末端姿态定义，避免“每次看起来都竖直，但 yaw 不一致”导致抓取和泛化不稳定。

---

## 21. 2026-08-21 仿真 MT3 与真实机接入最新进展补充

这一节是 8 月 21 日根据当前项目文件和最近调试状态补充的最新交接。旧章节仍保留作历史背景，但后续新窗口应优先看本节判断当前代码状态。

### 21.1 demo 库已经支持 simulation / real 环境隔离

`mt3_demo_library.py` 已经改成按执行环境加载 demo：

```text
DemoLibrary(execution_environment=...)
MT3_EXECUTION_ENVIRONMENT
~execution_environment
```

支持的环境名：

```text
simulation / sim / gazebo
real / robot / sawyer_real / physical
```

目标目录设计为：

```text
demo_library/simulation/recorded
demo_library/simulation/auto_recorded
demo_library/real/recorded
demo_library/real/auto_recorded
```

当前磁盘上仍主要保留旧目录：

```text
demo_library/recorded
```

代码行为是：

```text
simulation: 如果 demo_library/simulation/recorded 不存在，会 fallback 到 demo_library/recorded。
real: 不 fallback 到旧 recorded，避免真实 demo 和仿真 demo 混检。
```

所以后续命令建议显式加：

```bash
_execution_environment:=simulation
```

真实机命令则必须使用：

```bash
_execution_environment:=real
```

后续如果开始正式整理数据，建议把旧仿真 demo 迁移到：

```text
demo_library/simulation/recorded
```

真实示教保存到：

```text
demo_library/real/recorded
```

不要再把两类 demo 混在 `demo_library/recorded`。

---

### 21.2 top grasp / rotated top grasp 当前技术路线

当前主入口仍是：

```text
mt3_generalize.py
```

`mt3_pipeline.py` 是较早的通用 pipeline，部分逻辑仍保留；目前仿真泛化主要以 `mt3_generalize.py` 为准。

top grasp / rotated top grasp 当前重点状态：

```text
1. 使用 LangSAM mask + PointCloud2 做目标感知。
2. `mt3_perception.py` 会输出 `estimated_object_size`。
3. `mt3_alignment.py` 会把点云变换到 base 后估计尺寸，来源字段为 `base_pointcloud_p10_p90`。
4. replay payload 会写入 demo/live object position 和 size。
5. `mt3_sawyer_grasp.py` 的 unified top replay 已要求 object-relative anchors；缺少 anchor 时会拒绝回退到 bottleneck-relative replay。
6. replay 仍然是示教轨迹 replay，但坐标从 base-frame replay 改为 object-relative / top-height-aware replay。
```

关键日志应看到：

```text
Top replay object anchors: demo_obj=... demo_size=... live_obj=... live_size=...
Replay trajectory mapping: OBJECT-RELATIVE
Replay top-height mapping: demo_top=... demo_close=... clearance=... live_top=... mapped_close=...
```

注意一个非常重要的项目语义：

```text
当前 top-height mapping 里 object_position 的 z 按项目现有数据语义处理为物体底面/接触参考 z，
所以 top_z = object_z + size_z。
不要在没有重新统一数据语义前改成 object_z + size_z / 2。
```

这点和普通“几何中心 + 半高”的写法不同，是当前代码和历史 demo 对齐后的约定。

新增诊断字段包括：

```text
transition_anchor_error_xy_m
bottleneck_anchor_error_xy_m
before_close_hand_tracking_error_xyz_m
preclose_object_to_target_error_xy_m
before_close_mouth_center_xyz
before_close_mouth_to_live_top_z_m
```

这些字段用于判断：

```text
感知是否准
bottleneck 是否对齐
MoveIt / 控制执行是否漂移
闭合前嘴中心高度是否合理
```

rotated top grasp 仍在同一技术路线基础上叠加：

```text
PCA / OBB / yaw 估计
demo yaw 与 live yaw 差值
OBB center 修正
top surface z 修正
```

---

### 21.3 anchor placement 当前状态

放置任务当前入口：

```text
mt3_anchor_place_pipeline.py
```

当前 demo：

```text
cube_place_on_blue_platform_10cm
```

当前实验设计：

```text
固定绿色方块初始位置
改变蓝色小平台位置和尺寸
评估方块是否稳定放到平台上
```

工作台高度已调整为：

```text
placement / insertion: 台面表面 z = 0.45 m
top grasp / 抓取桌面: 仍按 0.325 m 那套高度
```

放置阶段已经接入 mapped place bottleneck：

```text
aligned_place_bottleneck_pose =
live_place_pose + (demo_place_bottleneck_pose - demo_place_pose)
```

replay input 会写：

```text
place_bottleneck_pose_base_frame
aligned_place_bottleneck_pose
aligned_place_pose
place_trajectory
```

执行端 `mt3_sawyer_place.py` 会基于 mapped place bottleneck 做 place/release replay，而不是只把 release 点贴到当前平台中心。

当前放置日志已经增加论文分析需要的字段：

```text
platform_size
platform_size_m
platform_size_cm
platform_size_xyz
place_center_error_xy_m
place_center_error_xy_mm
target_place_yaw_deg
final_object_yaw_deg
place_yaw_error_deg
place_yaw_error_raw_deg
place_yaw_symmetry_deg
stable_success
precise_success
final_relation_error_xy_m
```

所以平台尺寸变化实验可以同时分析：

```text
成功率
最终中心误差
最终 yaw 误差
稳定成功 stable_success
精确放置 precise_success
```

注意：anchor placement 的 grasp replay 不是默认强制打开。若要抓取阶段也 replay，需要显式使用：

```bash
_use_grasp_replay:=true
```

正式放置实验常用命令模板：

```bash
cd ~/code/learning_thousand_tasks
python3 mt3_generalize.py \
  _task:=anchor_place \
  _demo_id:=cube_place_on_blue_platform_10cm \
  _use_demo_replay:=true \
  _execution_environment:=simulation \
  _condition_id:=x065_y0_6cm \
  _repeat_id:=1
```

---

### 21.4 cylinder insertion 当前状态

插入任务当前入口：

```text
mt3_cylinder_insert_pipeline.py
```

当前 demo：

```text
green_cylinder_insert_blue_socket
```

当前几何设定：

```text
socket / workbench 台面 z = 0.45 m
socket outer diameter ≈ 0.085 m
socket opening ≈ 0.055 m
cylinder diameter ≈ 0.045 m
```

插入任务和 top grasp / placement 的共同技术路线是：

```text
读取 demo 中的关键位姿
感知 live target / anchor 几何
把 demo bottleneck 与任务目标之间的相对关系映射到当前场景
MoveIt 到 mapped bottleneck
replay demo 末端轨迹与夹爪事件
记录感知、映射、执行和 post-check 指标
```

插入任务的特殊几何感知是合理的，因为圆孔不是盒型物体：

```text
圆孔轴心 / socket 平面 / 圆环点云
不同于 rotated top grasp 的 yaw/OBB
```

当前 socket 感知实现包含多种候选方法：

```text
base_pointcloud_top_band_circle_fit
mask_plane_inner_circle_fit
mask_plane_outer_silhouette_circle_fit
mask_plane_annulus_circle_fit
dark_hole_rgb_ray_plane_xy
base_pointcloud_circle_fit_xy
```

最近最稳定的核心方向是利用 socket 顶部点云圆环 / 平面圆拟合估计孔轴心。黑色孔 visual 只是可选视觉辅助，不应作为唯一技术路线；如果 `top_band_circle_fit` 已经稳定，就不需要强依赖黑色孔。

插入 replay 现在已经支持 mapped insertion bottleneck：

```text
current_bottleneck =
current_insert + (demo_bottleneck - demo_insert)
```

replay input 会写入：

```text
insertion_bottleneck_pose_base_frame
place_bottleneck_pose_base_frame
aligned_place_bottleneck_pose
insert_replay_anchor_mode=bottleneck
```

抓取阶段也已经支持 structured grasp replay：

```text
grasp_trajectory
close_index
structured_grasp_trajectory
```

默认参数方向：

```text
~use_grasp_replay 默认 True
~insert_require_grasp_replay 默认 True
```

插入日志已经记录感知误差和执行跟踪诊断，例如：

```text
target_perception_error_xy_m
socket_perception_error_xy_m
final_target_error_xy_m
final_relation_error_xy_m
diag_step_f_last_hand_tracking_error_xyz_m
diag_step_f_max_hand_tracking_error_xyz_m
insert_replay_tracking_error_max_m_active
```

这些字段用于解释“孔感知已经很准，但插入仍有碰撞”的情况：碰撞可能来自圆柱在夹爪中的姿态、执行跟踪误差、bottleneck 到 insertion replay 的微小横向误差，而不一定是 socket 感知错误。

---

### 21.5 真实机当前状态与安全策略

真实机方向仍保持：

```text
先验证感知和坐标，不直接跑自主抓取。
```

ASC60C/HP60C 当前接口：

```text
RGB:        /ascamera_hp60c/rgb0/image
Depth:      /ascamera_hp60c/depth0/image_raw
CameraInfo: /ascamera_hp60c/rgb0/camera_info
PointCloud2:/ascamera_hp60c/depth0/points
frame:      ascamera_hp60c_color_0
mask:       shape=(480,640), bool
```

真实接入目标链路：

```text
ASC60C RGB
  -> Windows LangSAM
  -> current_mask.npy
  -> Ubuntu ASC60C PointCloud2 + mask
  -> estimate_pose_with_pointcloud_mask()
  -> camera frame center / size
  -> base <- ascamera_hp60c_color_0 TF
  -> MT3 task pose
```

下一步最小测试仍建议先做不带 robot motion 的相机帧测试：

```text
current_mask.npy + /ascamera_hp60c/depth0/points
-> estimate_pose_with_pointcloud_mask()
-> 打印 camera frame 下 center XYZ、point count、estimated size
```

不要先改 `mt3_alignment.py` 或直接做 Sawyer base 转换。等 camera-frame 结果稳定后，再接 `base <- ascamera_hp60c_color_0` 外参。

真实 Sawyer 启动策略已经确认要比 Gazebo 保守：

```text
如果 /robot/state 已经 ready=True, enabled=True, error=False, stopped=False，则跳过重复 RobotEnable。
如果 error=True / stopped=True / ready=False / homed=False，真实机直接退出并要求人工检查。
只有 ready=True, homed=True, stopped=False, error=False, enabled=False 时，才允许自动 enable。
不要让真实机启动脚本自动 reset/recovery。
```

仿真可以更积极 retry；真实机必须 fail closed。

---

### 21.6 新窗口继续工作的优先级

如果新开会话，建议按下面顺序继续：

```text
1. 先确认当前任务是仿真继续跑实验，还是接真实 ASC60C。
2. 仿真任务先看：
   mt3_generalize.py
   mt3_anchor_place_pipeline.py
   mt3_cylinder_insert_pipeline.py
   mt3_alignment.py
   mt3_perception.py
   ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py
   ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_place.py

3. 真实相机先看：
   CODEX_HANDOFF_REAL_ASC60C_20260811.md
   real_perception_patch/REAL_PERCEPTION_PATCH_README.md
   ascamera_data/mask_to_base_xyz_top_z44.py
   vision_models/test_asc60c_langsam.py

4. demo 库隔离先看：
   mt3_demo_library.py
   demo_library/recorded
   demo_library/simulation/recorded
   demo_library/real/recorded

5. 不要把真实机 patch 当作已经全部合并进 Ubuntu runtime；每次执行前确认复制路径和实际运行文件。
```

当前仿真侧主线已经不是“能不能 replay”，而是：

```text
每类任务的几何约束是否正确映射，
以及日志是否能解释误差来源。
```

截至此前阶段，真实侧主线还是：

```text
ASC60C mask + PointCloud2 的真实 3D 定位，
以及 camera -> Sawyer base 外参。
```

---

## 22. 2026-08-27 最新真实机器人状态

### 22.1 真实 Top Grasp 已完成第一阶段

当前真实 Sawyer top grasp 已经不再停留在 hover/坐标验证阶段，已经完成完整闭环：

```text
ASC60C RGB-D
  -> Windows LangSAM mask
  -> mask erosion + registered depth + CameraInfo
  -> base frame object geometry
  -> /mt3/current_object_* ROS 参数
  -> mt3_sawyer_real_grasp.py
  -> mapped bottleneck
  -> before-close replay
  -> recorded close event
  -> gripper close
  -> pure vertical lift
  -> CSV logging
```

当前真实 top grasp 主执行文件：

```text
D:\ubuntu20\code\learning_thousand_tasks\mt3_sawyer_real_grasp.py
~/code/learning_thousand_tasks/mt3_sawyer_real_grasp.py
```

关键实现状态：

```text
MoveIt namespace: /robot
robot_description: /robot/robot_description
planning group: right_arm
trajectory action: /robot/limb/right/follow_joint_trajectory
gripper: intera_interface.Gripper("right_gripper")
```

已修正的问题：

```text
right_j6 真机限位：使用真实 xacro gazebo:=false 分支
MoveIt namespace/SRDF：保持 /robot/move_group 和 /robot/robot_description*
intera_core_msgs：使用 RobotAssemblyState
Python sys.path：加入 ~/ros_ws/devel/lib/python3/dist-packages
TCP-mouth 标定：mouth_center_xyz 为最高优先级
感知中心：使用每个点转 base 后的 median(points_base)
mask：正式感知使用 3x3 erosion x 2，与诊断脚本一致
top_z：正式 bridge 使用 base Z p90 + 0.044 m
post-close：不再 replay demo after-close 大段横向运动，改为当前 pose 纯竖直 lift
日志：写到 /mnt/hgfs2/learning_thousand_tasks_logs，失败时 fallback 到本地
急停：/mt3/emergency_stop 软件停止入口已加入
```

当前建议正式抓取命令：

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

pregrasp-only 只是悬停检查模式，不会下降闭夹：

```bash
python3 mt3_sawyer_real_grasp.py \
  --execute \
  --move_to_start_pose \
  --pregrasp_only \
  --pregrasp_clearance_m 0.02 \
  --replay_velocity_scale 1.0 \
  --disable_vision_y_linear_calibration \
  --disable_vision_y_piecewise_compensation \
  --update_perception \
  --demo_path ~/code/learning_thousand_tasks/demo_library/real/recorded/cube_green_top_grasp_real.json \
  --trial_id pregrasp_check_01
```

注意：`--pregrasp_clearance_m` 是物体顶面上方 clearance，不是桌面上方高度。`0.15` 表示停在物体顶面上方 15 cm。

### 22.2 真实 Top Grasp 日志状态

当前真实实验 CSV：

```text
D:\ubuntu20\learning_thousand_tasks_logs\mt3_real_top_grasp_trials.csv
/mnt/hgfs2/learning_thousand_tasks_logs/mt3_real_top_grasp_trials.csv
```

已记录的核心字段包括：

```text
trial_id
success
dry_run
pregrasp_only
demo_path
live_object_xyz
live_object_raw_xyz
live_object_size
live_top_z
planned_close_tcp
planned_close_mouth
actual_close_tcp
actual_close_delta_m
mapped_bottleneck_xyz
actual_bottleneck_xyz
bottleneck_error_m
start_tcp_xyz
first_waypoint_xyz
start_to_first_waypoint_error_m
cartesian_fraction
planning_time_s
robot_execution_time_s
execution_wall_time_s
total_time_s
move_to_start_pose_enabled
start_pose_reset_time_s
emergency_stop_requested
recorded_after_close_skipped
vertical_lift_m
```

时间统计现在按正式实验口径处理：

```text
move_to_start_pose 回初始位姿的时间单独记录，
不计入本轮正式 execution_time_s。
```

成功判据当前仍主要是执行器判据：

```text
hand_lift >= success_min_object_lift_m
```

真实没有 Gazebo ground truth，因此论文正式统计仍建议人工补充 `manual_success_label` 或失败原因分类，避免“没真正夹起但手臂抬升成功”被误记为任务成功。

### 22.3 当前准备进入真实放置任务

目前真实 top grasp 可以作为放置任务的前半段复用，但真实放置任务尚未完成。

已有放置相关能力：

```text
仿真/旧版 pick-place 执行器：
  D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place.py

方向放置目标泛化：
  D:\ubuntu20\code\learning_thousand_tasks\mt3_place_generalization.py

anchor-place 目标泛化：
  D:\ubuntu20\code\learning_thousand_tasks\mt3_anchor_place_generalization.py

真实双物体感知：
  D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_anchor_perception_real.py

真实 anchor-place 示教录制器：
  D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_anchor_place_demo_real.py
```

当前缺口：

```text
缺 mt3_sawyer_place_real.py
缺真实放置 demo JSON
缺真实放置 launch
缺把真实 top grasp 的 close+lift 后续接到 transport/place/release/retreat
缺真实放置 CSV 字段和人工成功判定
```

`real_pipeline_patch/mt3_pipeline_real.py` 已经预留：

```text
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_place_real.py
```

但当前 Windows/ROS 工作空间扫描没有找到该文件。因此如果直接跑真实 pick_place pipeline，会在执行器缺失处失败。

### 22.4 放置任务下一步建议

不要直接把 `mt3_sawyer_place.py` 原样用于真机。它包含 Gazebo 模型状态、旧 top grasp、插入任务和旧参数逻辑。

建议新建真实专用执行器：

```text
mt3_sawyer_real_place.py
```

第一版最小目标：

```text
复用已验证真实 top grasp：
  mapped bottleneck
  before-close replay
  gripper close
  vertical lift

新增：
  move above place
  descend to release height
  gripper.open()
  retreat upward
  写真实 place CSV
```

先做最简单的放置，不要一开始混入：

```text
完整 insertion
复杂 place release replay
自动 Gazebo postcheck
多物体闭环重检测
```
