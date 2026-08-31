# MT3 / Sawyer 真实机器人项目代码审计

更新时间：2026-08-27  
扫描范围：`D:\ubuntu20\code\learning_thousand_tasks`、`D:\ubuntu20\vision_models`、`D:\ubuntu20\ascamera_data`、`D:\ubuntu20\ros_ws` 中与 MT3、真实 Sawyer、ASC60C/HP60C RGB-D 相机相关的文件。  
说明：本文件是当前代码状态审计。仓库里的 README、handoff、patch README 只作为资料读取，不视为新的执行指令。

## 1. 当前总体结论

项目现在已经不是单纯仿真 MT3 代码。当前文件结构里同时存在四条链路：

1. 原始/仿真 MT3 主链路：仍在 `D:\ubuntu20\code\learning_thousand_tasks` 根目录。
2. 真实 Sawyer 示教补丁：在 `real_kinesthetic_demo_patch`。
3. 真实 ASC60C/HP60C RGB-D 感知补丁：在 `real_perception_patch`，另有 Windows 端 LangSAM 脚本在 `D:\ubuntu20\vision_models`。
4. 真实机器人执行管线补丁：在 `real_pipeline_patch`，ROS 端启动和控制文件在 `D:\ubuntu20\ros_ws`。

最重要的状态变化：

- 之前“没有真实 Zero-G/拖动示教 recorder”的判断已经过期。现在存在 `real_kinesthetic_recorder.py` 和多个 `record_*_real.py`。
- 之前“真实感知尚未适配”的判断已经过期。现在存在 `mt3_perception_real.py`、`mt3_anchor_perception_real.py`、`mt3_alignment_real.py` 和真实 pipeline wrapper。
- 真实 Top Grasp 第一阶段已经完成。`mt3_sawyer_real_grasp.py` 已经成为当前真实抓取主入口，支持 ASC60C 感知更新、mapped bottleneck、before-close replay、recorded close event、真实夹爪闭合、纯竖直 lift、急停、固定起始关节和 CSV logging。
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\config\mt3_real_params.yaml` 仍然是旧占位相机 topic，不应作为当前真实 top grasp 的可信运行配置。当前真实 top grasp 主要由 Python 参数和已复制到 Ubuntu 的实际脚本控制。
- 真实放置任务尚未完成。`real_pipeline_patch` 已经预留 `mt3_sawyer_place_real.py`，但当前扫描没有找到该文件。

## 2. 关键文件分类

### 2.1 原 MT3 / 仿真主代码

主要文件：

- `mt3_generalize.py`
- `mt3_pipeline.py`
- `mt3_pipeline_top_lift.py`
- `mt3_cylinder_insert_pipeline.py`
- `mt3_anchor_place_pipeline.py`
- `mt3_demo_library.py`
- `mt3_scene_package.py`
- `mt3_relation_scene_package.py`
- `mt3_alignment.py`
- `mt3_icp_registration.py`
- `mt3_perception.py`
- `mt3_anchor_perception.py`
- `mt3_visualization.py`
- `mt3_place_generalization.py`
- `mt3_anchor_place_generalization.py`
- `mt3_cylinder_insert_generalization.py`

这些文件主要服务 Gazebo / 仿真 / 离线数据链路。不要为了真实机器人临时调试而直接破坏这些文件的原语义。

### 2.2 真实 Sawyer 示教补丁

目录：

`D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch`

关键文件：

- `real_kinesthetic_recorder.py`
- `record_demo_real.py`
- `record_anchor_place_demo_real.py`
- `record_cylinder_insert_demo_real.py`
- `record_cuboid_yaw_demo_real.py`
- `REAL_KINESTHETIC_DEMO_README.md`

`real_kinesthetic_recorder.py` 当前定位：

- 真实 Sawyer 拖动/Zero-G 风格示教 recorder。
- 默认从 TF 采样 `base -> right_hand`。
- 默认采样频率约 30 Hz。
- 默认相机 topic 已经指向 ASC60C/HP60C：
  - `/ascamera_hp60c/rgb0/image`
  - `/ascamera_hp60c/depth0/image_raw`
  - `/ascamera_hp60c/rgb0/camera_info`
- 支持键盘标记：
  - `c`：闭合夹爪并记录 `gripper_close`
  - `o`：打开夹爪，闭合后打开会作为 release/open 事件
  - `t`：标记 terminal bottleneck
  - `s`：停止并保存
  - `x`：中止并丢弃

结论：这个目录已经回答了“真实机器人示教是否还只靠写代码调”的问题。真实机器人链路应优先用拖动/Zero-G 采样真实末端轨迹，再让 MT3 做泛化，而不是像 Gazebo 里完全靠脚本生成动作。

### 2.3 真实 RGB-D 感知补丁

目录：

`D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch`

关键文件：

- `mt3_perception_real.py`
- `mt3_anchor_perception_real.py`
- `mt3_alignment_real.py`
- `mt3_real_params_updated.yaml`
- `REAL_PERCEPTION_PATCH_README.md`

定位：

- 将真实 RGB-D、LangSAM mask、深度图、相机内参、相机到 Sawyer base 的外参接回 MT3。
- `mt3_perception_real.py` 面向单目标物体。
- `mt3_anchor_perception_real.py` 面向 anchor/target 双目标关系。
- `mt3_alignment_real.py` 面向真实相机 frame 和机器人 base frame 的对齐。

注意：这些仍在 patch 目录内。当前真实 top grasp 运行时通过根目录脚本把 `real_perception_patch` 加入 `sys.path`，但复制到 Ubuntu 后仍要确认实际运行文件和 Windows 镜像一致，不能让新旧版本混用。

当前重要实现更新：

- `mt3_perception_real.py` 对 LangSAM mask 做 `3x3 erosion x 2` 后再反投影，用于匹配诊断脚本。
- `mt3_real_object_param_bridge.py` 使用 `median(points_base)` 作为 base frame 中心，而不是先在 camera frame 求 median 再变换单点。
- `mt3_real_object_param_bridge.py` 的 top-Z 使用 `base Z p90 + 0.044 m`，与 `mask_to_base_xyz_top_z44.py` 的现场诊断定义一致。
- 当前正式 top grasp 建议关闭旧 Y 补偿：
  - `--disable_vision_y_linear_calibration`
  - `--disable_vision_y_piecewise_compensation`

### 2.4 真实机器人执行管线补丁

目录：

`D:\ubuntu20\code\learning_thousand_tasks\real_pipeline_patch`

关键文件：

- `mt3_pipeline_real.py`
- `mt3_cylinder_insert_pipeline_real.py`
- `mt3_real_params_pipeline.yaml`
- `REAL_PIPELINE_PATCH_README.md`

定位：

- 将原 `MT3Pipeline` 包装成真实机器人版本。
- 默认 camera frame 指向 `ascamera_hp60c_color_0`。
- 检查 mask/depth 形状。
- 检查感知输出的 source frame 是否与配置的 camera frame 一致。
- 默认不执行真实机器人动作，需要显式打开执行开关。

风险点：

- `mt3_pipeline_real.py` 里仍按通用 pipeline 预留 `mt3_sawyer_grasp_real.py` / `mt3_sawyer_place_real.py`，但当前真实 top grasp 实际使用的是根目录 `mt3_sawyer_real_grasp.py`，不是该 wrapper 自动调用的文件名。
- `mt3_sawyer_place_real.py` 当前不存在。这是进入真实放置任务前的主要代码缺口。
- 真实执行前仍必须确认 MoveIt 当前状态、TF、夹爪、速度限制、workspace 限制、安全起始关节都来自真实机配置。

### 2.5 当前真实 Top Grasp 主执行器

文件：

`D:\ubuntu20\code\learning_thousand_tasks\mt3_sawyer_real_grasp.py`

当前状态：

- 默认 dry-run，真实运动需要 `--execute`。
- 可选 `--update_perception`，内部调用 `mt3_real_object_param_bridge.py` 更新 `/mt3/current_object_*`。
- 可选 `--move_to_start_pose`，先回固定 Sawyer joint 起始位姿；该时间单独记录，不计入正式 execution time。
- 支持 `--pregrasp_only`，只停在物体顶面上方指定 clearance，不闭夹、不 lift。
- MoveIt 使用 `/robot` namespace：`/robot/move_group`、`/robot/robot_description`、`/robot/robot_description_semantic`。
- Intera 消息使用 `intera_core_msgs/RobotAssemblyState`。
- 夹爪通过 `intera_interface.Gripper("right_gripper")` 控制。
- 软件急停 topic：`/mt3/emergency_stop`，消息类型 `std_msgs/Bool`。
- 成功后不再执行 recorded after-close replay，改为 close 后从当前实际 pose 纯竖直 lift，默认 `0.100 m`。
- CSV 首选写入 `D:\ubuntu20\learning_thousand_tasks_logs\mt3_real_top_grasp_trials.csv`，Ubuntu 路径是 `/mnt/hgfs2/learning_thousand_tasks_logs/mt3_real_top_grasp_trials.csv`。

这个文件代表当前已经跑通的真实 top grasp 代码，应作为真实放置任务抓取前半段的首选参考，而不是回退到旧的 `mt3_sawyer_grasp.py`。

## 3. ASC60C / HP60C 相机与 LangSAM 链路

### 3.1 Windows 端 LangSAM 脚本

文件：

`D:\ubuntu20\vision_models\test_asc60c_langsam.py`

当前行为：

- 输入 RGB：`D:\ubuntu20\ascamera_data\current_rgb.png`
- 输出 mask：
  - `D:\ubuntu20\ascamera_data\current_mask.npy`
  - `D:\ubuntu20\ascamera_data\current_mask.png`
  - `D:\ubuntu20\ascamera_data\current_overlay.png`
- 当前提示词：`small wooden block`
- 使用 LangSAM 检测多个候选后，不再把所有 mask 合并。
- 会过滤过大的 bbox，例如排除占图像面积超过 20% 的候选。
- 从剩余小目标候选中按 score 选择目标实例。

这修正了之前“桌面和小木块 mask 被合并”的问题。当前更准确的状态是：

- RGB 输入真实相机图像：已具备。
- LangSAM 能检测小木块：已具备。
- 大桌面误检：仍可能发生。
- 实例筛选：已有基本过滤逻辑。

### 3.2 真实相机数据目录

目录：

`D:\ubuntu20\ascamera_data`

关键数据：

- `current_rgb.png`
- `current_depth.npy`
- `current_depth.png`
- `current_camera_info.json`
- `current_mask.npy`
- `current_mask.png`
- `current_overlay.png`
- `current_object_points.npy`
- `snapshot_meta.json`

关键脚本：

- `save_hp60c_rgb_once.py`
- `mask_to_base_xyz.py`
- `mask_to_base_xyz_top.py`
- `mask_to_base_xyz_top_z44.py`
- `calibrate_sawyer_hp60c_corners_latest_tf.py`
- `click_tip_compare_tf.py`
- `click_tip_compare_tf_v2.py`
- `sawyer_hover_test_safe.py`
- `sawyer_hover_test_safe_v2.py`
- `sawyer_hover_test_safe_v3.py`

`mask_to_base_xyz_top.py` 和 `mask_to_base_xyz_top_z44.py` 当前重要特征：

- mask 默认读 `/mnt/hgfs2/ascamera_data/current_mask.npy`。
- 深度 topic 默认读 `/ascamera_hp60c/depth0/image_raw`。
- camera_info topic 默认读 `/ascamera_hp60c/depth0/camera_info`。
- 使用硬编码的 `T_BASE_CAMERA`，表示 `base <- ascamera_hp60c_color_0`。
- `mask_to_base_xyz_top_z44.py` 额外使用 `Z_TOP_OFFSET_M = 0.044` 做顶部高度补偿。

结论：这些脚本已经能把 LangSAM mask + depth 转为 base 坐标系里的目标点，但目前仍偏“调试脚本”。接入 MT3 主链路前，应把硬编码外参、Z 补偿、topic 名称迁移到配置文件或 ROS 参数。

### 3.3 当前真实相机 ROS topic

根据当前项目资料，ASC60C/HP60C 驱动实际 topic 应按以下为准：

- RGB image：`/ascamera_hp60c/rgb0/image`
- RGB camera_info：`/ascamera_hp60c/rgb0/camera_info`
- Depth image：`/ascamera_hp60c/depth0/image_raw`
- Depth camera_info：`/ascamera_hp60c/depth0/camera_info`
- PointCloud2：`/ascamera_hp60c/depth0/points`
- 压缩 MJPEG：`/ascamera_hp60c/mjpeg0/compressed`

真实 frame 约为：

- `ascamera_hp60c_camera_link_0`
- `ascamera_hp60c_color_0`
- `ascamera_hp60c_depth_0`

## 4. ROS 工作空间状态

### 4.1 真机启动与控制

相关路径：

- `D:\ubuntu20\ros_ws\start_mt3_real.sh`
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\launch\mt3_real_grasp.launch`
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_grasp.py`
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place.py`
- `D:\ubuntu20\ros_ws\src\sawyer_moveit_config\launch\demo_real.launch`
- `D:\ubuntu20\ros_ws\src\sawyer_moveit_config\launch\move_group_real.launch`
- `D:\ubuntu20\ros_ws\src\sawyer_moveit_config\launch\trajectory_execution_real.launch.xml`
- `D:\ubuntu20\ros_ws\src\sawyer_moveit_config\config\controllers_real.yaml`

已知真实 Sawyer 链路进展：

- ROS Master、网络、joint_states、endpoint_state、夹爪、TF 已经打通过。
- `/robot/move_group` 能启动。
- RViz Planning Group 使用 `right_arm`。
- FollowJointTrajectory controller 已连接。
- 之前 `right_j6` 限位问题应通过真实 xacro 分支解决，避免加载仿真静态 URDF。
- 之前 `head-right_l2` self collision 问题已处理。

### 4.2 `mt3_real_params.yaml` 当前问题

文件：

`D:\ubuntu20\ros_ws\src\sawyer_gazebo\config\mt3_real_params.yaml`

当前仍包含旧占位相机配置，例如：

- `/io/internal_camera/head_camera/image_raw`
- `/camera/depth/image_raw`
- `/camera/color/camera_info`
- `/camera/depth_registered/points`
- `camera_color_optical_frame`

这些不符合当前 ASC60C/HP60C 实测链路。后续应改为类似：

- `rgb_topic: /ascamera_hp60c/rgb0/image`
- `depth_topic: /ascamera_hp60c/depth0/image_raw`
- `rgb_camera_info_topic: /ascamera_hp60c/rgb0/camera_info`
- `depth_camera_info_topic: /ascamera_hp60c/depth0/camera_info`
- `pointcloud_topic: /ascamera_hp60c/depth0/points`
- `camera_frame: ascamera_hp60c_color_0`
- `target_frame: base`

是否使用 TF 外参还要看最终是否发布了稳定的 `base <- ascamera_hp60c_color_0`。如果没有发布 TF，只能暂时使用硬编码外参，但这不应长期散落在脚本里。

## 5. “要发给别人看的文件”对应关系

如果新聊天或另一个 AI 要看“LangSAM + 3D + MT3/MoveIt 接口”，优先发这些文件。

### 5.1 LangSAM 推理脚本

- `D:\ubuntu20\vision_models\test_asc60c_langsam.py`

用途：真实 RGB 图像进入 LangSAM，输出 `current_mask.npy/png` 和 overlay。

### 5.2 mask 到真实 3D 坐标

- `D:\ubuntu20\ascamera_data\mask_to_base_xyz.py`
- `D:\ubuntu20\ascamera_data\mask_to_base_xyz_top.py`
- `D:\ubuntu20\ascamera_data\mask_to_base_xyz_top_z44.py`
- `D:\ubuntu20\ascamera_data\current_camera_info.json`

用途：把 mask + depth + camera_info + 外参转成 Sawyer `base` 坐标。

### 5.3 MT3 感知接入

- `D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_perception_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_anchor_perception_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_alignment_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\langsam_depth_localization.py`
- `D:\ubuntu20\code\learning_thousand_tasks\mt3_scene_package.py`

用途：把真实相机观测转换成 MT3 需要的 scene/object pose。

### 5.4 MT3 到 MoveIt / Sawyer 执行入口

- `D:\ubuntu20\code\learning_thousand_tasks\real_pipeline_patch\mt3_pipeline_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_pipeline_patch\mt3_cylinder_insert_pipeline_real.py`
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_grasp.py`
- `D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place.py`
- `D:\ubuntu20\ros_ws\start_mt3_real.sh`

用途：看最终 MT3 输出如何送入真实 Sawyer / MoveIt。

### 5.5 真实示教

- `D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\real_kinesthetic_recorder.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_demo_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_anchor_place_demo_real.py`
- `D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_cylinder_insert_demo_real.py`

用途：看真实 Sawyer 的拖动示教如何保存为 MT3 demo。

## 6. 当前主要风险

1. 真实 patch 文件尚未确认是否已合并到 Ubuntu runtime。
   - Windows 镜像里有 `real_*_patch`。
   - ROS 和 Python 实际运行时可能仍在用旧根目录或 `~/ros_ws` 中的文件。
   - 下一步要明确复制/合并策略。

2. `mt3_real_params.yaml` 相机 topic 仍旧。
   - 当前配置会指向 Sawyer head camera 或 Kinect 风格 topic。
   - 与 ASC60C/HP60C topic 不一致。

3. 外参仍有硬编码调试痕迹。
   - `mask_to_base_xyz_top*.py` 使用硬编码 `T_BASE_CAMERA`。
   - 在真实 MT3 pipeline 中应改成配置项或 TF。

4. 高度补偿仍是经验项。
   - `mask_to_base_xyz_top_z44.py` 中 `Z_TOP_OFFSET_M = 0.044` 是实测补偿。
   - 需要按真实物块、桌面高度、抓取 TCP 重新验证。

5. 真实执行器仍需二次核对。
   - 当前能看到 `mt3_sawyer_grasp.py`、`mt3_sawyer_place.py`。
   - 如果 real pipeline 调用的是 `*_real.py` 或期望不同参数，需要补齐或改入口。

6. 仿真和真机共用工作区。
   - 不应全局替换 frame/topic/URDF。
   - 真机改动应放在 `*_real.launch`、`*_real.py`、`mt3_real_params*.yaml` 里，避免破坏 Gazebo。

## 7. 建议的下一步顺序

1. 以 `mt3_sawyer_real_grasp.py` 作为真实 top grasp 已验证基线，不再回到旧 hover 脚本调主链路。
2. 进入真实放置任务前，先补真实专用执行器 `mt3_sawyer_real_place.py` 或等价文件。
3. 第一版真实放置应复用当前 top grasp 的安全和执行框架：`--execute`、`--move_to_start_pose`、急停、MoveIt `/robot` namespace、ASC60C 感知、CSV fallback。
4. 真实放置第一阶段只做：
   - 抓取
   - vertical lift
   - move above place
   - descend to release height
   - gripper.open()
   - retreat upward
5. 暂时不要混入 Gazebo postcheck、复杂 release replay、插入任务 socket 逻辑或自动 success demo 录入。
6. 录制或准备真实 place demo JSON，至少要明确 target、anchor/place target、place release pose 和 release/open event。
7. 给真实 place CSV 增加 target/anchor 感知、planned place、actual release TCP、open event、retreat、manual success/failure reason 等字段。

## 8. 本次审计未做的事

- 没有运行真实 Sawyer。
- 没有启动 ROS launch。
- 没有执行真实机器人轨迹。
- 没有修改 ROS 工作空间中的 launch/yaml/python 文件。
- 没有确认 Ubuntu 虚拟机内实际文件是否与 Windows 镜像完全同步。

本次只做静态扫描和文档更新。

---

## 9. 2026-08-27 放置任务差距审计

当前用户已经完成真实 top grasp，准备进入放置任务。静态扫描后的结论如下。

### 9.1 已有的放置相关代码

仿真/旧版 pick-place 执行器：

`D:\ubuntu20\ros_ws\src\sawyer_gazebo\scripts\mt3_sawyer_place.py`

能力：

- 读取 `/sawyer_auto_grasp/grasp_*` 和 `/sawyer_auto_grasp/place_*` 参数。
- 执行抓取、lift、move above place、下降、`gripper.open()`、retreat。
- 支持 `use_place_release_replay`，从 demo 的 release/open 事件附近 replay 放置段。
- 记录 MoveIt planning/execution timing。

限制：

- 文件头和大量逻辑仍以 Gazebo 为背景。
- 包含 `/gazebo/get_model_state`、Gazebo postcheck、insert 诊断、旧 top grasp 逻辑。
- 不包含当前真实 top grasp 已验证的 ASC60C bridge、mask erosion、top-Z +44 mm、固定起点、急停、CSV fallback 等完整安全链路。

方向放置泛化：

`D:\ubuntu20\code\learning_thousand_tasks\mt3_place_generalization.py`

anchor-place 泛化：

`D:\ubuntu20\code\learning_thousand_tasks\mt3_anchor_place_generalization.py`

真实双物体感知：

`D:\ubuntu20\code\learning_thousand_tasks\real_perception_patch\mt3_anchor_perception_real.py`

真实 anchor-place 示教录制器：

`D:\ubuntu20\code\learning_thousand_tasks\real_kinesthetic_demo_patch\record_anchor_place_demo_real.py`

### 9.2 当前缺失项

- 缺真实放置执行器：`mt3_sawyer_place_real.py` 当前不存在。
- 缺真实放置 demo JSON：当前 Windows 项目中没有 `demo_library/real/recorded` 目录和真实 anchor-place demo。
- 缺真实放置 launch：当前 `mt3_real_grasp.launch` 仍是 grasp 入口，不是 place 入口。
- 缺从当前 `mt3_sawyer_real_grasp.py` 抽出的“抓取前半段 + 抓起后运输/放置”真实流程。
- 缺真实放置 CSV schema：需要记录 target/anchor 感知、planned place、release TCP、open event、retreat、manual success/failure reason。
- 缺真实放置成功判定：真实环境没有 Gazebo GT，不能沿用仿真 postcheck，需要人工标签或外部视觉复检。

### 9.3 建议实现路线

第一版真实放置不要直接复制旧 `mt3_sawyer_place.py` 全部逻辑。建议新建或派生真实专用：

```text
mt3_sawyer_real_place.py
```

最小闭环：

```text
复用 mt3_sawyer_real_grasp.py 的：
  感知更新
  real MoveIt 初始化
  emergency stop
  move_to_start_pose
  mapped bottleneck
  before-close replay
  gripper close
  vertical lift

新增：
  读取 place target
  move above place
  descend to release height
  gripper.open()
  retreat upward
  写真实 place CSV
```

暂时不要混入：

```text
Gazebo postcheck
插入任务 socket 逻辑
复杂 release replay
自动成功录入 demo library
闭环重检测
```
