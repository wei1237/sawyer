# Real Sawyer Kinesthetic Demo Recorder

These files are **real-robot additions**. They do not replace the existing Gazebo recorders.

Files:
- `real_kinesthetic_recorder.py` — shared Zero-G recorder, no Sawyer arm motion commands.
- `record_demo_real.py` — top grasp.
- `record_cuboid_yaw_demo_real.py` — yawed/cuboid top grasp.
- `record_anchor_place_demo_real.py` — complete kinesthetic pick-place.
- `record_cylinder_insert_demo_real.py` — complete kinesthetic pick-insert-release.

Copy all five `.py` files into:

```bash
~/code/learning_thousand_tasks/
```

## Recording controls

The script samples `base <- right_hand` at 30 Hz by default. You physically guide Sawyer using its Zero-G/cuff mode.

- `c` — close the real gripper and record `gripper_close`.
- `o` — open the real gripper. If a grasp already occurred, record `release_open`.
- `t` — for placement/insertion, mark the local terminal bottleneck before the final place/insert descent.
- `s` — stop and save.
- `x` — abort/discard.

The recorder itself sends **no arm trajectory commands**.

## ASC60C snapshot defaults

```text
RGB       /ascamera_hp60c/rgb0/image
Depth     /ascamera_hp60c/depth0/image_raw
CameraInfo /ascamera_hp60c/rgb0/camera_info
```

At recording start it optionally saves a raw RGB/depth/CameraInfo snapshot. This is intentionally independent of the old Sawyer head-camera/Gazebo intrinsics path.

## Important metadata rule

The real wrappers intentionally do **not** use the old simulation Z defaults (`-0.58`, etc.). Until the ASC60C perception path is fully connected, pass real object/anchor coordinates explicitly.

Example top-grasp skeleton:

```bash
cd ~/ros_ws
./intera.sh
source ~/ascam_ws/devel/setup.bash
cd ~/code/learning_thousand_tasks
python3 record_demo_real.py \
  _object_x:=... _object_y:=... _object_z:=... \
  _demo_name:=cube_green_top_grasp_real
```

For anchor placement, pass `_target_x/y/z` and `_anchor_x/y/z`.
For insertion, pass `_target_x/y/z` and `_socket_x/y/z`.

Do not run formal demo recording until the real camera-to-base TF and real perception metadata path have been validated. The recorder itself is safe to dry-test because it never commands the arm, but `c/o` do command the gripper.
