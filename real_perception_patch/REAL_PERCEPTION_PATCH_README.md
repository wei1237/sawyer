# Sawyer + ASC60C real perception patch

This patch keeps the existing simulation files unchanged.

## New real-only files

- `mt3_perception_real.py`
- `mt3_alignment_real.py`
- `mt3_anchor_perception_real.py`

## Existing generic file kept unchanged

- `mt3_scene_package.py` — already supports `uint16 mm -> meters` and `depth + mask + K -> XYZ`.

## Existing real config to replace/update

- `mt3_real_params_updated.yaml` should be copied over the existing **real-only** `mt3_real_params.yaml`.

## Design

Real perception uses:

`current_mask.npy + /ascamera_hp60c/depth0/image_raw + /ascamera_hp60c/rgb0/camera_info`

and outputs points in `ascamera_hp60c_color_0`.

`mt3_alignment_real.py` then requires a calibrated TF:

`base <- ascamera_hp60c_color_0`

There is no Sawyer head-camera fallback, no Gazebo empirical offset, and no REP103 frame guess in the real path.

## Copy to Ubuntu

Assuming this folder is saved on Windows under:

`D:\\ubuntu20\\code\\learning_thousand_tasks\\real_perception_patch`

copy the Python files with:

```bash
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_perception_real.py ~/code/learning_thousand_tasks/
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_alignment_real.py ~/code/learning_thousand_tasks/
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_anchor_perception_real.py ~/code/learning_thousand_tasks/
```

The YAML belongs in ROS config:

```bash
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_perception_patch/mt3_real_params_updated.yaml \
  ~/ros_ws/src/sawyer_gazebo/config/mt3_real_params.yaml
```

The original simulation files stay untouched.

## Syntax test

```bash
cd ~/code/learning_thousand_tasks
python3 -m py_compile \
  mt3_perception_real.py \
  mt3_alignment_real.py \
  mt3_anchor_perception_real.py
```

## Important

Do not run autonomous motion from this patch yet. Until ChArUco calibration is accepted and `base <- ascamera_hp60c_color_0` exists in TF, the real alignment module is expected to fail closed rather than invent an extrinsic.
