# Top Grasp Unified v2 — demo-relative transition-anchor replay

Build marker: `2026-08-19_top_grasp_demo_relative_anchor_v2`

## Runtime files

- `mt3_generalize.py`
- `mt3_sawyer_grasp.py`

Do **not** use `mt3_generalize_top_lift.py`, `mt3_sawyer_grasp_top_lift.py`, or `mt3_pipeline.py` for the new formal Top Grasp rerun.

## Frozen demonstration

The demonstration is not re-recorded and the bottleneck is not re-selected.

- id: `cube_top_grasp_center`
- recording date: `2026-07-31 01:26:19`
- format: `mt3_recorded_v2`
- SHA256: `bf39be7f38a9d21e20077ac4ee008d050868b71ca0a1999d8449cff4a505c5a6`

From this exact demo:

- bottleneck right_hand relative to demo object XY ≈ `[-1.891, -0.663] mm`
- recorded mouth offset relative to right_hand XY ≈ `[+2.695, +0.191] mm`
- therefore bottleneck mouth relative to demo object XY ≈ `[+0.803, -0.471] mm`

The code preserves this demo-relative mouth relation. It does **not** force the bottleneck mouth to the exact live-object center.

## Unified Top Grasp execution

For `experiment_group=top_grasp` only:

1. Retrieve the recorded demonstration and geometrically map its bottleneck to the live scene.
2. Move to the already-existing transition pose: mapped bottleneck + `0.100 m` in Z.
3. At this safe transition height, measure the real fingertip-derived mouth center.
4. Refine only the execution error so that the real mouth center reaches the **mapped demo-relative bottleneck mouth target**.
5. Preserve the corrected actual XY/orientation and lower only in Z to the mapped bottleneck height.
6. Use the actual reached pose as the replay anchor.
7. Replay the complete demonstration trajectory from bottleneck to the recorded close event. No low-height XY correction is allowed after replay starts.
8. Close the gripper at the recorded event.
9. Skip the recorded post-close trajectory and use the same `0.060 m` pure-Z verification lift for every Top Grasp shape.
10. Run the existing post-grasp physical success check.

There is no sphere-specific branch, no partial-replay rescue, and no scripted fallback for unified Top Grasp.

Rotated Top Grasp and other experiment groups remain on the legacy executor path.

## New diagnostics

The compact trial log now includes build/demo identity, demo-relative bottleneck geometry, transition/bottleneck actual mouth alignment, replay-to-close state, before-close hand/mouth/object diagnostics, close-induced object shift, lift/retention metrics, and active success thresholds. Detailed snapshots are also written into the rollout JSON.

Important fields include:

- `experiment_phase`, `pipeline_build_marker`, `executor_build_marker`, `execution_variant`
- `retrieved_demo_recording_date`, `retrieved_demo_format`, `retrieved_demo_sha256`
- `demo_bottleneck_hand_offset_xy`, `demo_bottleneck_mouth_offset_xy`
- `mapped_bottleneck_hand_xy`, `mapped_bottleneck_mouth_target_xy`
- `transition_actual_hand_xyz`, `transition_actual_mouth_xy`, `transition_anchor_error_xy_m`
- `bottleneck_actual_hand_xyz`, `bottleneck_actual_mouth_xy`, `bottleneck_anchor_error_xy_m`
- `replay_to_close_attempted`, `replay_to_close_success`
- `before_close_planned_hand_xyz`, `before_close_actual_hand_xyz`, `before_close_hand_tracking_error_xyz_m`
- `preclose_object_xyz`, `preclose_object_shift_xy_m`
- existing `before_close_mouth_*` fields plus mouth-to-actual-object error
- `postclose_object_xyz`, `close_object_shift_xy_m`
- `lift_attempted`, `lift_success`, `final_lift_delta_m`, `object_retained_after_lift`

## Validation first

For `top_grasp`, the default `experiment_phase` is `validation`. Do not merge validation trials into the final formal table. After motion and logging are checked and the build is frozen, explicitly run the formal matrix with `~experiment_phase:=formal` and populate `~repeat_id` for every repeat.

## VM deployment

```bash
cp /mnt/hgfs2/code/learning_thousand_tasks/mt3_generalize.py \
~/code/learning_thousand_tasks/mt3_generalize.py

cp /mnt/hgfs2/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py

chmod +x \
~/code/learning_thousand_tasks/mt3_generalize.py \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py

python3 -m py_compile \
~/code/learning_thousand_tasks/mt3_generalize.py \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py

grep -n "2026-08-19_top_grasp_demo_relative_anchor_v2" \
~/code/learning_thousand_tasks/mt3_generalize.py \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py

grep -n "demo_relative_transition_anchor_full_replay_to_close_vertical_lift_v2" \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py

grep -n "mapped_bottleneck_mouth_target_xy" \
~/code/learning_thousand_tasks/mt3_generalize.py \
~/ros_ws/src/sawyer_gazebo/scripts/mt3_sawyer_grasp.py
```
