# MT3 Real Pipeline Patch

This patch adds real-only pipeline entry points. Existing simulation files are not modified.

Files:
- `mt3_pipeline_real.py`
- `mt3_cylinder_insert_pipeline_real.py`
- `mt3_real_params_pipeline.yaml`

Dependencies already added by the previous real-perception patch:
- `mt3_perception_real.py`
- `mt3_alignment_real.py`
- `mt3_anchor_perception_real.py`

Safety defaults:
- `dry_run: true`
- `allow_real_execution: false`
- no Gazebo ground-truth postcheck
- real executor paths point to `mt3_sawyer_grasp_real.py` / `mt3_sawyer_place_real.py`, which must be reviewed/created before motion is enabled.

Ubuntu copy example (assuming Windows folder is shared at `/mnt/hgfs2/code/learning_thousand_tasks/real_pipeline_patch`):

```bash
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_pipeline_patch/mt3_pipeline_real.py \
  ~/code/learning_thousand_tasks/
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_pipeline_patch/mt3_cylinder_insert_pipeline_real.py \
  ~/code/learning_thousand_tasks/
cp -v /mnt/hgfs2/code/learning_thousand_tasks/real_pipeline_patch/mt3_real_params_pipeline.yaml \
  ~/ros_ws/src/sawyer_gazebo/config/mt3_real_params.yaml

cd ~/code/learning_thousand_tasks
python3 -m py_compile mt3_pipeline_real.py mt3_cylinder_insert_pipeline_real.py
```

Do not set `allow_real_execution: true` yet. First review/create the real Sawyer executors and verify workspace, table z, TCP/flange offsets, safe joints, and gripper parameters.
