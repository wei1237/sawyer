MT3 REAL TOP-GRASP DEMO — READY PACK
====================================

Files to place in ~/code/learning_thousand_tasks/
  record_demo_real.py
  real_kinesthetic_recorder.py
  mt3_pipeline_real.py
  check_real_demo_ready.py

1) Static check
cd ~/code/learning_thousand_tasks
python3 -m py_compile record_demo_real.py real_kinesthetic_recorder.py mt3_pipeline_real.py check_real_demo_ready.py

2) Read-only live preflight (does NOT move arm or gripper)
python3 check_real_demo_ready.py _mask_path:=/mnt/hgfs2/ascamera_data/current_mask.npy

Expected final line:
READY: software/data path is ready for formal demo recording.

3) Current formal cube metadata from the validated scene
object center XY     = [0.6290, 0.0300] m
object top raw       = -0.4899 m
real top-Z offset    = +0.0440 m
object top corrected = -0.4459 m
cube height          = 0.0450 m
object bottom Z      = -0.4909 m

4) Formal recording command
python3 record_demo_real.py \
  _demo_name:=cube_green_top_grasp_real_v1 \
  _object_x:=0.6290 \
  _object_y:=0.0300 \
  _object_z:=-0.4909 \
  _object_size:="[0.045,0.045,0.045]" \
  _object_z_semantics:=bottom_surface_base \
  _object_top_z_raw:=-0.4899 \
  _object_top_z_corrected:=-0.4459 \
  _real_top_z_offset_m:=0.044 \
  _mask_path:=/mnt/hgfs2/ascamera_data/current_mask.npy

Recorder workflow for simple Top Grasp:
  - keep constrained_zeroG.py -p -r 10 active
  - do NOT press cuff
  - at the clean bottleneck, press ENTER
  - WAIT until log says: Bottleneck locked. Begin the manual interaction trajectory now.
  - manually descend vertically
  - press c at grasp/close
  - manually lift
  - press s to save
  - do NOT press t for simple Top Grasp

On success, the recorder writes:
  demo_library/recorded/cube_green_top_grasp_real_v1.json
  demo_library/rollout_trajectories/<session>/rollout.json
  demo_library/rollout_trajectories/<session>/scene_snapshot/{rgb.png,depth.npy,camera_info.json,mask.npy}
  demo_library/scene_packages/demo_cube_green_top_grasp_real_v1/{rgb.png,depth.npy,segmap.npy,intrinsics.npy,pointcloud.npy,metadata.json}

Important:
  object_x/object_y are object center XY.
  object_z is object BOTTOM/table-contact Z, not corrected top Z.
  +44 mm is applied only to camera-derived live top surfaces. Explicit recorded demo bottom+height metadata is not shifted again.
