#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <robot_hostname_or_ip> [your_pc_ip] [ros_distro]"
  echo "Example: $0 192.168.1.100 192.168.1.20 noetic"
  exit 2
fi

ROBOT_HOST="$1"
YOUR_IP="${2:-${ROS_IP:-}}"
ROS_DISTRO_NAME="${3:-${ROS_DISTRO:-noetic}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "devel/setup.bash"

export ROS_MASTER_URI="http://${ROBOT_HOST}:11311"
export ROS_NAMESPACE="/robot"
export ROS_USE_SIM_TIME="false"
if [[ -n "${YOUR_IP}" ]]; then
  export ROS_IP="${YOUR_IP}"
  unset ROS_HOSTNAME || true
fi

echo "ROS_MASTER_URI=${ROS_MASTER_URI}"
echo "ROS_IP=${ROS_IP:-}"
echo "ROS_NAMESPACE=${ROS_NAMESPACE}"

rosparam set /use_sim_time false

robot_state_snapshot() {
  timeout 5 rostopic echo -n 1 /robot/state 2>/dev/null || true
}

state_homed_ok() {
  local state_msg="$1"
  if echo "${state_msg}" | grep -q "^homed:"; then
    echo "${state_msg}" | grep -q "homed: True"
  else
    return 0
  fi
}

robot_ready_enabled() {
  local state_msg
  state_msg="$(robot_state_snapshot)"
  echo "${state_msg}" | grep -q "ready: True" &&
    echo "${state_msg}" | grep -q "enabled: True" &&
    echo "${state_msg}" | grep -q "error: False" &&
    echo "${state_msg}" | grep -q "stopped: False" &&
    state_homed_ok "${state_msg}"
}

wait_for_robot_ready_enabled() {
  local timeout_s="${1:-20}"
  local sleep_s="${2:-2}"
  local elapsed=0
  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    if robot_ready_enabled; then
      return 0
    fi
    sleep "${sleep_s}"
    elapsed=$((elapsed + sleep_s))
  done
  return 1
}

wait_for_topic() {
  local topic="$1"
  local label="${2:-${topic}}"
  echo "      Waiting for ${label}..."
  until rostopic list 2>/dev/null | grep -q "^${topic}$"; do
    sleep 1
  done
  echo "      ${label} ready."
}

wait_for_tf() {
  local target_frame="$1"
  local source_frame="$2"
  local label="${target_frame} <- ${source_frame}"
  echo "      Waiting for TF ${label}..."
  until timeout 3 rosrun tf tf_echo "${target_frame}" "${source_frame}" \
      2>/dev/null | grep -q "Translation"; do
    sleep 1
  done
  echo "      TF ${label} ready."
}

TRAJ_PID=""
CAMERA_PID=""
TF_PID=""
MOVEIT_PID=""

cleanup() {
  kill "${TRAJ_PID:-}" "${CAMERA_PID:-}" "${TF_PID:-}" "${MOVEIT_PID:-}" \
    2>/dev/null || true
}
trap cleanup EXIT

echo "[1/7] Checking/enabling Sawyer..."
STATE_MSG="$(robot_state_snapshot)"
if [[ -z "${STATE_MSG}" ]]; then
  echo "ERROR: failed to read /robot/state."
  exit 1
fi

if echo "${STATE_MSG}" | grep -q "error: True"; then
  echo "ERROR: Sawyer reports error=True; inspect and clear fault manually."
  exit 1
fi
if echo "${STATE_MSG}" | grep -q "stopped: True"; then
  echo "ERROR: Sawyer reports stopped=True; check E-stop / stop condition manually."
  exit 1
fi
if ! echo "${STATE_MSG}" | grep -q "ready: True"; then
  echo "ERROR: Sawyer reports ready=False; refusing automatic enable."
  exit 1
fi
if ! state_homed_ok "${STATE_MSG}"; then
  echo "ERROR: Sawyer is not homed; refusing startup motion."
  exit 1
fi

if echo "${STATE_MSG}" | grep -q "enabled: True"; then
  echo "      Sawyer already ready and enabled."
  echo "      Skipping enable_robot.py."
else
  echo "      Sawyer healthy but not enabled. Running enable_robot.py..."
  ENABLE_OK=0
  for attempt in 1 2 3 4 5; do
    echo "      enable attempt ${attempt}/5"
    if rosrun intera_interface enable_robot.py -e; then
      if wait_for_robot_ready_enabled 20 2; then
        ENABLE_OK=1
        break
      fi
    fi
    sleep 2
  done
  if [ "${ENABLE_OK}" -ne 1 ]; then
    echo "ERROR: failed to enable Sawyer."
    exit 1
  fi
fi

echo "[2/7] Starting Intera joint trajectory action server..."
rosrun intera_interface joint_trajectory_action_server.py -m position &
TRAJ_PID=$!

wait_for_topic "/robot/limb/right/follow_joint_trajectory/goal" \
  "Sawyer follow_joint_trajectory action"

echo "[3/7] Starting ASC60C camera..."
roslaunch ascamera hp60c.launch &
CAMERA_PID=$!

wait_for_topic "/ascamera_hp60c/rgb0/image" "ASC60C RGB image"
wait_for_topic "/ascamera_hp60c/depth0/image_raw" "ASC60C depth image"
wait_for_topic "/ascamera_hp60c/rgb0/camera_info" "ASC60C RGB camera_info"

echo "[4/7] Starting ASC60C eye-to-hand TF..."
roslaunch sawyer_description ascamera_eye_to_hand_tf.launch &
TF_PID=$!

wait_for_tf "base" "ascamera_hp60c_color_0"

echo "[5/7] Starting MoveIt real launch..."
roslaunch sawyer_moveit_config demo_real.launch use_sim_time:=false use_rviz:=false &
MOVEIT_PID=$!

echo "[6/7] Waiting for MoveIt action server..."
wait_for_topic "/robot/move_group/goal" "MoveIt move_group action"

echo "[7/7] Starting MT3 real grasp execution..."
roslaunch sawyer_gazebo mt3_real_grasp.launch
