#!/bin/bash
set -euo pipefail

source ~/ros_ws/devel/setup.bash

# MT3_TOP_GRASP_SIM_LAUNCHER_V1 = 2026-08-20_top_grasp_table_0325_env_pose_v1


wait_for_service() {
  local service_name="$1"
  local timeout_s="${2:-60}"

  echo "      Waiting for service ${service_name}..."

  for _ in $(seq 1 "${timeout_s}"); do
    if rosservice list 2>/dev/null | grep -qx "${service_name}"; then
      echo "      OK: ${service_name}"
      return 0
    fi
    sleep 1
  done

  echo "      ERROR: service ${service_name} not available after ${timeout_s}s"
  return 1
}


wait_for_topic_once() {
  local topic_name="$1"
  local timeout_s="${2:-30}"

  echo "      Waiting for topic ${topic_name}..."

  if timeout "${timeout_s}" \
      rostopic echo -n 1 "${topic_name}" >/dev/null 2>&1; then
    echo "      OK: ${topic_name}"
    return 0
  fi

  echo "      ERROR: no message received on ${topic_name} after ${timeout_s}s"
  return 1
}


wait_for_right_joint_state_stream() {
  local timeout_s="${1:-15}"
  local required_messages="${2:-5}"

  echo "      Verifying /robot/joint_states stream before FJT startup..."

  python3 - "${timeout_s}" "${required_messages}" <<'PY'
import sys
import time

import rospy
from sensor_msgs.msg import JointState

timeout_s = float(sys.argv[1])
required = int(sys.argv[2])
wanted = {"right_j%d" % i for i in range(7)}

rospy.init_node(
    "mt3_wait_for_joint_state_stream",
    anonymous=True,
    disable_signals=True,
)

deadline = time.time() + timeout_s
good = 0
last_stamp = None

while time.time() < deadline and not rospy.is_shutdown():
    remaining = max(0.2, min(2.0, deadline - time.time()))
    try:
        msg = rospy.wait_for_message(
            "/robot/joint_states",
            JointState,
            timeout=remaining,
        )
    except Exception:
        good = 0
        continue

    names = set(msg.name)
    if not wanted.issubset(names):
        good = 0
        continue

    # Require several distinct messages, not one stale sample.
    stamp = (
        msg.header.stamp.to_sec()
        if msg.header.stamp is not None
        else 0.0
    )
    if last_stamp is None or stamp != last_stamp:
        good += 1
        last_stamp = stamp

    if good >= required:
        print(
            "      OK: /robot/joint_states is live "
            "(%d consecutive Sawyer joint-state samples)." % good
        )
        sys.exit(0)

print(
    "ERROR: /robot/joint_states did not provide a stable Sawyer stream "
    "within %.1fs." % timeout_s
)
sys.exit(1)
PY
}


robot_ready_enabled() {
  local state_msg

  state_msg="$(
    timeout 5 rostopic echo -n 1 /robot/state 2>/dev/null || true
  )"

  echo "${state_msg}" | grep -q "ready: True" &&
    echo "${state_msg}" | grep -q "enabled: True" &&
    echo "${state_msg}" | grep -q "error: False" &&
    echo "${state_msg}" | grep -q "stopped: False"
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


wait_for_fjt_action_server() {
  local timeout_s="${1:-30}"
  local elapsed=0

  echo "      Waiting for FollowJointTrajectory action server..."

  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    if rostopic info \
        /robot/limb/right/follow_joint_trajectory/status \
        >/dev/null 2>&1; then

      if timeout 3 rostopic echo -n 1 \
          /robot/limb/right/follow_joint_trajectory/status \
          >/dev/null 2>&1; then

        echo "      OK: FollowJointTrajectory action server is alive."
        return 0
      fi
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "      ERROR: FollowJointTrajectory action server did not become ready."
  return 1
}


start_fjt_with_retry() {
  local max_attempts="${1:-3}"
  local attempt
  local pid=""

  for attempt in $(seq 1 "${max_attempts}"); do
    echo "      FJT startup attempt ${attempt}/${max_attempts}"

    # The Intera Limb constructor blocks waiting for /robot/joint_states.
    # Prove the stream is alive immediately before each launch attempt.
    if ! wait_for_right_joint_state_stream 15 5; then
      echo "      WARN: joint-state stream check failed before FJT attempt ${attempt}."
      sleep 2
      continue
    fi

    rosrun intera_interface \
      joint_trajectory_action_server.py \
      -m position &

    pid=$!

    # Give the process a short window to fail fast if Limb initialization
    # cannot subscribe to joint states.
    sleep 2

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "      WARN: FJT process exited during initialization on attempt ${attempt}."
      wait "${pid}" 2>/dev/null || true
      sleep 2
      continue
    fi

    if wait_for_fjt_action_server 20; then
      FJT_PID="${pid}"
      export FJT_PID
      echo "      OK: FJT startup succeeded on attempt ${attempt} (PID ${FJT_PID})."
      return 0
    fi

    echo "      WARN: FJT did not advertise a live action server on attempt ${attempt}."
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 2
  done

  echo "ERROR: FollowJointTrajectory action server failed after ${max_attempts} attempts."
  return 1
}


model_exists() {
  local model_name="$1"

  python3 - "${model_name}" <<'PY'
import sys
import rospy
from gazebo_msgs.msg import ModelStates

model_name = sys.argv[1]

try:
    rospy.init_node(
        "mt3_model_exists_check",
        anonymous=True,
        disable_signals=True,
    )
    msg = rospy.wait_for_message(
        "/gazebo/model_states",
        ModelStates,
        timeout=3.0,
    )
except Exception:
    sys.exit(2)

sys.exit(0 if model_name in msg.name else 1)
PY
}


wait_for_model() {
  local model_name="$1"
  local timeout_s="${2:-20}"

  echo "      Waiting for Gazebo model ${model_name}..."

  for _ in $(seq 1 "${timeout_s}"); do
    if model_exists "${model_name}"; then
      echo "      OK: model ${model_name}"
      return 0
    fi
    sleep 1
  done

  echo "      ERROR: Gazebo model ${model_name} not available after ${timeout_s}s"
  return 1
}


delete_model_if_present() {
  local model_name="$1"

  if model_exists "${model_name}"; then
    echo "      Removing stale experiment model before Sawyer startup motion: ${model_name}"

    rosservice call /gazebo/delete_model \
      "{model_name: '${model_name}'}" >/dev/null || true

    for _ in $(seq 1 20); do
      if ! model_exists "${model_name}"; then
        echo "      OK: removed ${model_name}"
        return 0
      fi
      sleep 0.25
    done

    echo "      ERROR: failed to remove stale Gazebo model ${model_name}"
    return 1
  fi

  echo "      OK: ${model_name} is not present before arm motion."
}


spawn_sdf_model() {
  local model_name="$1"
  local sdf_path="$2"
  local x="$3"
  local y="$4"
  local z="$5"
  local yaw="${6:-0.0}"

  if model_exists "${model_name}"; then
    echo "      Model ${model_name} already exists; deleting stale instance first..."

    rosservice call /gazebo/delete_model \
      "{model_name: '${model_name}'}" >/dev/null

    sleep 0.5
  fi

  echo "      Spawning ${model_name} at x=${x} y=${y} z=${z} yaw=${yaw}"

  rosrun gazebo_ros spawn_model \
    -sdf \
    -file "${sdf_path}" \
    -model "${model_name}" \
    -x "${x}" \
    -y "${y}" \
    -z "${z}" \
    -Y "${yaw}"

  wait_for_model "${model_name}" 20
}


verify_world_launch_has_no_experiment_spawn() {
  local launch_path
  launch_path="$(rospack find sawyer_gazebo)/launch/sawyer_world_grasp.launch"

  echo "      Verifying Top-Grasp world launch does not auto-spawn experiment objects:"
  echo "      ${launch_path}"

  python3 - "${launch_path}" <<'PY'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
root = ET.parse(path).getroot()

bad = []
for node in root.iter("node"):
    if node.attrib.get("type") != "spawn_model":
        continue

    text = " ".join([
        node.attrib.get("name", ""),
        node.attrib.get("args", ""),
    ])

    if any(name in text for name in (
        "grasp_object",
        "green_rectangular_prism",
        "green_short_cylinder",
        "green_sphere",
        "green_insert_cylinder",
        "blue_insert_socket",
    )):
        bad.append(text)

if bad:
    print(
        "ERROR: active experiment-object spawn node still exists "
        "in sawyer_world.launch:"
    )
    for item in bad:
        print("  " + item)
    sys.exit(1)

print(
    "      OK: no active experiment-object spawn node."
)
PY
}


wait_for_right_arm_settled() {
  local timeout_s="${1:-25}"
  local stable_s="${2:-2.5}"
  local vel_threshold="${3:-0.020}"

  echo "      Waiting until Sawyer right arm is physically settled..."
  echo "      requirement: max |joint velocity| < ${vel_threshold} rad/s continuously for ${stable_s}s"

  python3 - "${timeout_s}" "${stable_s}" "${vel_threshold}" <<'PY'
import sys
import time

import rospy
from sensor_msgs.msg import JointState

timeout_s = float(sys.argv[1])
stable_s = float(sys.argv[2])
threshold = float(sys.argv[3])

wanted = {"right_j%d" % i for i in range(7)}

rospy.init_node(
    "mt3_wait_for_right_arm_settled",
    anonymous=True,
    disable_signals=True,
)

deadline = time.time() + timeout_s
stable_since = None
prev_t = None
prev_pos = {}

while time.time() < deadline and not rospy.is_shutdown():
    remaining = max(0.2, min(2.0, deadline - time.time()))

    try:
        msg = rospy.wait_for_message(
            "/robot/joint_states",
            JointState,
            timeout=remaining,
        )
    except Exception:
        stable_since = None
        continue

    now = time.time()
    name_to_idx = {
        name: i
        for i, name in enumerate(msg.name)
    }
    indices = [
        name_to_idx[n]
        for n in sorted(wanted)
        if n in name_to_idx
    ]

    if len(indices) < 7:
        stable_since = None
        continue

    speeds = []
    velocity_usable = (
        len(msg.velocity) == len(msg.name)
        and len(msg.velocity) > 0
    )

    if velocity_usable:
        speeds = [
            abs(float(msg.velocity[i]))
            for i in indices
        ]
    else:
        current = {
            msg.name[i]: float(msg.position[i])
            for i in indices
        }

        if prev_t is not None:
            dt = max(now - prev_t, 1e-3)
            for name, pos in current.items():
                if name in prev_pos:
                    speeds.append(
                        abs(pos - prev_pos[name]) / dt
                    )

        prev_t = now
        prev_pos = current

        if len(speeds) < 7:
            stable_since = None
            continue

    max_speed = max(speeds) if speeds else float("inf")

    if max_speed < threshold:
        if stable_since is None:
            stable_since = now
        elif now - stable_since >= stable_s:
            print(
                "      OK: Sawyer right arm settled "
                "(max joint speed %.4f rad/s)." % max_speed
            )
            sys.exit(0)
    else:
        stable_since = None

print(
    "ERROR: Sawyer right arm did not settle "
    "within %.1fs." % timeout_s
)
sys.exit(1)
PY
}


echo "[0/8] Cleaning stale MT3 trajectory processes..."

pkill -f "joint_trajectory_action_server.py -m position" 2>/dev/null || true
pkill -f "trajectory_converter.py" 2>/dev/null || true

sleep 1


echo "[1/8] Starting Sawyer Gazebo..."

verify_world_launch_has_no_experiment_spawn

roslaunch sawyer_gazebo sawyer_world_grasp.launch \
  electric_gripper:=true \
  use_sim_time:=true &

wait_for_service /gazebo/get_model_state 90
wait_for_service /gazebo/delete_model 90
wait_for_service /robot/controller_manager/list_controllers 90
wait_for_topic_once /gazebo/model_states 30

# If Gazebo was already running, stale experimental models can survive.
# Remove them BEFORE Sawyer is enabled or commanded to the startup pose.
echo "      Clearing stale experiment objects before any Sawyer startup motion..."
delete_model_if_present grasp_object
delete_model_if_present green_rectangular_prism
delete_model_if_present green_short_cylinder
delete_model_if_present green_sphere
delete_model_if_present green_insert_cylinder
delete_model_if_present blue_insert_socket


echo "[2/8] Starting/checking controllers..."

rosservice call /robot/controller_manager/switch_controller "{
  start_controllers: [
    'right_joint_position_controller',
    'joint_state_controller',
    'electric_gripper_controller'
  ],
  stop_controllers: [],
  strictness: 1
}" || true

sleep 1

echo "      Current controller list:"
rosservice call /robot/controller_manager/list_controllers || true


echo "[3/8] Waiting for core Sawyer topics..."

wait_for_topic_once /robot/state 30
wait_for_right_joint_state_stream 20 5


echo "[4/8] Checking/enabling Sawyer and moving to startup pose..."

if robot_ready_enabled; then
  echo "      Sawyer already ready and enabled."
  echo "      Skipping enable_robot.py."
else
  echo "      Sawyer not enabled yet. Running enable_robot.py..."

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


echo "      Confirming stable Sawyer state..."

if ! wait_for_robot_ready_enabled 20 2; then
  echo "ERROR: Sawyer is not ready/enabled."
  exit 1
fi


echo "      Commanding MT3 zero startup pose..."

python3 ~/ros_ws/move_to_mt3_start_pose.py \
  --speed 0.12 \
  --timeout 30

# Do not proceed while Gazebo is still physically moving the arm.
sleep 1
wait_for_right_arm_settled 25 2.5 0.020

echo ""
echo "      Sawyer startup motion is complete and the arm is stationary."
echo "      Experiment objects are still NOT present."
echo ""


echo "[5/8] Starting trajectory execution stack before loading objects..."

if ! start_fjt_with_retry 3; then
  echo ""
  echo "ERROR: could not start FollowJointTrajectory reliably."
  echo "      Current /robot/joint_states sample:"
  timeout 5 rostopic echo -n 1 /robot/joint_states || true
  exit 1
fi

echo "      FollowJointTrajectory topics:"
rostopic list | grep \
  "/robot/limb/right/follow_joint_trajectory" || true


echo "      Starting trajectory converter..."

rosrun sawyer_gazebo trajectory_converter.py &

CONVERTER_PID=$!

sleep 2

if ! kill -0 "${CONVERTER_PID}" 2>/dev/null; then
  echo "ERROR: trajectory_converter.py exited during startup."
  exit 1
fi

echo "      OK: trajectory_converter.py running (PID ${CONVERTER_PID})"


echo "[6/8] Starting MoveIt before loading objects..."

roslaunch sawyer_moveit_config demo.launch &

wait_for_service /robot/compute_ik 60

sleep 3

echo "      Checking MoveIt controller configuration..."
rosparam get /robot/move_group/controller_list


echo "[7/8] Final robot software readiness gate..."

# Recheck the exact dependency that previously failed.
wait_for_right_joint_state_stream 15 5

if ! wait_for_fjt_action_server 10; then
  echo "ERROR: FollowJointTrajectory action server disappeared after MoveIt startup."
  exit 1
fi

if ! kill -0 "${CONVERTER_PID}" 2>/dev/null; then
  echo "ERROR: trajectory_converter.py is no longer running."
  exit 1
fi

if ! wait_for_robot_ready_enabled 10 1; then
  echo "ERROR: Sawyer is no longer ready/enabled before scene spawn."
  exit 1
fi

wait_for_right_arm_settled 10 1.5 0.020

echo ""
echo "      Robot software stack is fully ready."
echo "      Sawyer is stationary."
echo "      Only now are experiment objects allowed to appear."
echo ""


echo "[8/8] Spawning ONE Top-Grasp experiment object LAST..."

wait_for_service /gazebo/spawn_sdf_model 30
wait_for_service /gazebo/delete_model 30

SAWYER_GAZEBO_PATH="$(rospack find sawyer_gazebo)"

# Original Top-Grasp workbench surface is z=0.325 m.
# XY/Z defaults follow the old Top-Grasp scene; yaw defaults to 0 for clean top_grasp.
GRASP_MODEL="${GRASP_MODEL:-cube}"
GRASP_X="${GRASP_X:-0.70}"
GRASP_Y="${GRASP_Y:--0.08}"
GRASP_Z="${GRASP_Z:-0.325}"
GRASP_YAW="${GRASP_YAW:-0.0}"

case "${GRASP_MODEL,,}" in
  cube|box|grasp_object)
    MODEL_NAME="grasp_object"
    SDF_FILE="grasp_object.sdf"
    MODEL_KIND="cube"
    ;;
  rectangular|rectangle|rect|prism|rectangular_prism|green_rectangular_prism)
    MODEL_NAME="green_rectangular_prism"
    SDF_FILE="green_rectangular_prism.sdf"
    MODEL_KIND="rectangular"
    ;;
  cylinder|short_cylinder|green_short_cylinder)
    MODEL_NAME="green_short_cylinder"
    SDF_FILE="green_short_cylinder.sdf"
    MODEL_KIND="cylinder"
    ;;
  sphere|ball|green_sphere)
    MODEL_NAME="green_sphere"
    SDF_FILE="green_sphere.sdf"
    MODEL_KIND="sphere"
    ;;
  *)
    echo "ERROR: unsupported GRASP_MODEL='${GRASP_MODEL}'"
    echo "       valid: cube | rectangular | cylinder | sphere"
    exit 2
    ;;
esac

SDF_PATH="${SAWYER_GAZEBO_PATH}/worlds/${SDF_FILE}"
if [ ! -f "${SDF_PATH}" ]; then
  echo "ERROR: missing SDF: ${SDF_PATH}"
  exit 2
fi

spawn_sdf_model \
  "${MODEL_NAME}" \
  "${SDF_PATH}" \
  "${GRASP_X}" "${GRASP_Y}" "${GRASP_Z}" "${GRASP_YAW}"

sleep 2

echo ""
echo "      Top-Grasp experiment scene ready:"
echo "        model kind = ${MODEL_KIND}"
echo "        Gazebo name = ${MODEL_NAME}"
echo "        SDF         = ${SDF_FILE}"
echo "        pose        = (${GRASP_X}, ${GRASP_Y}, ${GRASP_Z}), yaw=${GRASP_YAW} rad"
echo "        table       = original grasp table, surface z=0.325 m"
echo ""

echo "============================================================"
echo "MT3 Sawyer Top-Grasp simulation stack started successfully."
echo "Robot stack was fully ready BEFORE experiment-object spawn."
echo "============================================================"
echo ""

echo "Sawyer state:"
timeout 5 rostopic echo -n 1 /robot/state || true

echo ""
echo "FollowJointTrajectory:"
rostopic list | grep \
  "/robot/limb/right/follow_joint_trajectory" || true

echo ""
echo "Virtual camera is NOT started by this script."
echo "If needed:"
echo "  rosrun my_sawyer_sim virtual_camera"

echo ""
echo "You can now run:"
echo "  cd ~/code/learning_thousand_tasks"
echo "  python3 mt3_generalize.py ... _experiment_group:=top_grasp"
