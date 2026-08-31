#!/bin/bash
# MT3 Cube Grasping System — Setup Script
# Run on Ubuntu 20.04 with ROS Noetic installed
# Usage: bash setup.sh

set -e

echo "============================================"
echo "MT3 Cube Grasping System Setup"
echo "============================================"

# 1. Install system dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip3 install --user opencv-python scipy || pip install --user opencv-python scipy

# 2. Set up the mt3_env virtualenv with OpenCV
echo ""
echo "[2/4] Setting up virtual environment..."
VENV_DIR="$(dirname "$0")/mt3_env"
if [ -d "$VENV_DIR" ]; then
    echo "  Virtualenv exists at $VENV_DIR"
    if [ -f "$VENV_DIR/bin/python" ]; then
        "$VENV_DIR/bin/python" -c "import cv2" 2>/dev/null && echo "  OpenCV available in venv" || echo "  NOTE: OpenCV not in venv — will use system packages"
    fi
else
    echo "  Creating new virtualenv..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install opencv-python scipy numpy
fi

# 3. Generate template
echo ""
echo "[3/4] Generating cube geometric template..."
python3 "$(dirname "$0")/demo_library/create_template.py"

# 4. Test demo library
echo ""
echo "[4/4] Testing demo library..."
python3 "$(dirname "$0")/mt3_demo_library.py" --test

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Launch Gazebo:"
echo "     roslaunch sawyer_gazebo sawyer_world.launch"
echo ""
echo "  2. Launch camera relay:"
echo "     rosrun my_sawyer_sim virtual_camera.py"
echo ""
echo "  3. Run MT3 pipeline (dry run first):"
echo "     python3 $(dirname "$0")/mt3_pipeline.py _dry_run:=true"
echo ""
echo "  4. Run full pipeline with grasp:"
echo "     python3 $(dirname "$0")/mt3_pipeline.py"
echo ""
echo "  Or run grasp manually after perception:"
echo "     rosrun sawyer_gazebo mt3_sawyer_grasp.py"
echo "============================================"
