#!/usr/bin/env python3
"""
Bridge node: relays Gazebo camera feed from /io/internal_camera/ to standard topics.

The Gazebo URDF plugins publish to:
  /io/internal_camera/head_camera/image_raw          (RGB)
  /io/internal_camera/head_camera/camera_info
  /io/internal_camera/head_camera/depth/image_raw     (Depth - NEW)
  /io/internal_camera/head_camera/depth/camera_info   (Depth info - NEW)
  /io/internal_camera/right_hand_camera/image_raw
  /io/internal_camera/right_hand_camera/camera_info

This node relays them to:
  /head_camera/image_raw  +  /head_camera/camera_info
  /head_camera/depth/image_raw  +  /head_camera/depth/camera_info
  /right_hand_camera/image_raw  +  /right_hand_camera/camera_info
"""
import rospy
from sensor_msgs.msg import Image, CameraInfo


class CameraRelay:
    """Relays image_raw and camera_info from one namespace to another."""

    def __init__(self, source_ns, target_ns):
        self.target_ns = target_ns
        self.image_pub = rospy.Publisher(
            f"{target_ns}/image_raw", Image, queue_size=10)
        self.info_pub = rospy.Publisher(
            f"{target_ns}/camera_info", CameraInfo, queue_size=10, latch=True)

        self.image_sub = rospy.Subscriber(
            f"{source_ns}/image_raw", Image, self._image_cb, queue_size=5)
        self.info_sub = rospy.Subscriber(
            f"{source_ns}/camera_info", CameraInfo, self._info_cb, queue_size=5)

        rospy.loginfo(f"CameraRelay: {source_ns} -> {target_ns}")

    def _image_cb(self, msg):
        msg.header.frame_id = self.target_ns.lstrip("/")
        self.image_pub.publish(msg)

    def _info_cb(self, msg):
        msg.header.frame_id = self.target_ns.lstrip("/")
        self.info_pub.publish(msg)


class DepthRelay:
    """Relays depth image and depth camera_info from one namespace to another."""

    def __init__(self, source_ns, target_ns):
        self.target_ns = target_ns
        self.depth_image_pub = rospy.Publisher(
            f"{target_ns}/image_raw", Image, queue_size=10)
        self.depth_info_pub = rospy.Publisher(
            f"{target_ns}/camera_info", CameraInfo, queue_size=10, latch=True)

        self.depth_image_sub = rospy.Subscriber(
            f"{source_ns}/image_raw", Image, self._depth_cb, queue_size=5)
        self.depth_info_sub = rospy.Subscriber(
            f"{source_ns}/camera_info", CameraInfo, self._info_cb, queue_size=5)

        rospy.loginfo(f"DepthRelay:  {source_ns} -> {target_ns}")

    def _depth_cb(self, msg):
        msg.header.frame_id = self.target_ns.lstrip("/")
        self.depth_image_pub.publish(msg)

    def _info_cb(self, msg):
        msg.header.frame_id = self.target_ns.lstrip("/")
        self.depth_info_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("camera_relay")

    # Head camera RGB: Gazebo -> standard topic
    head_relay = CameraRelay(
        source_ns="/io/internal_camera/head_camera",
        target_ns="/head_camera"
    )

    # Head camera DEPTH: Gazebo -> standard topic (NEW)
    head_depth_relay = DepthRelay(
        source_ns="/io/internal_camera/head_camera/depth",
        target_ns="/head_camera/depth"
    )

    # Right hand camera: Gazebo -> standard topic
    wrist_relay = CameraRelay(
        source_ns="/io/internal_camera/right_hand_camera",
        target_ns="/right_hand_camera"
    )

    rospy.loginfo("Camera relay active — bridging Gazebo cameras (RGB + Depth) to standard topics")
    rospy.spin()
