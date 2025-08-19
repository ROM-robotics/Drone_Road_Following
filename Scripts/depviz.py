#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthViz(Node):
    def __init__(self):
        super().__init__('depth_viz_node')
        # allow setting your “max” depth via a ROS2 param
        self.declare_parameter('max_depth', 30.0)
        self.max_depth = self.get_parameter('max_depth').value

        self.sub = self.create_subscription(
            Image, '/camera/depth/image', self.depth_callback, 10)
        self.pub = self.create_publisher(
            Image, '/depthviz', 10)
        self.bridge = CvBridge()
        self.get_logger().info(f"DepthViz up—max_depth={self.max_depth}m")

    def depth_callback(self, msg: Image):
        # to numpy (float32)
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # handle special values
        # neg inf → 0, pos inf → max_depth, nan → 0
        depth[np.isneginf(depth)] = 0.0
        depth[np.isposinf(depth)] = self.max_depth
        depth[np.isnan(depth)]    = 0.0

        # back to ROS Image
        out = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        out.header = msg.header
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = DepthViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
