#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2

class StereoSGBMNode(Node):
    def __init__(self):
        super().__init__('stereo_sgbm_node')
        self.bridge = CvBridge()

        # Publishers for disparity and depth
        self.disp_pub = self.create_publisher(Image, '/disparity', 10)
        self.depth_pub = self.create_publisher(Image, '/depth', 10)

        # Subscribers for left/right images and camera info
        self.left_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/front_left/camera_info')
        self.left_img_sub = message_filters.Subscriber(self, Image, '/camera/front_left/image')
        self.right_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/front_right/camera_info')
        self.right_img_sub = message_filters.Subscriber(self, Image, '/camera/front_right/image')

        # Time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_info_sub, self.left_img_sub, self.right_info_sub, self.right_img_sub],
            queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

        # StereoSGBM parameters
        min_disp = -2
        num_disp = 16 * 1  # Must be divisible by 16
        block_size = 3
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def callback(self, left_info, left_img, right_info, right_img):
        # Convert ROS Image to OpenCV
        left_cv = self.bridge.imgmsg_to_cv2(left_img, desired_encoding='mono8')
        right_cv = self.bridge.imgmsg_to_cv2(right_img, desired_encoding='mono8')

        # Compute disparity
        disp = self.sgbm.compute(left_cv, right_cv).astype(np.float32) / 16.0

        # Clean disparity: median filter
        disp_clean = cv2.medianBlur(disp, 5)

        # Get focal length and baseline from CameraInfo
        # fx from P[0], baseline = -P[0][3] / fx
        fx = left_info.k[0]
        baseline = abs(left_info.p[3] / fx)

        # Compute depth: depth = fx * baseline / disparity
        depth = np.zeros(disp_clean.shape, np.float32)
        valid = disp_clean > 0
        depth[valid] = (fx * baseline) / disp_clean[valid]

        # Convert to ROS images
        disp_msg = self.bridge.cv2_to_imgmsg((disp_clean * 16).astype(np.int32), encoding='32SC1')
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')

        # Header sync
        disp_msg.header = left_img.header
        depth_msg.header = left_img.header

        # Publish
        self.disp_pub.publish(disp_msg)
        self.depth_pub.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StereoSGBMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
