#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.count = 0
        self.max_images = 100000  # how many pairs to save
        self.save_dir = 'stereo_images'
        os.makedirs(self.save_dir, exist_ok=True)

        self.left_img = None
        self.right_img = None

        self.left_sub = self.create_subscription(Image, '/camera/front_left/image', self.left_callback, 10)
        self.right_sub = self.create_subscription(Image, '/camera/front_right/image', self.right_callback, 10)

    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.save_if_ready()

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.save_if_ready()

    def save_if_ready(self):
        if self.left_img is not None and self.right_img is not None:
            left_path = os.path.join(self.save_dir, f'left_{self.count}.png')
            right_path = os.path.join(self.save_dir, f'right_{self.count}.png')
            cv2.imwrite(left_path, self.left_img)
            cv2.imwrite(right_path, self.right_img)
            self.get_logger().info(f"Saved pair #{self.count}")
            self.count += 1

            self.left_img = None
            self.right_img = None

            if self.count >= self.max_images:
                self.get_logger().info("Max images saved. Shutting down.")
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
