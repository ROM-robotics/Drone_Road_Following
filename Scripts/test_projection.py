#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import numpy as np
import cv2

class CollisionNode(Node):
    def __init__(self):
        super().__init__('collision_node')
        self.bridge = CvBridge()
        self.K = None
        self.dist_coeffs = None
        self.depth_img = None

        # subs
        self.create_subscription(CameraInfo,
                                 '/camera/depth/camera_info',
                                 self.caminfo_cb, 10)
        self.create_subscription(Image,
                                 '/camera/depth/image',
                                 self.depth_cb, 10)

        # pubs
        self.coll_pub = self.create_publisher(Bool, '/collision', 10)
        self.sol_pub  = self.create_publisher(String, '/collision/solution', 10)

        # run check at 10 Hz
        self.create_timer(0.1, self.check_collisions)


    def caminfo_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3,3)
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info('Got camera intrinsics')
        # only need once
        self.destroy_subscription(self.caminfo_cb)


    def depth_cb(self, msg: Image):
        # depth in meters, float32
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, '32FC1')


    def cspace2collision(self, query_pt, safety_r=0.5, min_pts=5):
        """
        query_pt: np.array([X,Y,Z]) in camera frame
        safety_r: half‐width of shield cube
        """
        if self.K is None or self.depth_img is None:
            return False

        # 1) build 3D shield points
        r = safety_r
        pts3d = np.array([
            query_pt + [ 0,  0,  r],  # center
            query_pt + [-r, -r,  r],  # TL
            query_pt + [ r, -r,  r],  # TR
            query_pt + [ r,  r,  r],  # 
            query_pt + [-r,  r,  r],  # BBRL
        ], dtype=np.float32)

        # 2) project to 2D
        rvec = np.zeros((3,1), dtype=np.float32)
        tvec = np.zeros((3,1), dtype=np.float32)
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, self.dist_coeffs)
        pts2d = pts2d.reshape(-1,2).astype(int)

        # 3) define pixel window TL→BR
        H, W = self.depth_img.shape
        tl = pts2d[1]; tr = pts2d[2]; bl = pts2d[4]
        x0, y0 = np.clip(tl[0], 0, W-1), np.clip(tl[1], 0, H-1)
        x1, y1 = np.clip(tr[0], 0, W-1), np.clip(bl[1], 0, H-1)
        if x1 <= x0 or y1 <= y0:
            return False

        # 4) scan depths
        window = self.depth_img[y0:y1, x0:x1]
        shield_z = pts3d[1,2]
        coll_pts = np.count_nonzero(window < shield_z)
        return (coll_pts >= min_pts)


    def check_collisions(self):
        # query points 7 m ahead in five directions (camera frame):
        queries = {
            'front': np.array([0.0, 0.0, 7.0], dtype=np.float32),
            'left' : np.array([-1.0, 0.0, 7.0], dtype=np.float32),
            'right': np.array([ 1.0, 0.0, 7.0], dtype=np.float32),
            'up'   : np.array([0.0,  1.0, 7.0], dtype=np.float32),
            'down' : np.array([0.0, -1.0, 7.0], dtype=np.float32),
        }
        coll = {}
        for name, pt in queries.items():
            coll[name] = self.cspace2collision(pt, safety_r=0.5)

        # publish front collision flag
        front_coll = Bool(data=coll['front'])
        self.coll_pub.publish(front_coll)

        # decide solution
        solution = 'none'
        if coll['front']:
            # try dodge order: left → right → up → down
            for dir_ in ('left','right','up','down'):
                if not coll[dir_]:
                    solution = dir_
                    break

        self.sol_pub.publish(String(data=solution))


def main(args=None):
    rclpy.init(args=args)
    node = CollisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
