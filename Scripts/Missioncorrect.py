#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point
from sensor_msgs.msg import Image
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import numpy as np
import math
import cv2


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = np.zeros(4)
    q[0] = cr * cp * cy + sr * sp * sy  # w
    q[1] = sr * cp * cy - cr * sp * sy  # x
    q[2] = cr * sp * cy + sr * cp * sy  # y
    q[3] = cr * cp * sy - sr * sp * cy  # z
    from geometry_msgs.msg import Quaternion
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MissionControllerNode(Node):
    def __init__(self):
        super().__init__('mission_controller_node')
        self.bridge = CvBridge()

        # --- Parameters ---
        self.declare_parameter('target_altitude', 10.0)
        self.declare_parameter('alt_tolerance', 0.3)
        self.declare_parameter('forward_speed', 3.0)
        self.declare_parameter('kp_yaw', 0.5)
        self.declare_parameter('angle_tolerance', 0.2)
        self.declare_parameter('alpha_angle', 0.2)
        self.declare_parameter('kp_centroid', 2.0)
        self.declare_parameter('max_lat_speed', 2.5)
        self.declare_parameter('alpha_vel', 0.3)
        self.declare_parameter('dodge_yaw_kp', 2.0)
        self.declare_parameter('dodge_speed', 1.5)
        self.declare_parameter('dodge_cooldown', 1.0)

        # QoS profiles
        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=1,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pose_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        setpoint_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers & clients
        self.setpoint_pos_pub = self.create_publisher(PoseStamped,
            '/mavros/setpoint_position/local', setpoint_qos)
        self.setpoint_vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', setpoint_qos)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Subscribers: state, pose, road line, collision topics
        self.create_subscription(State, '/mavros/state', self.state_cb, state_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, pose_qos)
        self.create_subscription(Image, '/road_line', self.line_cb, img_qos)
        self.create_subscription(Bool, '/collision', self.collision_cb, 10)
        self.create_subscription(String, '/collision/solution', self.solution_cb, 10)

        # Internal state
        self.current_state = None
        self.current_pose = None
        self.current_yaw = 0.0
        self.initial_pose_captured = False
        self.takeoff_complete = False
        self.offboard_armed = False
        self.last_request = self.get_clock().now()
        self.road_line_img = None
        self.filtered_angle_error = None
        self.prev_lat_vel = None

        # Collision flags
        self.collision = False
        self.collision_solution = 'none'
        self.last_dodge_time = None

        # Timer
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('MissionController with avoidance initialized')

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg
        self.current_yaw = quaternion_to_yaw(msg.pose.orientation)
        if not self.initial_pose_captured and msg.header.stamp.sec > 0:
            self.initial_pose_captured = True
            self.get_logger().info('Initial pose captured')

    def line_cb(self, msg):
        try:
            self.road_line_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Line img error: {e}')

    def collision_cb(self, msg):
        self.collision = msg.data

    def solution_cb(self, msg):
        self.collision_solution = msg.data

    def call_service(self, client, req, name):
        if client.wait_for_service(timeout_sec=1):
            client.call_async(req)
            self.get_logger().info(f'Requested {name}')
        else:
            self.get_logger().warning(f'{name} service not available')

    def send_initial_setpoint(self):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = 'map'
        if self.initial_pose_captured:
            sp.pose = self.current_pose.pose
        else:
            sp.pose.position.z = 0.1
        self.setpoint_pos_pub.publish(sp)

    def control_loop(self):
        now = self.get_clock().now()
        # 1) wait for connection & pose
        if not self.current_state or not self.current_state.connected or not self.initial_pose_captured:
            self.get_logger().info('Waiting for FCU/pose', throttle_duration_sec=5)
            self.send_initial_setpoint()
            return
        # 2) set OFFBOARD & arm
        if not self.offboard_armed:
            self.send_initial_setpoint()
            if (now - self.last_request).nanoseconds / 1e9 > 1.0:
                if self.current_state.mode != 'OFFBOARD':
                    req = SetMode.Request(); req.custom_mode = 'OFFBOARD'
                    self.call_service(self.set_mode_client, req, 'OFFBOARD')
                elif not self.current_state.armed:
                    req = CommandBool.Request(); req.value = True
                    self.call_service(self.arm_client, req, 'ARM')
                else:
                    self.offboard_armed = True
                    self.takeoff_start = now
                    self.get_logger().info('Offboard & armed')
                self.last_request = now
            return
        # 3) takeoff
        target_z = self.get_parameter('target_altitude').value
        tol = self.get_parameter('alt_tolerance').value
        if not self.takeoff_complete:
            if (now - self.takeoff_start).nanoseconds / 1e9 > self.get_parameter('dodge_cooldown').value:
                sp = PoseStamped(); sp.header.stamp = now.to_msg(); sp.header.frame_id='map'
                sp.pose.position.x = self.current_pose.pose.position.x
                sp.pose.position.y = self.current_pose.pose.position.y
                sp.pose.position.z = target_z
                sp.pose.orientation = self.current_pose.pose.orientation
                self.setpoint_pos_pub.publish(sp)
                if abs(self.current_pose.pose.position.z - target_z) < tol:
                    self.takeoff_complete = True; self.get_logger().info('Takeoff done')
            return
        # 4) collision avoidance
        cooldown = self.get_parameter('dodge_cooldown').value
        if self.collision and (not self.last_dodge_time or (now - self.last_dodge_time).nanoseconds/1e9 > cooldown):
            sol = self.collision_solution
            yaw_kp = self.get_parameter('dodge_yaw_kp').value
            dodge_speed = self.get_parameter('dodge_speed').value
            # map solution to yaw error or vertical dodge
            yaw_err = 0.0; dz = 0.0; dy = 0.0
            if sol == 'left': yaw_err = math.radians(30)
            elif sol == 'right': yaw_err = -math.radians(30)
            elif sol == 'up': dz = dodge_speed
            elif sol == 'down': dz = -dodge_speed
            # compute yaw rate
            yaw_rate = np.clip(yaw_kp * yaw_err, -2.0, 2.0)
            cmd = Twist()
            cmd.angular.z = yaw_rate
            cmd.linear.z = dz
            self.setpoint_vel_pub.publish(cmd)
            self.last_dodge_time = now
            self.get_logger().warn(f'Avoiding {sol}')
            return
        # 5) line following
        img = self.road_line_img
        if img is None:
            self.get_logger().warn('No line image, hovering', throttle_duration_sec=2)
            self.setpoint_vel_pub.publish(Twist())
            return
        # threshold red
        red, gr, bl = img[...,2], img[...,1], img[...,0]
        mask = (red>150)&(gr<50)&(bl<50)
        ys, xs = np.where(mask)
        if xs.size==0:
            self.get_logger().warn('Line lost, hover', throttle_duration_sec=2)
            self.setpoint_vel_pub.publish(Twist())
            return
        pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1,1,2)
        mv = cv2.fitLine(pts, cv2.DIST_L2,0,0.01,0.01).flatten()
        vx, vy = mv[0], mv[1]
        angle_line = math.atan2(vy, vx)
        if angle_line >0: angle_line -= math.pi
        desired = -math.pi/2
        err = angle_line - desired
        err = (err+math.pi)%(2*math.pi)-math.pi
        alpha_ang = self.get_parameter('alpha_angle').value
        if self.filtered_angle_error is None: self.filtered_angle_error=err
        else: self.filtered_angle_error = alpha_ang*err + (1-alpha_ang)*self.filtered_angle_error
        kp_yaw = self.get_parameter('kp_yaw').value
        yaw_rate = -kp_yaw * self.filtered_angle_error
        if abs(self.filtered_angle_error) < self.get_parameter('angle_tolerance').value:
            # forward + lateral
            err_x = (np.mean(xs) - img.shape[1]/2)/(img.shape[1]/2)
            raw_lat = -self.get_parameter('kp_centroid').value * err_x * self.get_parameter('max_lat_speed').value
            if self.prev_lat_vel is None: lat = raw_lat
            else: lat = self.get_parameter('alpha_vel').value*raw_lat + (1-self.get_parameter('alpha_vel').value)*self.prev_lat_vel
            self.prev_lat_vel = lat
            fwd = self.get_parameter('forward_speed').value
        else:
            fwd=0.0; lat=0.0
        # convert to local
        ψ=self.current_yaw; u=fwd; v=lat
        vn = u*math.cos(ψ)-v*math.sin(ψ)
        ve = u*math.sin(ψ)+v*math.cos(ψ)
        cmd=Twist()
        cmd.linear.x=vn; cmd.linear.y=ve; cmd.linear.z=0.0; cmd.angular.z=yaw_rate
        self.setpoint_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MissionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
