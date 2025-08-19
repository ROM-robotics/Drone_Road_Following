#!/usr/bin/env python3
# depth_line_follower.py – ROS 2 Humble, PX4 + MAVROS
# Follows a red ground line (/road_line) and avoids obstacles from depth camera (/depth_camera).

from __future__ import annotations
import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import Image
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge


# ───────────────────────────── Utility ─────────────────────────────
def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────── Controller Node ───────────────────────
class MissionController(Node):
    # ---------- Tunables ----------
    TARGET_ALT = 7.0        # m
    ALT_TOL    = 0.3         # m
    TKO_TO     = 30.0        # s

    KP_YAW      = 0.5
    KP_CENTER   = 1
    MAX_LAT_SPD = 2.5        # m/s
    FWD_SPD     = 4.0        # m/s cruise

    OBS_THRESH  = 7.0        # m
    AVOID_SPD   = FWD_SPD        # m/s strafe when avoiding

    ANG_TOL     = 0.2        # rad

    ALPHA_ANG   = 0.2        # EMA smoothing
    ALPHA_LAT   = 0.3
    ALPHA_FWD   = 0.25
    # --------------------------------

    def __init__(self) -> None:
        super().__init__("depth_line_follower")

        self.bridge = CvBridge()

        # QoS
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        pose_qos  = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        img_qos   = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        sp_qos    = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers / Services
        self.vel_pub = self.create_publisher(
            Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", sp_qos)
        self.pos_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", sp_qos)

        self.arm_cli  = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")

        # Subscriptions
        self.create_subscription(State, "/mavros/state",
                                 self.state_cb, state_qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose",
                                 self.pose_cb, pose_qos)
        self.create_subscription(Image, "/road_line",
                                 self.line_cb, img_qos)
        self.create_subscription(Image, "/camera/depth/image",
                                 self.depth_cb, img_qos)

        # Internal state
        self.state = State()
        self.pose  = PoseStamped()
        self.yaw   = 0.0

        self.have_pose = False
        self.offboard  = False
        self.takeoff_done = False
        self.last_req = self.get_clock().now()

        self.line_img:  Optional[np.ndarray] = None
        self.depth_img: Optional[np.ndarray] = None

        self.filt_ang: Optional[float] = None
        self.prev_lat: Optional[float] = None
        self.prev_fwd: Optional[float] = None

        # Timer
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Depth-aware line-follower initialised")

    # --------------- Callbacks ----------------
    def state_cb(self, msg: State) -> None:
        self.state = msg

    def pose_cb(self, msg: PoseStamped) -> None:
        self.pose = msg
        self.yaw  = quaternion_to_yaw(msg.pose.orientation)
        if not self.have_pose and msg.header.stamp.sec:
            self.have_pose = True
            self.get_logger().info("Pose locked")

    def line_cb(self, msg: Image) -> None:
        try:
            self.line_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Line img conversion failed: {e}")

    def depth_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if img.dtype == np.uint16:          # mm → m
                img = img.astype(np.float32) * 0.001
            self.depth_img = img
        except Exception as e:
            self.get_logger().warn(f"Depth img conversion failed: {e}")

    # --------------- Helpers ------------------
    def _call(self, cli, req, label: str) -> None:
        if cli.wait_for_service(1.0):
            cli.call_async(req)
            self.get_logger().info(f"{label} request")
        else:
            self.get_logger().warn(f"{label} service unavailable")

    def arm(self) -> None:
        self._call(self.arm_cli, CommandBool.Request(value=True), "Arm")

    def set_mode(self, mode: str) -> None:
        self._call(self.mode_cli, SetMode.Request(custom_mode=mode), f"Mode {mode}")

    # def dummy_sp(self) -> PoseStamped:
    #     sp = PoseStamped()
    #     sp.header.stamp = self.get_clock().now().to_msg()
    #     sp.header.frame_id = "map"
    #     if self.have_pose:
    #         sp.pose = self.pose.pose
    #     else:
    #         sp.pose.position = Point(0.0, 0.0, 0.1)
    #         sp.pose.orientation = euler_to_quaternion(0, 0, 0)
    #     return sp
    def dummy_sp(self) -> PoseStamped:
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"

        if self.have_pose:
            sp.pose = self.pose.pose
        else:
            # Point accepts keyword args or attribute assignment – **not** positionals
            sp.pose.position = Point(x=0.0, y=0.0, z=0.1)
            sp.pose.orientation = euler_to_quaternion(0.0, 0.0, 0.0)

        return sp
    # --------------- Main loop ----------------
    def control_loop(self) -> None:
        now = self.get_clock().now()

        # Wait for FCU + pose
        if not self.state.connected or not self.have_pose:
            self.pos_pub.publish(self.dummy_sp())
            self.get_logger().info("Waiting for FCU/pose…", throttle_duration_sec=5)
            return

        # Switch to OFFBOARD and arm
        if not self.offboard:
            self.pos_pub.publish(self.dummy_sp())
            if (now - self.last_req).nanoseconds / 1e9 > 1.0:
                if self.state.mode != "OFFBOARD":
                    self.set_mode("OFFBOARD")
                elif not self.state.armed:
                    self.arm()
                self.last_req = now

            if self.state.mode == "OFFBOARD" and self.state.armed:
                self.offboard = True
                self.tko_start = now
                self.get_logger().info("OFFBOARD + armed")
            return

        # Take-off
        if not self.takeoff_done:
            if (now - self.tko_start).nanoseconds / 1e9 > self.TKO_TO:
                self.get_logger().error("Take-off timeout")
                return

            sp = PoseStamped()
            sp.header.stamp = now.to_msg()
            sp.header.frame_id = "map"
            sp.pose.position.x = self.pose.pose.position.x
            sp.pose.position.y = self.pose.pose.position.y
            sp.pose.position.z = self.TARGET_ALT
            sp.pose.orientation = self.pose.pose.orientation
            self.pos_pub.publish(sp)

            if abs(self.pose.pose.position.z - self.TARGET_ALT) < self.ALT_TOL:
                self.takeoff_done = True
                self.get_logger().info("Reached target altitude")
            return

        # Line detection
        if self.line_img is None:
            self.vel_pub.publish(Twist())
            self.get_logger().warn("No /road_line image", throttle_duration_sec=2)
            return

        img = self.line_img
        H, W = img.shape[:2]
        cx_img = W / 2.0

        red, green, blue = img[:, :, 2], img[:, :, 1], img[:, :, 0]
        mask = (red > 150) & (green < 50) & (blue < 50)
        ys, xs = np.where(mask)
        if xs.size == 0:
            self.vel_pub.publish(Twist())
            self.get_logger().warn("Line lost", throttle_duration_sec=2)
            return

        pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = math.atan2(vy, vx)
        if angle > 0:
            angle -= math.pi
        raw_ang = ((angle + math.pi / 2) + math.pi) % (2 * math.pi) - math.pi
        self.filt_ang = raw_ang if self.filt_ang is None else (
            self.ALPHA_ANG * raw_ang + (1 - self.ALPHA_ANG) * self.filt_ang)
        ang_err = self.filt_ang

        M = cv2.moments(mask.astype(np.uint8))
        cx_line = (M["m10"] / M["m00"]) if M["m00"] else cx_img
        err_norm = (cx_line - cx_img) / (W / 2)
        raw_lat = -self.KP_CENTER * err_norm * self.MAX_LAT_SPD
        self.prev_lat = raw_lat if self.prev_lat is None else (
            self.ALPHA_LAT * raw_lat + (1 - self.ALPHA_LAT) * self.prev_lat)
        lat_cmd = self.prev_lat

        base_fwd = self.FWD_SPD if abs(ang_err) < self.ANG_TOL else 0.0

        # Depth avoidance
        fwd_cmd = base_fwd
        if self.depth_img is not None:
            dimg = self.depth_img
            rows = slice(int(2 * dimg.shape[0] / 3), dimg.shape[0])
            cols = slice(int(dimg.shape[1] / 3), int(2 * dimg.shape[1] / 3))
            roi = dimg[rows, cols]
            valid = (roi > 0.2) & np.isfinite(roi)
            if np.any(valid):
                min_depth = float(roi[valid].min())
                if min_depth < self.OBS_THRESH:
                    print(f"Obstacle detected at {min_depth:.2f} m")
                    mid_col = roi.shape[1] // 2
                    left_valid  = valid[:, :mid_col]
                    right_valid = valid[:, mid_col:]
                    left_mean  = float(np.nanmean(np.where(left_valid,  roi[:, :mid_col], np.nan)))
                    right_mean = float(np.nanmean(np.where(right_valid, roi[:, mid_col:], np.nan)))
                    lat_cmd = -self.AVOID_SPD if right_mean > left_mean else self.AVOID_SPD
                    tgt_fwd = max(0.0, (min_depth / self.OBS_THRESH) * base_fwd)
                    self.prev_fwd = tgt_fwd if self.prev_fwd is None else (
                        self.ALPHA_FWD * tgt_fwd + (1 - self.ALPHA_FWD) * self.prev_fwd)
                    fwd_cmd = self.prev_fwd

        self.prev_fwd = fwd_cmd if self.prev_fwd is None else self.prev_fwd
        yaw_rate = -self.KP_YAW * ang_err

        # Body → ENU
        ψ = self.yaw
        north = fwd_cmd * math.cos(ψ) - lat_cmd * math.sin(ψ)
        east  = fwd_cmd * math.sin(ψ) + lat_cmd * math.cos(ψ)

        twist = Twist()
        twist.linear.x = float(north)
        twist.linear.y = float(east)
        twist.linear.z = 0.0
        twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(twist)


# ──────────────────────────── Main ────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()