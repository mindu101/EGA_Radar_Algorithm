#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import PointCloud2


class PC2RetimeRelay(Node):
    def __init__(self):
        super().__init__('pc2_retime_relay')

        self.declare_parameter('input_topic',  '/livox/lidar')
        self.declare_parameter('output_topic', '/livox/lidar_sync')
        self.declare_parameter('stamp_mode',   'offset')  # 'now' or 'offset'
        self.declare_parameter('use_system_time', True)
        self.declare_parameter('frame_id_override', '')  # '' means keep original

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.stamp_mode = self.get_parameter('stamp_mode').value
        self.use_system_time = bool(self.get_parameter('use_system_time').value)
        self.frame_id_override = self.get_parameter('frame_id_override').value

        # 用 SYSTEM_TIME（epoch wall time）去对齐你当前 /tf 的时间戳
        self.clock = Clock(clock_type=ClockType.SYSTEM_TIME) if self.use_system_time else self.get_clock()

        self.pub = self.create_publisher(PointCloud2, self.output_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, qos_profile_sensor_data)

        # 只存纳秒整数，避免不同 clock type 的 Time 运算
        self.offset_ns = None
        self.last_in_ns = None

        self.get_logger().info(
            f"Relaying {self.input_topic} -> {self.output_topic}, "
            f"stamp_mode={self.stamp_mode}, use_system_time={self.use_system_time}"
        )

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _ns_to_stamp(ns: int, stamp):
        stamp.sec = int(ns // 1_000_000_000)
        stamp.nanosec = int(ns % 1_000_000_000)

    def cb(self, msg: PointCloud2):
        now_ns = int(self.clock.now().nanoseconds)
        in_ns = self._stamp_to_ns(msg.header.stamp)

        # 检测 rosbag loop / 时间回跳（输入时间戳变小）-> 重置 offset
        if self.last_in_ns is not None and in_ns < self.last_in_ns:
            self.offset_ns = None
        self.last_in_ns = in_ns

        if self.stamp_mode == 'now':
            out_ns = now_ns
        else:  # 'offset'
            if self.offset_ns is None:
                self.offset_ns = now_ns - in_ns
            out_ns = in_ns + self.offset_ns

        self._ns_to_stamp(out_ns, msg.header.stamp)

        if self.frame_id_override:
            msg.header.frame_id = self.frame_id_override

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PC2RetimeRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
