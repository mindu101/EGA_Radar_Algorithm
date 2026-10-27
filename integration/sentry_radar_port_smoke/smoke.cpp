#include "sentry_decide/ports_adapter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rm2026_vision_interface/msg/sentry_radar_snapshot.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("sentry_radar_port_smoke");
  sentry_decide::RosPortsConfig cfg;
  cfg.radar_topic = "/smoke/radar_snapshot";
  cfg.radar_stale_ms = 200.0;
  cfg.radar_lost_ms = 400.0;
  sentry_decide::RosPorts ports(node.get(), cfg);
  sentry_decide::FrameTransform transform;
  transform.enabled = true;
  transform.dx = 1.0;
  transform.dy = 2.0;
  ports.configure_radar(transform, cfg.radar_stale_ms, cfg.radar_lost_ms);
  if (ports.radar_state() != sentry_decide::kRadarLost) return 1;

  using Snapshot = rm2026_vision_interface::msg::SentryRadarSnapshot;
  auto publisher = node->create_publisher<Snapshot>(cfg.radar_topic, rclcpp::QoS(1).reliable());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  Snapshot message;
  message.header.frame_id = cfg.radar_frame_id;
  message.pipeline_epoch = 1;
  message.sequence = 1;
  message.faction = Snapshot::FACTION_RED;
  message.source_age_ms = 0.0f;
  message.kills = -1;
  message.enemies[0].robot_id = 101;
  message.enemies[0].robot_type = 1;
  message.enemies[0].known = true;
  message.enemies[0].visible = true;
  message.enemies[0].x_m = 5.0f;
  message.enemies[0].y_m = 6.0f;

  bool received = false;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    publisher->publish(message);
    executor.spin_some();
    auto enemies = ports.radar_enemies();
    if (enemies[0].known && enemies[0].id == 101 &&
        std::abs(enemies[0].rel.x - 6.0) < 1e-6 &&
        std::abs(enemies[0].rel.y - 8.0) < 1e-6)
    {
      received = true;
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  if (!received || ports.radar_state() != sentry_decide::kRadarOk) return 2;

  std::this_thread::sleep_for(250ms);
  if (ports.radar_state() != sentry_decide::kRadarStale ||
      ports.radar_enemies()[0].known) return 3;
  std::this_thread::sleep_for(200ms);
  if (ports.radar_state() != sentry_decide::kRadarLost ||
      ports.radar_enemies()[0].known) return 4;

  std::cout << "radar port smoke passed" << std::endl;
  executor.remove_node(node);
  rclcpp::shutdown();
  return 0;
}
