cmake_minimum_required(VERSION 3.16)
project(sentry_radar_port_smoke)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rm2026_vision_interface REQUIRED)

if(NOT DEFINED SENTRY_DECIDE_SOURCE_DIR)
  message(FATAL_ERROR "Set SENTRY_DECIDE_SOURCE_DIR to the extracted 27BT-UC/src/sentry_decide")
endif()

add_executable(sentry_radar_port_smoke
  smoke.cpp
  ${SENTRY_DECIDE_SOURCE_DIR}/src/ports_adapter.cpp
  ${SENTRY_DECIDE_SOURCE_DIR}/src/core.cpp)
target_include_directories(sentry_radar_port_smoke PRIVATE
  ${SENTRY_DECIDE_SOURCE_DIR}/include)
ament_target_dependencies(sentry_radar_port_smoke
  rclcpp std_msgs geometry_msgs nav_msgs rm2026_vision_interface)
