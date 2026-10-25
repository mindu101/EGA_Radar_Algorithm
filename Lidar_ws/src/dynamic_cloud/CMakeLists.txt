cmake_minimum_required(VERSION 3.8)
project(dynamic_cloud)
set(CMAKE_BUILD_TYPE "RELEASE")
find_package (ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies ()
# 静态点剔除的最近邻搜索用 OpenMP 并行。
find_package(OpenMP REQUIRED)

include_directories(include)

set(DYNAMIC_CLOUD "dynamic_cloud")
file(GLOB_RECURSE DYNAMIC_CLOUD_SRCS "src/*.cpp")

ament_auto_add_library(${DYNAMIC_CLOUD} SHARED
    ${DYNAMIC_CLOUD_SRCS}
)

message(${PCL_LIBRARIES})
target_link_libraries(${DYNAMIC_CLOUD} ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

rclcpp_components_register_node(${DYNAMIC_CLOUD}
    PLUGIN "tdt_radar::DynamicCloud"
    EXECUTABLE ${DYNAMIC_CLOUD}_node
)

target_link_libraries(dynamic_cloud_node ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)
ament_auto_package ()
