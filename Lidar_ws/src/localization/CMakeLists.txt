cmake_minimum_required(VERSION 3.8)
project(localization)
set(CMAKE_BUILD_TYPE "RELEASE") 
#gicp配准需要release编译

find_package (ament_cmake_auto REQUIRED) 
ament_auto_find_build_dependencies ()
find_package(ament_index_cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pclomp REQUIRED)

set(LOCALIZATION_NODE "localization")
file(GLOB_RECURSE LOCALIZATION_NODE_SRCS "src/*.cpp")

ament_auto_add_library(${LOCALIZATION_NODE} SHARED
    ${LOCALIZATION_NODE_SRCS}
)

target_link_libraries(${LOCALIZATION_NODE}
    ${PCL_LIBRARIES}
    pclomp::registration
)
ament_target_dependencies(${LOCALIZATION_NODE} ament_index_cpp)

rclcpp_components_register_node(${LOCALIZATION_NODE}
    PLUGIN "tdt_radar::Localization"
    EXECUTABLE ${LOCALIZATION_NODE}_node
)

ament_auto_package ()
