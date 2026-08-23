#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <chrono>
#include <deque>
#include <string>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <rm2026_vision_interface/msg/radar_warn.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace tdt_radar {
enum class DynamicRegion {
    Near,
    Middle,
    Far,
};

struct BoxFilter {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double z_min;
    double z_max;
};

struct RotatedBoxFilter {
    double center_x;
    double center_y;
    double yaw_deg;
    double length;
    double width;
    double z_min;
    double z_max;
};

class DynamicCloud : public rclcpp::Node {
public:
    DynamicCloud(const rclcpp::NodeOptions& node_options);
    ~DynamicCloud() {}

private:
    int                                              accumulate_time = 3;
    int                                              near_accumulate_time_ = 3;
    int                                              middle_accumulate_time_ = 3;
    int                                              far_accumulate_time_ = 3;
    std::string                                      region_split_axis_ = "x";
    double                                           near_middle_split_position_ = 7.0;
    double                                           middle_far_split_position_ = 14.0;
    bool                                             near_region_positive_ = false;
    double                                           voxel_leaf_size_ = 0.1;
    double                                           near_map_distance_threshold_ = 0.1;
    double                                           middle_map_distance_threshold_ = 0.1;
    double                                           far_map_distance_threshold_ = 0.1;
    int                                              dynamic_thread_num_ = 12;
    int                                              dart_min_points_ = 5;
    int                                              fly_min_points_ = 40;
    BoxFilter main_roi_{0.1, 5.0, 0.5, 5.0, -1000.0, 1000.0};
    BoxFilter dart_exclude_roi_{26.5, 28.0, 3.5, 5.5, 0.0, 2.5};
    BoxFilter home_exclude_roi_{26.5, 28.0, 13.0, 15.0, 0.0, 3.0};
    RotatedBoxFilter map_exclude_rotated_roi_{0.0, 0.0, 0.0, 0.0,
                                              0.0, -1000.0, 1000.0};
    std::vector<BoxFilter> map_exclude_axis_aligned_rois_;
    BoxFilter dart_roi_{27.2226, 27.4111, 3.925, 4.525, 1.7132, 2.4722};
    BoxFilter fly_safe_roi_{25.225, 27.5, 0.2, 2.2, 1.7, 3.0};
    BoxFilter fly_warn_roi_{19.83, 25.3, 0.2, 4.556, 1.7, 3.0};
    BoxFilter fly_alarm_roi_{13.0, 20.5, 0.2, 4.556, 1.7, 3.0};
    BoxFilter fly_detect_roi_{13.0, 25.3, 0.2, 4.556, 1.7, 3.0};
    pcl::PointCloud<pcl::PointXYZ>::Ptr              map_cloud;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        near_accumulated_clouds_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        middle_accumulated_clouds_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        far_accumulated_clouds_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        other_accumulated_clouds_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        fly_accumulated_clouds_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr other_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fly_pub_;
    rclcpp::Publisher<rm2026_vision_interface::msg::RadarWarn>::SharedPtr
         detect_pub_;
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void GetDynamicCloud(pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>& output_cloud,
                         int thread_num);
    DynamicRegion GetDynamicRegion(const pcl::PointXYZ& point) const;
    double GetMapDistanceThreshold(const pcl::PointXYZ& point) const;
    void PushAccumulatedCloud(
        std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds,
        const pcl::PointCloud<pcl::PointXYZ>& cloud, int max_frames);
    void AppendAccumulatedClouds(
        const std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds,
        pcl::PointCloud<pcl::PointXYZ>& output_cloud) const;
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_Tree;
    tf2_ros::Buffer                 tf_buffer_;
    tf2_ros::TransformListener      tf_listener_;
};
}  // namespace tdt_radar
