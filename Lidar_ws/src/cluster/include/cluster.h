#include <pcl/kdtree/kdtree.h>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
namespace tdt_radar {
enum class ClusterRegion {
    Near,
    Middle,
    Far,
};

struct ClusterParameters {
    double tolerance;
    int min_size;
    int max_size;
};

class Cluster : public rclcpp::Node {
public:
    Cluster(const rclcpp::NodeOptions& node_options);
    ~Cluster() {}

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr fly_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    fly_pub_;
    std::string region_split_axis_ = "x";
    double near_middle_split_position_ = 7.0;
    double middle_far_split_position_ = 14.0;
    bool near_region_positive_ = false;
    ClusterParameters near_cluster_params_{0.25, 5, 1000};
    ClusterParameters middle_cluster_params_{0.25, 5, 1000};
    ClusterParameters far_cluster_params_{0.25, 5, 1000};
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void flyCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void clusterAndPublish(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher);
    ClusterRegion GetClusterRegion(const pcl::PointXYZ& point) const;
    void AppendClusterCenters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const ClusterParameters& params,
        pcl::PointCloud<pcl::PointXYZ>& output_cloud) const;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
};
}  // namespace tdt_radar
