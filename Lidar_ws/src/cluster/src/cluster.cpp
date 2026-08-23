#include "cluster.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/duration.hpp>
namespace tdt_radar {

Cluster::Cluster(const rclcpp::NodeOptions& node_options)
    : Node("cluster_node", node_options)
{
    RCLCPP_INFO(this->get_logger(), "cluster_node start");
    this->declare_parameter<std::string>(
        "region_split_axis", region_split_axis_);
    this->declare_parameter<double>(
        "near_middle_split_position", near_middle_split_position_);
    this->declare_parameter<double>(
        "middle_far_split_position", middle_far_split_position_);
    this->declare_parameter<bool>(
        "near_region_positive", near_region_positive_);
    this->declare_parameter<double>(
        "near_cluster_tolerance", near_cluster_params_.tolerance);
    this->declare_parameter<int>(
        "near_min_cluster_size", near_cluster_params_.min_size);
    this->declare_parameter<int>(
        "near_max_cluster_size", near_cluster_params_.max_size);
    this->declare_parameter<double>(
        "middle_cluster_tolerance", middle_cluster_params_.tolerance);
    this->declare_parameter<int>(
        "middle_min_cluster_size", middle_cluster_params_.min_size);
    this->declare_parameter<int>(
        "middle_max_cluster_size", middle_cluster_params_.max_size);
    this->declare_parameter<double>(
        "far_cluster_tolerance", far_cluster_params_.tolerance);
    this->declare_parameter<int>(
        "far_min_cluster_size", far_cluster_params_.min_size);
    this->declare_parameter<int>(
        "far_max_cluster_size", far_cluster_params_.max_size);

    this->get_parameter("region_split_axis", region_split_axis_);
    this->get_parameter(
        "near_middle_split_position", near_middle_split_position_);
    this->get_parameter(
        "middle_far_split_position", middle_far_split_position_);
    this->get_parameter("near_region_positive", near_region_positive_);
    this->get_parameter(
        "near_cluster_tolerance", near_cluster_params_.tolerance);
    this->get_parameter(
        "near_min_cluster_size", near_cluster_params_.min_size);
    this->get_parameter(
        "near_max_cluster_size", near_cluster_params_.max_size);
    this->get_parameter(
        "middle_cluster_tolerance", middle_cluster_params_.tolerance);
    this->get_parameter(
        "middle_min_cluster_size", middle_cluster_params_.min_size);
    this->get_parameter(
        "middle_max_cluster_size", middle_cluster_params_.max_size);
    this->get_parameter(
        "far_cluster_tolerance", far_cluster_params_.tolerance);
    this->get_parameter(
        "far_min_cluster_size", far_cluster_params_.min_size);
    this->get_parameter(
        "far_max_cluster_size", far_cluster_params_.max_size);

    std::transform(
        region_split_axis_.begin(), region_split_axis_.end(),
        region_split_axis_.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (region_split_axis_ != "x" && region_split_axis_ != "y") {
        RCLCPP_WARN(
            this->get_logger(),
            "region_split_axis仅支持x或y，当前为%s，回退到x",
            region_split_axis_.c_str());
        region_split_axis_ = "x";
    }
    if (!std::isfinite(near_middle_split_position_) ||
        !std::isfinite(middle_far_split_position_)) {
        RCLCPP_WARN(
            this->get_logger(),
            "聚类区域分界坐标不是有限数，回退到7.0m和14.0m");
        near_middle_split_position_ =
            near_region_positive_ ? 14.0 : 7.0;
        middle_far_split_position_ =
            near_region_positive_ ? 7.0 : 14.0;
    }
    if ((!near_region_positive_ &&
         near_middle_split_position_ > middle_far_split_position_) ||
        (near_region_positive_ &&
         near_middle_split_position_ < middle_far_split_position_)) {
        RCLCPP_WARN(
            this->get_logger(),
            "聚类区域分界顺序与near_region_positive不一致，自动交换两个分界值");
        std::swap(
            near_middle_split_position_, middle_far_split_position_);
    }
    auto validate_cluster_params = [this](
                                       const char* region,
                                       ClusterParameters& params) {
        if (!std::isfinite(params.tolerance) || params.tolerance <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "%s_cluster_tolerance必须为正有限数，回退到0.25",
                region);
            params.tolerance = 0.25;
        }
        if (params.min_size < 1) {
            RCLCPP_WARN(
                this->get_logger(),
                "%s_min_cluster_size必须大于0，回退到5", region);
            params.min_size = 5;
        }
        if (params.max_size < params.min_size) {
            RCLCPP_WARN(
                this->get_logger(),
                "%s_max_cluster_size必须不小于min，回退到%d",
                region, std::max(1000, params.min_size));
            params.max_size = std::max(1000, params.min_size);
        }
    };
    validate_cluster_params("near", near_cluster_params_);
    validate_cluster_params("middle", middle_cluster_params_);
    validate_cluster_params("far", far_cluster_params_);

    RCLCPP_INFO(
        this->get_logger(),
        "聚类三分区配置: 近端=%.3fm/%d/%d 中间=%.3fm/%d/%d "
        "远端=%.3fm/%d/%d 分界轴=%s 近中=%.3fm 中远=%.3fm 近端位于%s方向",
        near_cluster_params_.tolerance, near_cluster_params_.min_size,
        near_cluster_params_.max_size, middle_cluster_params_.tolerance,
        middle_cluster_params_.min_size, middle_cluster_params_.max_size,
        far_cluster_params_.tolerance, far_cluster_params_.min_size,
        far_cluster_params_.max_size, region_split_axis_.c_str(),
        near_middle_split_position_, middle_far_split_position_,
        near_region_positive_ ? "正" : "负");

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10,
        std::bind(&Cluster::callback, this, std::placeholders::_1));
    fly_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_fly", 10,
        std::bind(&Cluster::flyCallback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10);
    fly_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_fly_cluster", 10);
}

void Cluster::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    clusterAndPublish(msg, pub_);
}

void Cluster::flyCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    clusterAndPublish(msg, fly_pub_);
}

ClusterRegion Cluster::GetClusterRegion(
    const pcl::PointXYZ& point) const
{
    const double coordinate =
        region_split_axis_ == "x" ? point.x : point.y;
    if (near_region_positive_) {
        if (coordinate >= near_middle_split_position_) {
            return ClusterRegion::Near;
        }
        if (coordinate >= middle_far_split_position_) {
            return ClusterRegion::Middle;
        }
        return ClusterRegion::Far;
    }
    if (coordinate <= near_middle_split_position_) {
        return ClusterRegion::Near;
    }
    if (coordinate <= middle_far_split_position_) {
        return ClusterRegion::Middle;
    }
    return ClusterRegion::Far;
}

void Cluster::AppendClusterCenters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const ClusterParameters& params,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud) const
{
    if (cloud->empty()) {
        return;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(params.tolerance);
    ec.setMinClusterSize(params.min_size);
    ec.setMaxClusterSize(params.max_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        pcl::PointXYZ center{};
        for (const int point_index : indices.indices) {
            const auto& point = cloud->points[point_index];
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
        }
        const float point_count =
            static_cast<float>(indices.indices.size());
        center.x /= point_count;
        center.y /= point_count;
        center.z /= point_count;
        output_cloud.points.push_back(center);
    }
}

void Cluster::clusterAndPublish(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher)
{
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr near_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr middle_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr far_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    near_cloud->reserve(cloud->size());
    middle_cloud->reserve(cloud->size());
    far_cloud->reserve(cloud->size());
    for (const auto& point : *cloud) {
        switch (GetClusterRegion(point)) {
        case ClusterRegion::Near:
            near_cloud->push_back(point);
            break;
        case ClusterRegion::Middle:
            middle_cloud->push_back(point);
            break;
        case ClusterRegion::Far:
            far_cloud->push_back(point);
            break;
        }
    }

    pcl::PointCloud<pcl::PointXYZ> out_cloud;
    AppendClusterCenters(
        near_cloud, near_cluster_params_, out_cloud);
    AppendClusterCenters(
        middle_cloud, middle_cluster_params_, out_cloud);
    AppendClusterCenters(
        far_cloud, far_cluster_params_, out_cloud);
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(out_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    publisher->publish(output);
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    RCLCPP_INFO(
        this->get_logger(), "Cluster callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Cluster)
