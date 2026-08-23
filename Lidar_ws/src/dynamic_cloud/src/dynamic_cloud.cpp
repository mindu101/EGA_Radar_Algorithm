#include "dynamic_cloud.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
#include <stdexcept>
namespace tdt_radar {
namespace {
BoxFilter ReadBoxParam(
    rclcpp::Node& node, const std::string& name,
    const BoxFilter& default_value)
{
    std::vector<double> default_vector = {
        default_value.x_min, default_value.x_max, default_value.y_min,
        default_value.y_max, default_value.z_min, default_value.z_max};
    node.declare_parameter<std::vector<double>>(name, default_vector);

    std::vector<double> value;
    node.get_parameter(name, value);
    if (value.size() != 6) {
        RCLCPP_WARN(
            node.get_logger(),
            "%s 必须包含6个值 [x_min, x_max, y_min, y_max, z_min, z_max]，使用默认值",
            name.c_str());
        return default_value;
    }
    return BoxFilter{value[0], value[1], value[2],
                     value[3], value[4], value[5]};
}

std::vector<BoxFilter> ReadBoxListParam(
    rclcpp::Node& node, const std::string& name)
{
    node.declare_parameter<std::vector<double>>(
        name, std::vector<double>{});

    std::vector<double> value;
    node.get_parameter(name, value);
    if (value.size() % 6 != 0) {
        const std::string message =
            name + " 必须按每6个值组成一个区域 "
                   "[x_min, x_max, y_min, y_max, z_min, z_max]";
        RCLCPP_ERROR(node.get_logger(), "%s", message.c_str());
        throw std::invalid_argument(message);
    }

    std::vector<BoxFilter> boxes;
    boxes.reserve(value.size() / 6);
    for (std::size_t offset = 0; offset < value.size(); offset += 6) {
        const BoxFilter box{
            value[offset], value[offset + 1], value[offset + 2],
            value[offset + 3], value[offset + 4], value[offset + 5]};
        if (std::isnan(box.x_min) || std::isnan(box.x_max) ||
            std::isnan(box.y_min) || std::isnan(box.y_max) ||
            std::isnan(box.z_min) || std::isnan(box.z_max) ||
            box.x_min >= box.x_max || box.y_min >= box.y_max ||
            box.z_min >= box.z_max) {
            const std::string message =
                name + " 第" + std::to_string(offset / 6 + 1) +
                "个区域边界非法，要求各维min < max且不能包含NaN";
            RCLCPP_ERROR(node.get_logger(), "%s", message.c_str());
            throw std::invalid_argument(message);
        }
        boxes.push_back(box);
    }
    return boxes;
}

RotatedBoxFilter ReadRotatedBoxParam(
    rclcpp::Node& node, const std::string& name,
    const RotatedBoxFilter& default_value)
{
    std::vector<double> default_vector = {
        default_value.center_x, default_value.center_y, default_value.yaw_deg,
        default_value.length,   default_value.width,    default_value.z_min,
        default_value.z_max};
    node.declare_parameter<std::vector<double>>(name, default_vector);

    std::vector<double> value;
    node.get_parameter(name, value);
    if (value.size() != 7) {
        RCLCPP_WARN(
            node.get_logger(),
            "%s 必须包含7个值 [center_x, center_y, yaw_deg, length, width, z_min, z_max]，使用默认值",
            name.c_str());
        return default_value;
    }
    return RotatedBoxFilter{value[0], value[1], value[2], value[3],
                            value[4], value[5], value[6]};
}

bool InBox(const pcl::PointXYZ& point, const BoxFilter& box)
{
    return point.x > box.x_min && point.x < box.x_max &&
           point.y > box.y_min && point.y < box.y_max &&
           point.z > box.z_min && point.z < box.z_max;
}

bool InAnyBox(
    const pcl::PointXYZ& point, const std::vector<BoxFilter>& boxes)
{
    return std::any_of(
        boxes.begin(), boxes.end(), [&point](const BoxFilter& box) {
            return InBox(point, box);
        });
}

bool InRotatedBox(const pcl::PointXYZ& point, const RotatedBoxFilter& box)
{
    if (box.length <= 0.0 || box.width <= 0.0 ||
        point.z <= box.z_min || point.z >= box.z_max) {
        return false;
    }

    const double yaw = box.yaw_deg * M_PI / 180.0;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double dx = point.x - box.center_x;
    const double dy = point.y - box.center_y;
    const double local_x = dx * cos_yaw + dy * sin_yaw;
    const double local_y = -dx * sin_yaw + dy * cos_yaw;
    return std::abs(local_x) < box.length * 0.5 &&
           std::abs(local_y) < box.width * 0.5;
}
}  // namespace

DynamicCloud::DynamicCloud(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("dynamic_cloud_node", node_options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Dynamic_cloud Node start");
    // 以下旧参数仅用于兼容历史启动文件；新配置应使用三个区域各自的参数。
    double map_distance_threshold = 0.1;
    std::string half_split_axis = "x";
    double half_split_position = 14.0;
    bool near_half_positive = false;
    std::string default_map_path = "config/RM2026.pcd";
    try {
        default_map_path =
            ament_index_cpp::get_package_share_directory("dynamic_cloud") +
            "/config/RM2026.pcd";
    }
    catch (const std::exception& ex) {
        RCLCPP_WARN(
            this->get_logger(),
            "Failed to resolve map path from ament index: %s", ex.what());
    }
    this->declare_parameter<std::string>("map_path", default_map_path);
    this->declare_parameter<double>("voxel_leaf_size", voxel_leaf_size_);
    this->declare_parameter<int>("accumulate_time", accumulate_time);
    this->declare_parameter<double>(
        "map_distance_threshold", map_distance_threshold);
    this->declare_parameter<std::string>(
        "half_split_axis", half_split_axis);
    this->declare_parameter<double>(
        "half_split_position", half_split_position);
    this->declare_parameter<bool>(
        "near_half_positive", near_half_positive);
    this->declare_parameter<int>("dynamic_thread_num", dynamic_thread_num_);
    this->declare_parameter<int>("dart_min_points", dart_min_points_);
    this->declare_parameter<int>("fly_min_points", fly_min_points_);
    std::string map_path = default_map_path;
    this->get_parameter("map_path", map_path);
    this->get_parameter("voxel_leaf_size", voxel_leaf_size_);
    this->get_parameter("accumulate_time", accumulate_time);
    this->get_parameter("map_distance_threshold", map_distance_threshold);
    this->get_parameter("half_split_axis", half_split_axis);
    this->get_parameter("half_split_position", half_split_position);
    this->get_parameter("near_half_positive", near_half_positive);
    // 兼容旧配置：未显式配置区域参数时，沿用原统一值和二分边界。
    near_accumulate_time_ = accumulate_time;
    middle_accumulate_time_ = accumulate_time;
    far_accumulate_time_ = accumulate_time;
    near_map_distance_threshold_ = map_distance_threshold;
    middle_map_distance_threshold_ = map_distance_threshold;
    far_map_distance_threshold_ = map_distance_threshold;
    region_split_axis_ = half_split_axis;
    near_middle_split_position_ = half_split_position;
    middle_far_split_position_ = half_split_position;
    near_region_positive_ = near_half_positive;
    this->declare_parameter<int>(
        "near_accumulate_time", near_accumulate_time_);
    this->declare_parameter<int>(
        "middle_accumulate_time", middle_accumulate_time_);
    this->declare_parameter<int>(
        "far_accumulate_time", far_accumulate_time_);
    this->declare_parameter<std::string>(
        "region_split_axis", region_split_axis_);
    this->declare_parameter<double>(
        "near_middle_split_position", near_middle_split_position_);
    this->declare_parameter<double>(
        "middle_far_split_position", middle_far_split_position_);
    this->declare_parameter<bool>(
        "near_region_positive", near_region_positive_);
    this->declare_parameter<double>(
        "near_map_distance_threshold", near_map_distance_threshold_);
    this->declare_parameter<double>(
        "middle_map_distance_threshold", middle_map_distance_threshold_);
    this->declare_parameter<double>(
        "far_map_distance_threshold", far_map_distance_threshold_);
    this->get_parameter("near_accumulate_time", near_accumulate_time_);
    this->get_parameter("middle_accumulate_time", middle_accumulate_time_);
    this->get_parameter("far_accumulate_time", far_accumulate_time_);
    this->get_parameter("region_split_axis", region_split_axis_);
    this->get_parameter(
        "near_middle_split_position", near_middle_split_position_);
    this->get_parameter(
        "middle_far_split_position", middle_far_split_position_);
    this->get_parameter("near_region_positive", near_region_positive_);
    this->get_parameter(
        "near_map_distance_threshold", near_map_distance_threshold_);
    this->get_parameter(
        "middle_map_distance_threshold", middle_map_distance_threshold_);
    this->get_parameter(
        "far_map_distance_threshold", far_map_distance_threshold_);
    this->get_parameter("dynamic_thread_num", dynamic_thread_num_);
    this->get_parameter("dart_min_points", dart_min_points_);
    this->get_parameter("fly_min_points", fly_min_points_);
    main_roi_ = ReadBoxParam(*this, "main_roi", main_roi_);
    dart_exclude_roi_ = ReadBoxParam(
        *this, "dart_exclude_roi", dart_exclude_roi_);
    home_exclude_roi_ = ReadBoxParam(
        *this, "home_exclude_roi", home_exclude_roi_);
    map_exclude_rotated_roi_ = ReadRotatedBoxParam(
        *this, "map_exclude_rotated_roi", map_exclude_rotated_roi_);
    map_exclude_axis_aligned_rois_ = ReadBoxListParam(
        *this, "map_exclude_axis_aligned_rois");
    RCLCPP_INFO(
        this->get_logger(), "已加载%zu个地图轴对齐剔除区域",
        map_exclude_axis_aligned_rois_.size());
    dart_roi_ = ReadBoxParam(*this, "dart_roi", dart_roi_);
    fly_safe_roi_ = ReadBoxParam(*this, "fly_safe_roi", fly_safe_roi_);
    fly_warn_roi_ = ReadBoxParam(*this, "fly_warn_roi", fly_warn_roi_);
    fly_alarm_roi_ = ReadBoxParam(*this, "fly_alarm_roi", fly_alarm_roi_);
    fly_detect_roi_ = ReadBoxParam(*this, "fly_detect_roi", fly_detect_roi_);
    if (voxel_leaf_size_ <= 0.0) {
        RCLCPP_WARN(this->get_logger(), "voxel_leaf_size <= 0，回退到0.1");
        voxel_leaf_size_ = 0.1;
    }
    if (accumulate_time < 1) {
        RCLCPP_WARN(this->get_logger(), "accumulate_time < 1，回退到3");
        accumulate_time = 3;
    }
    if (near_accumulate_time_ < 1) {
        RCLCPP_WARN(
            this->get_logger(),
            "near_accumulate_time < 1，回退到accumulate_time=%d",
            accumulate_time);
        near_accumulate_time_ = accumulate_time;
    }
    if (middle_accumulate_time_ < 1) {
        RCLCPP_WARN(
            this->get_logger(),
            "middle_accumulate_time < 1，回退到accumulate_time=%d",
            accumulate_time);
        middle_accumulate_time_ = accumulate_time;
    }
    if (far_accumulate_time_ < 1) {
        RCLCPP_WARN(
            this->get_logger(),
            "far_accumulate_time < 1，回退到accumulate_time=%d",
            accumulate_time);
        far_accumulate_time_ = accumulate_time;
    }
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
            "区域分界坐标不是有限数，回退到7.0m和14.0m");
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
            "区域分界顺序与near_region_positive不一致，自动交换两个分界值");
        std::swap(
            near_middle_split_position_, middle_far_split_position_);
    }
    const double fallback_map_distance_threshold =
        std::isfinite(map_distance_threshold) &&
                map_distance_threshold > 0.0
            ? map_distance_threshold
            : 0.1;
    auto validate_map_distance_threshold = [this, fallback_map_distance_threshold](
                                               const char* name,
                                               double& value) {
        if (std::isfinite(value) && value > 0.0) {
            return;
        }
        RCLCPP_WARN(
            this->get_logger(), "%s必须为正有限数，回退到%.3f",
            name, fallback_map_distance_threshold);
        value = fallback_map_distance_threshold;
    };
    validate_map_distance_threshold(
        "near_map_distance_threshold", near_map_distance_threshold_);
    validate_map_distance_threshold(
        "middle_map_distance_threshold", middle_map_distance_threshold_);
    validate_map_distance_threshold(
        "far_map_distance_threshold", far_map_distance_threshold_);
    if (near_middle_split_position_ == middle_far_split_position_) {
        RCLCPP_WARN(
            this->get_logger(),
            "两个区域分界值相同，中间区域为空；请配置独立分界值以启用三分区");
    }
    if (dynamic_thread_num_ < 1) {
        RCLCPP_WARN(this->get_logger(), "dynamic_thread_num < 1，回退到1");
        dynamic_thread_num_ = 1;
    }
    if (dart_min_points_ < 1) {
        RCLCPP_WARN(this->get_logger(), "dart_min_points < 1，回退到5");
        dart_min_points_ = 5;
    }
    if (fly_min_points_ < 1) {
        RCLCPP_WARN(this->get_logger(), "fly_min_points < 1，回退到40");
        fly_min_points_ = 40;
    }
    auto temp_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *temp_cloud) == -1) {
        PCL_ERROR("Couldn't read file map.pcd \n");
    }
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(temp_cloud);
    const float leaf_size = static_cast<float>(voxel_leaf_size_);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    auto result = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*result);
    map_cloud = result;
    kd_Tree.setInputCloud(map_cloud);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10,
        std::bind(&DynamicCloud::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10);
    other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_other", 10);
    fly_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_fly", 10);
    detect_pub_ = this->create_publisher<rm2026_vision_interface::msg::RadarWarn>(
        "/lidar_detect", 10);
    RCLCPP_INFO(
        this->get_logger(),
        "动态点三分区配置: 近端=%d帧/%.3fm^2 中间=%d帧/%.3fm^2 "
        "远端=%d帧/%.3fm^2 分界轴=%s 近中=%.3fm 中远=%.3fm "
        "近端位于%s方向; 其他/飞行点=%d帧",
        near_accumulate_time_, near_map_distance_threshold_,
        middle_accumulate_time_, middle_map_distance_threshold_,
        far_accumulate_time_, far_map_distance_threshold_,
        region_split_axis_.c_str(), near_middle_split_position_,
        middle_far_split_position_, near_region_positive_ ? "正" : "负",
        accumulate_time);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Dynamic Cloud Launch!");
}

DynamicRegion DynamicCloud::GetDynamicRegion(
    const pcl::PointXYZ& point) const
{
    const double coordinate =
        region_split_axis_ == "x" ? point.x : point.y;
    if (near_region_positive_) {
        if (coordinate >= near_middle_split_position_) {
            return DynamicRegion::Near;
        }
        if (coordinate >= middle_far_split_position_) {
            return DynamicRegion::Middle;
        }
        return DynamicRegion::Far;
    }
    if (coordinate <= near_middle_split_position_) {
        return DynamicRegion::Near;
    }
    if (coordinate <= middle_far_split_position_) {
        return DynamicRegion::Middle;
    }
    return DynamicRegion::Far;
}

double DynamicCloud::GetMapDistanceThreshold(
    const pcl::PointXYZ& point) const
{
    switch (GetDynamicRegion(point)) {
    case DynamicRegion::Near:
        return near_map_distance_threshold_;
    case DynamicRegion::Middle:
        return middle_map_distance_threshold_;
    case DynamicRegion::Far:
        return far_map_distance_threshold_;
    }
    return middle_map_distance_threshold_;
}

void DynamicCloud::PushAccumulatedCloud(
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds,
    const pcl::PointCloud<pcl::PointXYZ>& cloud, int max_frames)
{
    clouds.push_back(cloud.makeShared());
    while (clouds.size() > static_cast<size_t>(max_frames)) {
        clouds.pop_front();
    }
}

void DynamicCloud::AppendAccumulatedClouds(
    const std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud) const
{
    for (const auto& cloud : clouds) {
        output_cloud += *cloud;
    }
}

void DynamicCloud::GetDynamicCloud(
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud, int thread_num)
{
    int                                         K = 1;
    std::vector<std::thread>                    threads;
    thread_num = std::max(1, std::min(thread_num, static_cast<int>(
                                                 input_cloud.points.size())));
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    auto start = std::chrono::system_clock::now();
    int  cloud_size = input_cloud.points.size();
    if (cloud_size == 0) {
        return;
    }
    int  step = (cloud_size + thread_num - 1) / thread_num;
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread([i, step, cloud_size, &clouds,
                                       &input_cloud, this, K]() {
            const int begin = i * step;
            const int end = std::min((i + 1) * step, cloud_size);
            for (int j = begin; j < end; j++) {
                std::vector<int>   pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);
                if (kd_Tree.nearestKSearch(input_cloud.points[j], K,
                                           pointIdxNKNSearch,
                                           pointNKNSquaredDistance) > 0) {
                    const float threshold = static_cast<float>(
                        GetMapDistanceThreshold(input_cloud.points[j]));
                    if (pointNKNSquaredDistance[0] > threshold) {
                        clouds[i].points.push_back(input_cloud.points[j]);
                    }
                }
            }
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    auto end = std::chrono::system_clock::now();
}

void TransformCloud(pcl::PointCloud<pcl::PointXYZ>&      input_cloud,
                    pcl::PointCloud<pcl::PointXYZ>&      output_cloud,
                    geometry_msgs::msg::TransformStamped transform_stamped,
                    int                                  thread_num)
{
    std::vector<std::thread>                    threads;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    auto start = std::chrono::system_clock::now();
    int  cloud_size = input_cloud.points.size();
    int  step = cloud_size / thread_num;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);

    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(
            [i, step, cloud_size, &clouds, &input_cloud, transform]() {
                for (int j = i * step; j < (i + 1) * step && j < cloud_size;
                     j++) {
                    pcl::PointXYZ   point = input_cloud.points[j];
                    Eigen::Vector3f point_vec(point.x, point.y, point.z);
                    Eigen::Vector3f point_out = transform * point_vec;

                    pcl::PointXYZ transformed_point;
                    transformed_point.x = point_out.x();
                    transformed_point.y = point_out.y();
                    transformed_point.z = point_out.z();

                    clouds[i].points.push_back(transformed_point);
                }
            }));
    }

    for (auto& t : threads) {
        t.join();
    }
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    auto end = std::chrono::system_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "transform time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1000.0);
}

void DynamicCloud::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rm2026_vision_interface::msg::RadarWarn lidar_detect;
    auto dart_cloud_filter = [this](const pcl::PointXYZ& point) {
        return InBox(point, dart_roi_);
    };  // 飞镖检测
    auto fly_safe_filter = [this](const pcl::PointXYZ& point) {
        return InBox(point, fly_safe_roi_);
    };  // 飞机起飞

    auto fly_warn_filter = [this](const pcl::PointXYZ& point) {
        return InBox(point, fly_warn_roi_);
    };  // 飞机飞到半场

    auto fly_alarm_filter = [this](const pcl::PointXYZ& point) {
        return InBox(point, fly_alarm_roi_);
    };  // 飞机飞到中场

    auto fly_filter = [this](const pcl::PointXYZ& point) {
        return InBox(point, fly_detect_roi_);
    };  // 飞机检测


    auto receive_cloud = pcl::PointCloud<pcl::PointXYZ>();
    pcl::fromROSMsg(*msg, receive_cloud);

    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    geometry_msgs::msg::TransformStamped transform_stamped;
    auto ta = std::chrono::steady_clock::now();
    try {
        transform_stamped = tf_buffer_.lookupTransform(
            "rm_frame", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Transform error: %s",
                     ex.what());
        return;
    }
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    auto                           transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);
    pcl::transformPointCloud(receive_cloud, transformed_cloud, transform);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ> other_filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_filtered_cloud;
    for (size_t i = 0; i < transformed_cloud.size(); i++) {
        auto& point = transformed_cloud.points[i];
        // 地图中固定干扰区域直接剔除，避免进入动态点和告警检测。
        if (InBox(point, dart_exclude_roi_) ||
            InBox(point, home_exclude_roi_) ||
            InRotatedBox(point, map_exclude_rotated_roi_) ||
            InAnyBox(point, map_exclude_axis_aligned_rois_)) {
            continue;
        }
        if (!InBox(point, main_roi_)) {
            if (dart_cloud_filter(point) || fly_filter(point)) {
                other_filtered_cloud.push_back(point);
                if (fly_filter(point)) {
                    fly_filtered_cloud.push_back(point);
                }
            }
            continue;
        }
        filtered_cloud.push_back(point);
    }
    pcl::PointCloud<pcl::PointXYZ> dynamic_pointcloud;
    GetDynamicCloud(
        filtered_cloud, dynamic_pointcloud, dynamic_thread_num_);

    // 每个点只进入一个区域缓存；历史缓存可保证目标跨越分界时平滑过渡，
    // 不在边界重复点，避免人为增加聚类密度。
    pcl::PointCloud<pcl::PointXYZ> near_dynamic_cloud;
    pcl::PointCloud<pcl::PointXYZ> middle_dynamic_cloud;
    pcl::PointCloud<pcl::PointXYZ> far_dynamic_cloud;
    near_dynamic_cloud.reserve(dynamic_pointcloud.size());
    middle_dynamic_cloud.reserve(dynamic_pointcloud.size());
    far_dynamic_cloud.reserve(dynamic_pointcloud.size());
    for (const auto& point : dynamic_pointcloud) {
        switch (GetDynamicRegion(point)) {
        case DynamicRegion::Near:
            near_dynamic_cloud.push_back(point);
            break;
        case DynamicRegion::Middle:
            middle_dynamic_cloud.push_back(point);
            break;
        case DynamicRegion::Far:
            far_dynamic_cloud.push_back(point);
            break;
        }
    }

    PushAccumulatedCloud(
        near_accumulated_clouds_, near_dynamic_cloud,
        near_accumulate_time_);
    PushAccumulatedCloud(
        middle_accumulated_clouds_, middle_dynamic_cloud,
        middle_accumulate_time_);
    PushAccumulatedCloud(
        far_accumulated_clouds_, far_dynamic_cloud,
        far_accumulate_time_);
    // 飞镖/飞行器点云不属于地面机器人半场聚类，继续沿用旧累积参数。
    PushAccumulatedCloud(
        other_accumulated_clouds_, other_filtered_cloud, accumulate_time);
    PushAccumulatedCloud(
        fly_accumulated_clouds_, fly_filtered_cloud, accumulate_time);

    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;
    AppendAccumulatedClouds(near_accumulated_clouds_, accumulated_cloud);
    AppendAccumulatedClouds(
        middle_accumulated_clouds_, accumulated_cloud);
    AppendAccumulatedClouds(far_accumulated_clouds_, accumulated_cloud);
    pcl::PointCloud<pcl::PointXYZ> other_accumulated_cloud;
    AppendAccumulatedClouds(
        other_accumulated_clouds_, other_accumulated_cloud);
    pcl::PointCloud<pcl::PointXYZ> fly_accumulated_cloud;
    AppendAccumulatedClouds(
        fly_accumulated_clouds_, fly_accumulated_cloud);
    ta = std::chrono::steady_clock::now();
    sensor_msgs::msg::PointCloud2 output;
    accumulated_cloud.header.frame_id = "rm_frame";
    other_accumulated_cloud.header.frame_id = "rm_frame";
    fly_accumulated_cloud.header.frame_id = "rm_frame";
    pcl::toROSMsg(accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    pub_->publish(output);

    pcl::toROSMsg(other_accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    other_pub_->publish(output);

    pcl::toROSMsg(fly_accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    fly_pub_->publish(output);

    pcl::PointCloud<pcl::PointXYZ> dart_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (dart_cloud_filter(point)) {
            dart_cloud.push_back(point);
        }
    }

    if (dart_cloud.size() > static_cast<size_t>(dart_min_points_)) {
        // 终端不再输出飞镖告警，检测状态仍正常发布。
        // RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Find dart cloud!");
        // RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Dart cloud size: %zu",
        //             dart_cloud.size());
        lidar_detect.dart_state = 1;
    }

    pcl::PointCloud<pcl::PointXYZ> fly_safe_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_warn_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_alarm_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_cloud;

    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (fly_safe_filter(point)) {
            fly_safe_cloud.push_back(point);
        }
        if (fly_warn_filter(point)) {
            fly_warn_cloud.push_back(point);
        }
        if (fly_alarm_filter(point)) {
            fly_alarm_cloud.push_back(point);
        }
    }
    if (fly_safe_cloud.size() > static_cast<size_t>(fly_min_points_)) {
        lidar_detect.fly_state = 1;
    }

    if (fly_warn_cloud.size() > static_cast<size_t>(fly_min_points_)) {
        lidar_detect.fly_state = 2;
    }

    if (fly_alarm_cloud.size() > static_cast<size_t>(fly_min_points_)) {
        lidar_detect.fly_state = 3;
    }

    switch (lidar_detect.fly_state) {
    case 0:
        break;
    case 1:
        // RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
        //             "Safe fly object detected!");
        break;
    case 2:
        // RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
        //             "Warn fly object detected!");
        break;
    case 3:
        // RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
        //              "Alarm fly object detected!");
        break;
    default:
        break;
    }

    detect_pub_->publish(lidar_detect);
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "Dynamic callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::DynamicCloud)
