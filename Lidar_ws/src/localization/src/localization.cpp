#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <Eigen/Geometry>

namespace tdt_radar {
struct Grid {
    pcl::PointXYZ farthestPoint;
    double maxDistance = -1.0;
};

struct FartherCandidate {
    Grid grid;
    int confirmations = 0;
};

using GridKey = std::pair<int, int>;
using GridMap = std::map<GridKey, Grid>;
using CandidateMap = std::map<GridKey, FartherCandidate>;

class Localization : public rclcpp::Node {
public:
    Localization(const rclcpp::NodeOptions& node_options) : Node("localization", node_options) {
        this->declare_parameter<std::string>("map_path", "");
        this->declare_parameter<int>("accumulate_time", accumulate_time);
        this->declare_parameter<std::vector<double>>(
            "initial_extrinsics", std::vector<double>{});
        this->declare_parameter<std::vector<double>>(
            "initial_translation", std::vector<double>{});
        this->declare_parameter<std::vector<double>>(
            "initial_rpy", std::vector<double>{});
        this->declare_parameter<bool>("initial_extrinsics_are_parent_to_child", true);
        this->declare_parameter<std::string>("backend", "gicp");
        this->declare_parameter<bool>("enable_registration", true);
        this->declare_parameter<bool>(
            "continuous_registration", continuous_registration_);
        this->declare_parameter<bool>(
            "accept_unconverged_result", accept_unconverged_result_);
        this->declare_parameter<double>(
            "registration_interval_sec", registration_interval_sec_);
        this->declare_parameter<int>(
            "max_registration", max_registration_);
        this->declare_parameter<bool>("use_initial_guess", true);
        this->declare_parameter<double>("fitness_threshold", 0.2);
        this->declare_parameter<double>(
            "transformation_epsilon", transformation_epsilon_);
        this->declare_parameter<double>(
            "source_voxel_leaf_size", source_voxel_leaf_size_);
        this->declare_parameter<double>(
            "target_voxel_leaf_size", target_voxel_leaf_size_);
        this->declare_parameter<bool>(
            "enable_background_filter", enable_background_filter_);
        this->declare_parameter<double>(
            "background_angular_bin_deg", background_angular_bin_deg_);
        this->declare_parameter<double>(
            "background_range_tolerance", background_range_tolerance_);
        this->declare_parameter<int>(
            "background_candidate_confirm_count",
            background_candidate_confirm_count_);
        this->declare_parameter<double>(
            "background_candidate_tolerance",
            background_candidate_tolerance_);
        this->declare_parameter<bool>(
            "registration_reset_auto", registration_reset_auto_);
        this->declare_parameter<double>(
            "registration_reset_inlier_distance",
            registration_reset_inlier_distance_);
        this->declare_parameter<double>(
            "registration_reset_inlier_ratio_threshold",
            registration_reset_inlier_ratio_threshold_);
        this->declare_parameter<int>(
            "registration_reset_min_points", registration_reset_min_points_);
        this->declare_parameter<int>(
            "registration_reset_min_age_frames",
            registration_reset_min_age_frames_);
        this->declare_parameter<int>(
            "registration_reset_freeze_frames",
            registration_reset_freeze_frames_);
        this->declare_parameter<int>("stable_required_count", stable_required_count_);
        this->declare_parameter<double>("max_translation_jump", max_translation_jump_);
        this->declare_parameter<double>("max_rotation_jump_deg", max_rotation_jump_deg_);
        this->declare_parameter<double>("transform_smooth_alpha", transform_smooth_alpha_);
        this->declare_parameter<double>(
            "gicp.max_correspondence_distance",
            gicp_max_correspondence_distance_);
        this->declare_parameter<int>(
            "gicp.max_iteration", gicp_max_iteration_);
        this->declare_parameter<double>(
            "gicp.euclidean_fitness_epsilon", gicp_fitness_epsilon_);
        this->declare_parameter<double>(
            "gicp.rotation_epsilon", gicp_rotation_epsilon_);
        this->declare_parameter<int>(
            "gicp.correspondence_randomness",
            gicp_correspondence_randomness_);
        this->declare_parameter<int>(
            "gicp.maximum_optimizer_iterations",
            gicp_maximum_optimizer_iterations_);
        this->declare_parameter<double>("ndt.resolution", ndt_resolution_);
        this->declare_parameter<int>(
            "ndt.max_iteration", ndt_max_iteration_);
        this->declare_parameter<double>("ndt.step_size", ndt_step_size_);
        this->declare_parameter<double>(
            "ndt.fine.resolution", ndt_fine_resolution_);
        this->declare_parameter<int>(
            "ndt.fine.max_iteration", ndt_fine_max_iteration_);
        this->declare_parameter<double>(
            "ndt.fine.step_size", ndt_fine_step_size_);
        this->declare_parameter<double>(
            "ndt.outlier_ratio", ndt_outlier_ratio_);
        this->declare_parameter<int>("ndt.num_threads", ndt_num_threads_);
        this->declare_parameter<std::string>(
            "ndt.neighborhood_search", ndt_neighborhood_search_);

        std::string default_map_path = "config/RM2026.pcd";
        try {
            default_map_path =
                ament_index_cpp::get_package_share_directory("dynamic_cloud") +
                "/config/RM2026.pcd";
        } catch (const std::exception& ex) {
            RCLCPP_WARN(this->get_logger(),
                        "Failed to resolve map path from ament index: %s",
                        ex.what());
        }
        std::string target_pcd_file =
            this->get_parameter("map_path").as_string();
        if (target_pcd_file.empty()) {
            target_pcd_file = default_map_path;
        }
        accumulate_time = this->get_parameter("accumulate_time").as_int();
        backend_ = this->get_parameter("backend").as_string();
        if (backend_ != "gicp" && backend_ != "ndt" &&
            backend_ != "gicp_ndt") {
            const std::string message =
                "lidar_pipeline::localization::backend仅支持"
                "'gicp'、'ndt'或'gicp_ndt'，"
                "当前值为'" + backend_ + "'";
            RCLCPP_FATAL(this->get_logger(), "%s", message.c_str());
            throw std::invalid_argument(message);
        }
        enable_registration_ =
            this->get_parameter("enable_registration").as_bool();
        continuous_registration_ =
            this->get_parameter("continuous_registration").as_bool();
        accept_unconverged_result_ =
            this->get_parameter("accept_unconverged_result").as_bool();
        registration_interval_sec_ =
            this->get_parameter("registration_interval_sec").as_double();
        registration_interval_sec_ =
            std::max(registration_interval_sec_, 0.0);
        max_registration_ = static_cast<int>(
            this->get_parameter("max_registration").as_int());
        if (max_registration_ < 1) {
            RCLCPP_WARN(
                this->get_logger(),
                "max_registration < 1，回退到100");
            max_registration_ = 100;
        }
        use_initial_guess =
            this->get_parameter("use_initial_guess").as_bool();
        fitness_threshold =
            this->get_parameter("fitness_threshold").as_double();
        transformation_epsilon_ =
            this->get_parameter("transformation_epsilon").as_double();
        source_voxel_leaf_size_ =
            this->get_parameter("source_voxel_leaf_size").as_double();
        target_voxel_leaf_size_ =
            this->get_parameter("target_voxel_leaf_size").as_double();
        enable_background_filter_ =
            this->get_parameter("enable_background_filter").as_bool();
        background_angular_bin_deg_ =
            this->get_parameter("background_angular_bin_deg").as_double();
        background_range_tolerance_ =
            this->get_parameter("background_range_tolerance").as_double();
        background_candidate_confirm_count_ = static_cast<int>(
            this->get_parameter("background_candidate_confirm_count").as_int());
        background_candidate_tolerance_ =
            this->get_parameter("background_candidate_tolerance").as_double();
        registration_reset_auto_ =
            this->get_parameter("registration_reset_auto").as_bool();
        registration_reset_inlier_distance_ =
            this->get_parameter("registration_reset_inlier_distance").as_double();
        registration_reset_inlier_ratio_threshold_ =
            this->get_parameter("registration_reset_inlier_ratio_threshold")
                .as_double();
        registration_reset_min_points_ = static_cast<int>(
            this->get_parameter("registration_reset_min_points").as_int());
        registration_reset_min_age_frames_ = static_cast<int>(
            this->get_parameter("registration_reset_min_age_frames").as_int());
        registration_reset_freeze_frames_ = static_cast<int>(
            this->get_parameter("registration_reset_freeze_frames").as_int());
        stable_required_count_ = static_cast<int>(
            this->get_parameter("stable_required_count").as_int());
        stable_required_count_ = std::max(stable_required_count_, 1);
        max_translation_jump_ =
            this->get_parameter("max_translation_jump").as_double();
        max_rotation_jump_deg_ =
            this->get_parameter("max_rotation_jump_deg").as_double();
        transform_smooth_alpha_ =
            this->get_parameter("transform_smooth_alpha").as_double();
        gicp_max_correspondence_distance_ =
            this->get_parameter("gicp.max_correspondence_distance").as_double();
        gicp_max_iteration_ = static_cast<int>(
            this->get_parameter("gicp.max_iteration").as_int());
        gicp_max_iteration_ = std::max(gicp_max_iteration_, 1);
        gicp_fitness_epsilon_ =
            this->get_parameter("gicp.euclidean_fitness_epsilon").as_double();
        gicp_rotation_epsilon_ =
            this->get_parameter("gicp.rotation_epsilon").as_double();
        gicp_correspondence_randomness_ = static_cast<int>(
            this->get_parameter("gicp.correspondence_randomness").as_int());
        gicp_maximum_optimizer_iterations_ = static_cast<int>(
            this->get_parameter("gicp.maximum_optimizer_iterations").as_int());
        ndt_resolution_ =
            this->get_parameter("ndt.resolution").as_double();
        ndt_max_iteration_ = static_cast<int>(
            this->get_parameter("ndt.max_iteration").as_int());
        ndt_max_iteration_ = std::max(ndt_max_iteration_, 1);
        ndt_step_size_ =
            this->get_parameter("ndt.step_size").as_double();
        ndt_fine_resolution_ =
            this->get_parameter("ndt.fine.resolution").as_double();
        ndt_fine_max_iteration_ = static_cast<int>(
            this->get_parameter("ndt.fine.max_iteration").as_int());
        ndt_fine_max_iteration_ =
            std::max(ndt_fine_max_iteration_, 1);
        ndt_fine_step_size_ =
            this->get_parameter("ndt.fine.step_size").as_double();
        ndt_outlier_ratio_ =
            this->get_parameter("ndt.outlier_ratio").as_double();
        ndt_num_threads_ = static_cast<int>(
            this->get_parameter("ndt.num_threads").as_int());
        ndt_neighborhood_search_ =
            this->get_parameter("ndt.neighborhood_search").as_string();
        transform_smooth_alpha_ = std::clamp(transform_smooth_alpha_, 0.0, 1.0);
        if (accumulate_time < 1) {
            RCLCPP_WARN(
                this->get_logger(), "accumulate_time < 1，回退到1");
            accumulate_time = 1;
        }
        if (source_voxel_leaf_size_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "source_voxel_leaf_size <= 0，回退到0.1");
            source_voxel_leaf_size_ = 0.1;
        }
        if (target_voxel_leaf_size_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "target_voxel_leaf_size <= 0，回退到1.0");
            target_voxel_leaf_size_ = 1.0;
        }
        if (background_angular_bin_deg_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "background_angular_bin_deg <= 0，回退到0.1");
            background_angular_bin_deg_ = 0.1;
        }
        if (background_range_tolerance_ < 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "background_range_tolerance < 0，回退到0.1");
            background_range_tolerance_ = 0.1;
        }
        background_candidate_confirm_count_ =
            std::max(background_candidate_confirm_count_, 1);
        if (background_candidate_tolerance_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "background_candidate_tolerance <= 0，回退到0.15");
            background_candidate_tolerance_ = 0.15;
        }
        if (registration_reset_inlier_distance_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "registration_reset_inlier_distance <= 0，回退到0.25");
            registration_reset_inlier_distance_ = 0.25;
        }
        if (registration_reset_inlier_ratio_threshold_ <= 0.0 ||
            registration_reset_inlier_ratio_threshold_ >= 1.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "registration_reset_inlier_ratio_threshold不在(0, 1)内，回退到0.3");
            registration_reset_inlier_ratio_threshold_ = 0.3;
        }
        registration_reset_min_points_ =
            std::max(registration_reset_min_points_, 1);
        registration_reset_min_age_frames_ =
            std::max(registration_reset_min_age_frames_, 0);
        registration_reset_freeze_frames_ =
            std::max(registration_reset_freeze_frames_, 0);
        if (max_translation_jump_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "max_translation_jump <= 0，回退到1.0");
            max_translation_jump_ = 1.0;
        }
        if (max_rotation_jump_deg_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "max_rotation_jump_deg <= 0，回退到20.0");
            max_rotation_jump_deg_ = 20.0;
        }
        if (gicp_max_correspondence_distance_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "gicp_max_correspondence_distance <= 0，回退到1.5");
            gicp_max_correspondence_distance_ = 1.5;
        }
        gicp_correspondence_randomness_ =
            std::max(gicp_correspondence_randomness_, 1);
        gicp_maximum_optimizer_iterations_ =
            std::max(gicp_maximum_optimizer_iterations_, 1);
        if (ndt_resolution_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.resolution <= 0，回退到1.0");
            ndt_resolution_ = 1.0;
        }
        if (ndt_step_size_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.step_size <= 0，回退到0.1");
            ndt_step_size_ = 0.1;
        }
        if (ndt_fine_resolution_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.fine.resolution <= 0，回退到0.2");
            ndt_fine_resolution_ = 0.2;
        }
        if (ndt_fine_step_size_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.fine.step_size <= 0，回退到0.05");
            ndt_fine_step_size_ = 0.05;
        }
        if (ndt_outlier_ratio_ <= 0.0 || ndt_outlier_ratio_ >= 1.0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.outlier_ratio不在(0, 1)内，回退到0.35");
            ndt_outlier_ratio_ = 0.35;
        }
        if (ndt_num_threads_ <= 0) {
            RCLCPP_WARN(this->get_logger(),
                        "ndt.num_threads必须为正整数，回退到4");
            ndt_num_threads_ = 4;
        }
        if (backend_ != "gicp" &&
            ndt_neighborhood_search_ != "kdtree" &&
            ndt_neighborhood_search_ != "direct26" &&
            ndt_neighborhood_search_ != "direct7" &&
            ndt_neighborhood_search_ != "direct1") {
            const std::string message =
                "ndt.neighborhood_search仅支持kdtree/direct26/direct7/direct1，"
                "当前值为'" + ndt_neighborhood_search_ + "'";
            RCLCPP_FATAL(this->get_logger(), "%s", message.c_str());
            throw std::invalid_argument(message);
        }

        auto initial_vec =
            this->get_parameter("initial_extrinsics").as_double_array();
        auto initial_translation =
            this->get_parameter("initial_translation").as_double_array();
        auto initial_rpy =
            this->get_parameter("initial_rpy").as_double_array();
        bool initial_extrinsics_are_parent_to_child =
            this->get_parameter("initial_extrinsics_are_parent_to_child")
                .as_bool();
        if (initial_vec.size() == 16) {
            initial_guess_ = Eigen::Matrix4f::Identity();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    initial_guess_(i, j) =
                        static_cast<float>(initial_vec[i * 4 + j]);
                }
            }
        } else if (initial_translation.size() == 3 && initial_rpy.size() == 3) {
            const float deg_to_rad =
                static_cast<float>(M_PI / 180.0);
            Eigen::AngleAxisf roll(
                static_cast<float>(initial_rpy[0]) * deg_to_rad,
                Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf pitch(
                static_cast<float>(initial_rpy[1]) * deg_to_rad,
                Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf yaw(
                static_cast<float>(initial_rpy[2]) * deg_to_rad,
                Eigen::Vector3f::UnitZ());
            Eigen::Matrix3f rotation = (yaw * pitch * roll).toRotationMatrix();
            initial_guess_ = Eigen::Matrix4f::Identity();
            initial_guess_.block<3, 3>(0, 0) = rotation;
            initial_guess_(0, 3) = static_cast<float>(initial_translation[0]);
            initial_guess_(1, 3) = static_cast<float>(initial_translation[1]);
            initial_guess_(2, 3) = static_cast<float>(initial_translation[2]);
        } else {
            initial_guess_ = Eigen::Matrix4f::Identity();
        }
        if (!initial_extrinsics_are_parent_to_child) {
            initial_guess_ = initial_guess_.inverse();
        }
        current_transform_ = initial_guess_;
        has_valid_transform_ = true;
        if (!enable_registration_) {
            has_aligned_ = true;
            has_registered_once_ = true;
            RCLCPP_INFO(
                this->get_logger(),
                "点云配准已关闭，直接使用参数外参发布 rm_frame -> livox_frame TF");
        }
        RCLCPP_INFO(
            this->get_logger(),
            "Localization params: backend=%s enable_registration=%d "
            "continuous=%d accept_unconverged=%d interval=%.3fs "
            "max_registration=%d "
            "fitness_threshold=%.3f "
            "stable_required_count=%d max_translation_jump=%.3f "
            "max_rotation_jump_deg=%.3f smooth_alpha=%.2f "
            "source_leaf=%.3f target_leaf=%.3f background_filter=%d",
            backend_.c_str(), enable_registration_ ? 1 : 0,
            continuous_registration_ ? 1 : 0,
            accept_unconverged_result_ ? 1 : 0,
            registration_interval_sec_, max_registration_,
            fitness_threshold, stable_required_count_,
            max_translation_jump_, max_rotation_jump_deg_,
            transform_smooth_alpha_, source_voxel_leaf_size_,
            target_voxel_leaf_size_, enable_background_filter_ ? 1 : 0);
        if (backend_ == "gicp" || backend_ == "gicp_ndt") {
            RCLCPP_INFO(
                this->get_logger(),
                "GICP参数: max_iteration=%d max_correspondence=%.3f "
                "fitness_epsilon=%.3g "
                "rotation_epsilon=%.3g neighbors=%d optimizer_iterations=%d",
                gicp_max_iteration_, gicp_max_correspondence_distance_,
                gicp_fitness_epsilon_,
                gicp_rotation_epsilon_, gicp_correspondence_randomness_,
                gicp_maximum_optimizer_iterations_);
        }
        if (backend_ == "ndt" || backend_ == "gicp_ndt") {
            RCLCPP_INFO(
                this->get_logger(),
                "NDT-OMP粗配准参数: max_iteration=%d resolution=%.3f "
                "step_size=%.3f "
                "细配准参数: max_iteration=%d resolution=%.3f step_size=%.3f "
                "outlier_ratio=%.3f threads=%d neighborhood=%s",
                ndt_max_iteration_, ndt_resolution_, ndt_step_size_,
                ndt_fine_max_iteration_, ndt_fine_resolution_,
                ndt_fine_step_size_,
                ndt_outlier_ratio_,
                ndt_num_threads_, ndt_neighborhood_search_.c_str());
        }
        if (enable_background_filter_) {
            RCLCPP_INFO(
                this->get_logger(),
                "持久背景过滤: angle=%.3f deg range_tol=%.3f m "
                "candidate=%d次/%.3f m registration_reset_auto=%d "
                "reset=%.3f m/%.2f "
                "min_points=%d min_age=%d freeze=%d",
                background_angular_bin_deg_, background_range_tolerance_,
                background_candidate_confirm_count_,
                background_candidate_tolerance_,
                registration_reset_auto_ ? 1 : 0,
                registration_reset_inlier_distance_,
                registration_reset_inlier_ratio_threshold_,
                registration_reset_min_points_,
                registration_reset_min_age_frames_,
                registration_reset_freeze_frames_);
        }

        // 从pcd读取场地点云
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(target_pcd_file, *target_cloud_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load %s", target_pcd_file.c_str());
            return;
        }

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10, std::bind(&Localization::callback, this, std::placeholders::_1));
        registration_reset_callback_group_ =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        registration_reset_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "~/reset_registration",
                std::bind(
                    &Localization::handleRegistrationReset, this,
                    std::placeholders::_1, std::placeholders::_2),
                rmw_qos_profile_services_default,
                registration_reset_callback_group_);

        // 发布场地点云到 /livox/map 话题
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/map", 10);
        filter_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filter_map", 10);
        registration_source_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/livox/registration_source", 10);
        
        //为map_pub设置计时器
        timer_ = this->create_wall_timer(std::chrono::seconds(10), [this]() {
            sensor_msgs::msg::PointCloud2 target_msg;
            pcl::toROSMsg(*target_cloud_, target_msg);
            target_msg.header.frame_id = "rm_frame";
            publisher_->publish(target_msg);
        });

        // 初始化TF广播
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        // 发布初始猜测，提高早期鲁棒性
        publishTF(initial_guess_);
        logTransform("initial_guess (rm->livox)", initial_guess_);
        logTransform("publish_tf (rm->livox)", initial_guess_);
        if (enable_registration_) {
            prepareRegistrationTarget();
            registration_worker_ =
                std::thread(&Localization::registrationWorkerLoop, this);
        }
    }

    ~Localization() override {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stop_registration_worker_ = true;
        }
        registration_condition_.notify_all();
        if (registration_worker_.joinable()) {
            registration_worker_.join();
        }
    }

private:
    using NdtRegistration =
        pclomp::NormalDistributionsTransform<
            pcl::PointXYZ, pcl::PointXYZ>;

    struct RegistrationJob {
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        std::uint64_t generation = 0;
        int registration_index = 0;
    };

    static bool isFinitePoint(const pcl::PointXYZ& point) {
        return std::isfinite(point.x) && std::isfinite(point.y) &&
               std::isfinite(point.z);
    }

    static bool isInRegistrationRoi(const pcl::PointXYZ& point) {
        return point.x > 5.0F && point.x < 30.0F &&
               point.y > -10.0F && point.y < 8.0F && point.z < 7.0F;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleRegistrationCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const {
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        if (!cloud || cloud->empty()) {
            return downsampled_cloud;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        const float leaf_size =
            static_cast<float>(source_voxel_leaf_size_);
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_grid.setInputCloud(cloud);
        voxel_grid.filter(*downsampled_cloud);
        return downsampled_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filterRegistrationCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud) const {
        pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        roi_cloud->reserve(raw_cloud->size());
        for (const auto& point : *raw_cloud) {
            if (isFinitePoint(point) && isInRegistrationRoi(point)) {
                roi_cloud->push_back(point);
            }
        }
        if (roi_cloud->empty()) {
            return roi_cloud;
        }

        return downsampleRegistrationCloud(roi_cloud);
    }

    void insertCloudIntoGrid(
        GridMap& grid_map,
        const pcl::PointCloud<pcl::PointXYZ>& cloud,
        double angular_bin_deg) const {
        for (const auto& point : cloud) {
            if (!isFinitePoint(point)) {
                continue;
            }
            const double distance = std::sqrt(
                point.x * point.x + point.y * point.y + point.z * point.z);
            if (distance <= 0.0) {
                continue;
            }

            const double azimuth = std::atan2(point.y, point.x);
            const double elevation = std::atan2(
                point.z, std::sqrt(point.x * point.x + point.y * point.y));
            const int azimuth_index = static_cast<int>(
                std::floor(azimuth * 180.0 / M_PI / angular_bin_deg));
            const int elevation_index = static_cast<int>(
                std::floor(elevation * 180.0 / M_PI / angular_bin_deg));

            auto& grid = grid_map[GridKey{azimuth_index, elevation_index}];
            if (distance > grid.maxDistance) {
                grid.farthestPoint = point;
                grid.maxDistance = distance;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr gridToCloud(
        const GridMap& grid_map, bool apply_roi) const {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        cloud->reserve(grid_map.size());
        for (const auto& item : grid_map) {
            if (item.second.maxDistance <= 0.0) {
                continue;
            }
            if (apply_roi && !isInRegistrationRoi(item.second.farthestPoint)) {
                continue;
            }
            cloud->push_back(item.second.farthestPoint);
        }
        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr gridToRegistrationSource(
        const GridMap& grid_map) const {
        // 角度栅格会跨帧累积；再次使用与单帧筛选相同的体素尺寸，
        // 保证最终发布并送入配准器的old不会重新变得过密。
        return downsampleRegistrationCloud(gridToCloud(grid_map, false));
    }

    bool calculateBackgroundInlierRatio(
        const GridMap& current_grid, double& inlier_ratio,
        std::size_t& current_point_count,
        std::size_t& background_point_count) const {
        const auto current_cloud = gridToCloud(current_grid, false);
        const auto background_cloud = gridToCloud(background_grid_, false);
        current_point_count = current_cloud->size();
        background_point_count = background_cloud->size();
        if (current_point_count <
                static_cast<std::size_t>(registration_reset_min_points_) ||
            background_point_count <
                static_cast<std::size_t>(registration_reset_min_points_)) {
            return false;
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> background_tree;
        background_tree.setInputCloud(background_cloud);
        std::vector<int> nearest_index(1);
        std::vector<float> nearest_squared_distance(1);
        const float inlier_distance_squared = static_cast<float>(
            registration_reset_inlier_distance_ *
            registration_reset_inlier_distance_);
        std::size_t inlier_count = 0;
        for (const auto& point : *current_cloud) {
            if (background_tree.nearestKSearch(
                    point, 1, nearest_index, nearest_squared_distance) > 0 &&
                nearest_squared_distance[0] <= inlier_distance_squared) {
                ++inlier_count;
            }
        }
        inlier_ratio = static_cast<double>(inlier_count) /
                       static_cast<double>(current_point_count);
        return true;
    }

    void updateBackgroundGrid(const GridMap& current_grid) {
        for (const auto& item : current_grid) {
            const auto& key = item.first;
            const auto& current = item.second;
            auto background_it = background_grid_.find(key);
            if (background_it == background_grid_.end()) {
                background_grid_.emplace(key, current);
                farther_candidates_.erase(key);
                continue;
            }

            auto& background = background_it->second;
            if (current.maxDistance <=
                background.maxDistance + background_range_tolerance_) {
                // 当前点更近时视为前景遮挡；同一表面的小幅波动仍保留其中更远点。
                if (current.maxDistance > background.maxDistance) {
                    background = current;
                }
                farther_candidates_.erase(key);
                continue;
            }

            auto& candidate = farther_candidates_[key];
            if (candidate.confirmations > 0 &&
                std::abs(
                    candidate.grid.maxDistance - current.maxDistance) <=
                    background_candidate_tolerance_) {
                ++candidate.confirmations;
                if (current.maxDistance > candidate.grid.maxDistance) {
                    candidate.grid = current;
                }
            } else {
                candidate.grid = current;
                candidate.confirmations = 1;
            }

            if (candidate.confirmations >=
                background_candidate_confirm_count_) {
                background = candidate.grid;
                farther_candidates_.erase(key);
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr buildPersistentBackgroundSource(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud,
        bool& registration_reset) {
        GridMap current_grid;
        insertCloudIntoGrid(
            current_grid, *filtered_cloud, background_angular_bin_deg_);
        registration_reset = false;
        if (current_grid.empty()) {
            return gridToRegistrationSource(background_grid_);
        }

        if (background_grid_.empty()) {
            background_grid_ = current_grid;
            farther_candidates_.clear();
            background_age_frames_ = 0;
            RCLCPP_INFO(
                this->get_logger(),
                "持久背景old初始化: %zu个角度栅格",
                background_grid_.size());
            return gridToRegistrationSource(background_grid_);
        }

        if (registration_reset_freeze_frames_remaining_ > 0) {
            --registration_reset_freeze_frames_remaining_;
            ++background_age_frames_;
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "背景重置冻结中: 剩余%d帧，old不更新",
                registration_reset_freeze_frames_remaining_);
            return gridToRegistrationSource(background_grid_);
        }

        double inlier_ratio = 1.0;
        std::size_t current_point_count = 0;
        std::size_t background_point_count = 0;
        const bool ratio_valid =
            registration_reset_auto_ &&
            background_age_frames_ >= registration_reset_min_age_frames_ &&
            calculateBackgroundInlierRatio(
                current_grid, inlier_ratio, current_point_count,
                background_point_count);
        if (ratio_valid) {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "背景一致性: inlier_ratio=%.3f current=%zu old=%zu age=%d",
                inlier_ratio, current_point_count, background_point_count,
                background_age_frames_);
        }

        if (ratio_valid &&
            inlier_ratio < registration_reset_inlier_ratio_threshold_) {
            background_grid_ = current_grid;
            farther_candidates_.clear();
            background_age_frames_ = 0;
            registration_reset_freeze_frames_remaining_ =
                registration_reset_freeze_frames_;
            registration_reset = true;
            RCLCPP_WARN(
                this->get_logger(),
                "背景内点率%.3f低于阈值%.3f，判定雷达发生移动；"
                "old已用当前帧重置，并冻结%d帧",
                inlier_ratio, registration_reset_inlier_ratio_threshold_,
                registration_reset_freeze_frames_remaining_);
            return gridToRegistrationSource(background_grid_);
        }

        // old的维护独立于registration状态；配准锁定、达到次数上限或
        // NDT阶段达到迭代上限，都不应阻止背景继续吸收后续帧。
        updateBackgroundGrid(current_grid);
        ++background_age_frames_;
        return gridToRegistrationSource(background_grid_);
    }

    bool registrationFinishedLocked() const {
        return registration_count_ >= max_registration_ ||
               (has_aligned_ && !continuous_registration_);
    }

    void restartRegistrationLocked() {
        has_aligned_ = false;
        has_registered_once_ = false;
        stable_count_ = 0;
        registration_count_ = 0;
        has_last_registration_time_ = false;
        registration_job_pending_ = false;
        ++registration_generation_;
    }

    bool enqueueRegistrationLocked(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        bool ignore_interval) {
        if (!enable_registration_ || !source_cloud ||
            source_cloud->empty() || registrationFinishedLocked()) {
            return false;
        }

        const bool mailbox_was_empty = !registration_job_pending_;
        if (mailbox_was_empty) {
            const auto now = std::chrono::steady_clock::now();
            if (!ignore_interval && registration_interval_sec_ > 0.0 &&
                has_last_registration_time_) {
                const double elapsed_sec =
                    std::chrono::duration<double>(
                        now - last_registration_time_).count();
                if (elapsed_sec < registration_interval_sec_) {
                    return false;
                }
            }

            last_registration_time_ = now;
            has_last_registration_time_ = true;
        }

        // latest-only mailbox：worker忙时不积压队列，新帧直接覆盖尚未执行的任务。
        // registration次数和初值在worker真正取走任务时确定，覆盖不会消耗次数。
        registration_job_.source_cloud = source_cloud;
        registration_job_.generation = registration_generation_;
        registration_job_pending_ = true;
        if (mailbox_was_empty) {
            registration_condition_.notify_one();
        }
        return true;
    }

    void publishRegistrationSource(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const std_msgs::msg::Header& header) {
        if (!source_cloud || source_cloud->empty()) {
            return;
        }
        sensor_msgs::msg::PointCloud2 source_msg;
        pcl::toROSMsg(*source_cloud, source_msg);
        source_msg.header = header;
        source_msg.header.frame_id =
            header.frame_id.empty() ? "livox_frame" : header.frame_id;
        filter_publisher_->publish(source_msg);
        registration_source_publisher_->publish(source_msg);
    }

    void handleRegistrationReset(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        pcl::PointCloud<pcl::PointXYZ>::Ptr registration_source;
        std_msgs::msg::Header source_header;
        std::size_t grid_size = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!latest_filtered_cloud_ || latest_filtered_cloud_->empty()) {
                response->success = false;
                response->message = "尚未收到筛选后的有效雷达点云";
                RCLCPP_WARN(
                    this->get_logger(), "手动重置配准失败: %s",
                    response->message.c_str());
                return;
            }

            GridMap latest_grid;
            insertCloudIntoGrid(
                latest_grid, *latest_filtered_cloud_,
                enable_background_filter_ ? background_angular_bin_deg_
                                          : legacy_grid_size_degrees_);
            if (latest_grid.empty()) {
                response->success = false;
                response->message = "最近一帧筛选点云不包含有效点";
                RCLCPP_WARN(
                    this->get_logger(), "手动重置配准失败: %s",
                    response->message.c_str());
                return;
            }

            if (enable_background_filter_) {
                background_grid_ = std::move(latest_grid);
                farther_candidates_.clear();
                grid_size = background_grid_.size();
                registration_source =
                    gridToRegistrationSource(background_grid_);
            } else {
                accumulated_clouds_.clear();
                accumulated_clouds_.push_back(latest_filtered_cloud_);
                legacy_grid_map_ = std::move(latest_grid);
                grid_size = legacy_grid_map_.size();
                registration_source =
                    gridToRegistrationSource(legacy_grid_map_);
            }
            background_age_frames_ = 0;
            registration_reset_freeze_frames_remaining_ =
                registration_reset_freeze_frames_;
            restartRegistrationLocked();
            enqueueRegistrationLocked(registration_source, true);
            source_header = latest_filtered_header_;
        }

        publishRegistrationSource(registration_source, source_header);
        response->success = true;
        response->message = enable_registration_
                                ? "配准已重置并开始新一轮registration"
                                : "old已重置；当前点云配准已关闭";
        RCLCPP_WARN(
            this->get_logger(),
            "收到手动registration reset: old=%zu个角度栅格，冻结%d帧，"
            "registration_reset_auto=%d",
            grid_size, registration_reset_freeze_frames_,
            registration_reset_auto_ ? 1 : 0);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr buildLegacyAccumulatedSource(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_cloud) {
        if (accumulated_clouds_.size() <
            static_cast<std::size_t>(accumulate_time)) {
            accumulated_clouds_.push_back(current_cloud);
            return nullptr;
        }
        accumulated_clouds_.erase(accumulated_clouds_.begin());
        accumulated_clouds_.push_back(current_cloud);

        // 关闭持久背景过滤时，完整保留原有滑动累积取最远点逻辑。
        legacy_grid_map_.clear();
        for (const auto& accumulated_cloud : accumulated_clouds_) {
            insertCloudIntoGrid(
                legacy_grid_map_, *accumulated_cloud,
                legacy_grid_size_degrees_);
        }
        return gridToRegistrationSource(legacy_grid_map_);
    }

    void prepareRegistrationTarget() {
        if (target_downsampled_) {
            return;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_target(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> target_voxelgrid;
        const float target_leaf =
            static_cast<float>(target_voxel_leaf_size_);
        target_voxelgrid.setLeafSize(
            target_leaf, target_leaf, target_leaf);
        target_voxelgrid.setInputCloud(target_cloud_);
        target_voxelgrid.filter(*downsampled_target);
        if (!downsampled_target->empty()) {
            *target_cloud_ = *downsampled_target;
        }
        target_downsampled_ = true;
        initializeRegistration();
    }

    void initializeRegistration() {
        const bool need_gicp =
            backend_ == "gicp" || backend_ == "gicp_ndt";
        const bool need_ndt =
            backend_ == "ndt" || backend_ == "gicp_ndt";
        if ((!need_gicp || gicp_) &&
            (!need_ndt || (ndt_coarse_ && ndt_fine_))) {
            return;
        }

        if (need_gicp) {
            gicp_.reset(new pcl::GeneralizedIterativeClosestPoint<
                        pcl::PointXYZ, pcl::PointXYZ>());
            gicp_->setMaxCorrespondenceDistance(
                gicp_max_correspondence_distance_);
            gicp_->setMaximumIterations(gicp_max_iteration_);
            gicp_->setTransformationEpsilon(transformation_epsilon_);
            gicp_->setEuclideanFitnessEpsilon(gicp_fitness_epsilon_);
            gicp_->setRotationEpsilon(gicp_rotation_epsilon_);
            gicp_->setCorrespondenceRandomness(
                gicp_correspondence_randomness_);
            gicp_->setMaximumOptimizerIterations(
                gicp_maximum_optimizer_iterations_);
            gicp_->setInputTarget(target_cloud_);
        }
        if (need_ndt) {
            ndt_coarse_ = createNdtRegistration(
                ndt_resolution_, ndt_step_size_, ndt_max_iteration_);
            ndt_fine_ = createNdtRegistration(
                ndt_fine_resolution_, ndt_fine_step_size_,
                ndt_fine_max_iteration_);
        }

        RCLCPP_INFO(
            this->get_logger(),
            "%s配准器初始化完成: target=%zu gicp_max_iteration=%d "
            "ndt_iterations=coarse:%d+fine:%d transformation_epsilon=%.3g",
            backend_.c_str(), target_cloud_->size(), gicp_max_iteration_,
            ndt_max_iteration_, ndt_fine_max_iteration_,
            transformation_epsilon_);
    }

    NdtRegistration::Ptr createNdtRegistration(
        double resolution, double step_size, int max_iteration) {
        NdtRegistration::Ptr ndt(new NdtRegistration());
        ndt->setMaximumIterations(max_iteration);
        ndt->setTransformationEpsilon(transformation_epsilon_);
        ndt->setResolution(static_cast<float>(resolution));
        ndt->setStepSize(step_size);
        ndt->setOulierRatio(ndt_outlier_ratio_);
        ndt->setNumThreads(ndt_num_threads_);
        if (ndt_neighborhood_search_ == "kdtree") {
            ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
        } else if (ndt_neighborhood_search_ == "direct26") {
            ndt->setNeighborhoodSearchMethod(pclomp::DIRECT26);
        } else if (ndt_neighborhood_search_ == "direct7") {
            ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        } else {
            ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
        }
        // 粗、细NDT各自构建一次目标高斯体素，后续registration持续复用。
        ndt->setInputTarget(target_cloud_);
        return ndt;
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);
        const auto filtered_cloud = filterRegistrationCloud(raw_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr registration_source;
        Eigen::Matrix4f transform_to_publish = Eigen::Matrix4f::Identity();
        bool publish_transform = false;
        bool publish_source = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_filtered_cloud_ = filtered_cloud;
            latest_filtered_header_ = msg->header;
            const bool registration_active =
                enable_registration_ && !registrationFinishedLocked();
            bool registration_reset = false;

            if (enable_background_filter_) {
                registration_source = buildPersistentBackgroundSource(
                    filtered_cloud, registration_reset);
                publish_source =
                    registration_source && !registration_source->empty();
            } else {
                registration_source =
                    buildLegacyAccumulatedSource(filtered_cloud);
                publish_source =
                    registration_source && !registration_source->empty();
            }

            if (registration_reset) {
                // 自动reset同样开始全新的配准周期，旧工作结果通过generation失效。
                restartRegistrationLocked();
                enqueueRegistrationLocked(registration_source, true);
            } else if (registration_active && registration_source) {
                enqueueRegistrationLocked(registration_source, false);
            }

            if (has_valid_transform_) {
                transform_to_publish = current_transform_;
                publish_transform = true;
            }
        }

        if (publish_source) {
            publishRegistrationSource(registration_source, msg->header);
        }
        if (filtered_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "筛选后的localization输入点云为空，保持上一帧变换");
        }
        if (publish_transform) {
            publishTF(transform_to_publish);
        }
    }

    void registrationWorkerLoop() {
        while (true) {
            RegistrationJob job;
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                registration_condition_.wait(lock, [this]() {
                    return stop_registration_worker_ ||
                           registration_job_pending_;
                });
                if (stop_registration_worker_) {
                    return;
                }
                job = registration_job_;
                registration_job_pending_ = false;
                job.init_guess =
                    has_valid_transform_ ? current_transform_ : initial_guess_;
                job.registration_index = ++registration_count_;
                registration_in_progress_ = true;
            }

            Eigen::Matrix4f aligned_transform = job.init_guess;
            double fitness_score = std::numeric_limits<double>::infinity();
            const bool converged = align(
                job.source_cloud, job.init_guess, aligned_transform,
                fitness_score, job.registration_index);
            const bool transform_is_finite = aligned_transform.allFinite();
            const bool result_usable =
                transform_is_finite &&
                (converged || accept_unconverged_result_);

            Eigen::Matrix4f transform_to_publish = Eigen::Matrix4f::Identity();
            bool publish_transform = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                registration_in_progress_ = false;
                if (job.generation != registration_generation_) {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "丢弃reset前的registration结果: generation=%zu current=%zu",
                        static_cast<std::size_t>(job.generation),
                        static_cast<std::size_t>(registration_generation_));
                    continue;
                }

                if (!result_usable) {
                    stable_count_ = 0;
                    RCLCPP_WARN(
                        this->get_logger(),
                        "%s结果未采用: converged=%d finite=%d",
                        backend_.c_str(), converged ? 1 : 0,
                        transform_is_finite ? 1 : 0);
                } else {
                    if (!converged) {
                        RCLCPP_WARN(
                            this->get_logger(),
                            "%s未收敛，但实验参数允许采用本次有限位姿",
                            backend_.c_str());
                    }
                    bool jump_ok = true;
                    if (has_registered_once_) {
                        const double translation_jump = translationDistance(
                            current_transform_, aligned_transform);
                        const double rotation_jump = rotationDistanceDeg(
                            current_transform_, aligned_transform);
                        jump_ok =
                            translation_jump <= max_translation_jump_ &&
                            rotation_jump <= max_rotation_jump_deg_;
                        if (!jump_ok) {
                            RCLCPP_WARN(
                                this->get_logger(),
                                "拒绝异常跳变: dt=%.3f m dr=%.2f deg",
                                translation_jump, rotation_jump);
                        }
                    }

                    if (jump_ok) {
                        current_transform_ =
                            has_registered_once_
                                ? smoothTransform(
                                      current_transform_, aligned_transform,
                                      transform_smooth_alpha_)
                                : aligned_transform;
                        has_valid_transform_ = true;
                        has_registered_once_ = true;
                        if (fitness_score < fitness_threshold) {
                            ++stable_count_;
                            if (!has_aligned_ &&
                                stable_count_ >= stable_required_count_) {
                                has_aligned_ = true;
                                if (!continuous_registration_) {
                                    registration_job_pending_ = false;
                                }
                                RCLCPP_INFO(
                                    this->get_logger(),
                                    "Localization锁定: fitness=%.4f 连续%d帧达标；"
                                    "停止继续配准，old保持逐帧更新",
                                    fitness_score, stable_count_);
                            }
                        } else {
                            stable_count_ = 0;
                        }
                    } else {
                        stable_count_ = 0;
                    }
                }

                if (registration_count_ >= max_registration_) {
                    RCLCPP_INFO(
                        this->get_logger(),
                        "已达到本轮最大registration次数%d，停止继续配准；"
                        "old保持逐帧更新；"
                        "当前fitness=%.4f stable_count=%d/%d",
                        max_registration_, fitness_score, stable_count_,
                        stable_required_count_);
                }
                if (has_valid_transform_) {
                    transform_to_publish = current_transform_;
                    publish_transform = true;
                }
            }
            if (publish_transform) {
                publishTF(transform_to_publish);
            }
        }
    }

    bool runRegistration(
        const pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr& registration,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const Eigen::Matrix4f&                      init_guess,
        bool                                         use_guess,
        Eigen::Matrix4f&                            final_transform,
        double&                                      fitness_score,
        double&                                      elapsed_ms) {
        registration->setInputSource(source_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
            new pcl::PointCloud<pcl::PointXYZ>());
        auto t1 = std::chrono::steady_clock::now();
        if (use_guess) {
            registration->align(*aligned, init_guess);
        } else {
            registration->align(*aligned);
        }
        auto t2 = std::chrono::steady_clock::now();
        final_transform = registration->getFinalTransformation();
        fitness_score = registration->getFitnessScore();
        elapsed_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0;
        return registration->hasConverged();
    }

    bool runNdtTwoStage(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const Eigen::Matrix4f&                      coarse_init_guess,
        bool                                         use_coarse_guess,
        Eigen::Matrix4f&                            final_transform,
        double&                                      fitness_score,
        int                                         registration_index,
        const char*                                  pipeline_name,
        double                                       previous_elapsed_ms) {
        Eigen::Matrix4f coarse_transform = coarse_init_guess;
        double coarse_fitness = std::numeric_limits<double>::infinity();
        double coarse_elapsed_ms = 0.0;
        const bool coarse_converged = runRegistration(
            ndt_coarse_, source_cloud, coarse_init_guess, use_coarse_guess,
            coarse_transform, coarse_fitness, coarse_elapsed_ms);
        RCLCPP_INFO(
            this->get_logger(),
            "%s/coarse NDT [registration %d/%d]: converged=%d "
            "fitness=%.4f probability=%.4f iterations=%d time=%.2f ms",
            pipeline_name, registration_index, max_registration_,
            coarse_converged ? 1 : 0, coarse_fitness,
            ndt_coarse_->getTransformationProbability(),
            ndt_coarse_->getFinalNumIteration(), coarse_elapsed_ms);

        // 中间阶段即使未收敛，只要结果有限就继续用作细配准初值；
        // 无效结果则回退至粗配准收到的上一阶段初值。
        const Eigen::Matrix4f fine_init_guess =
            coarse_transform.allFinite()
                ? coarse_transform
                : coarse_init_guess;
        double fine_elapsed_ms = 0.0;
        const bool fine_converged = runRegistration(
            ndt_fine_, source_cloud, fine_init_guess, true,
            final_transform, fitness_score, fine_elapsed_ms);
        RCLCPP_INFO(
            this->get_logger(),
            "%s/fine NDT [registration %d/%d]: converged=%d "
            "fitness=%.4f probability=%.4f iterations=%d time=%.2f ms "
            "ndt_total=%.2f ms pipeline_total=%.2f ms",
            pipeline_name, registration_index, max_registration_,
            fine_converged ? 1 : 0, fitness_score,
            ndt_fine_->getTransformationProbability(),
            ndt_fine_->getFinalNumIteration(), fine_elapsed_ms,
            coarse_elapsed_ms + fine_elapsed_ms,
            previous_elapsed_ms + coarse_elapsed_ms + fine_elapsed_ms);
        return fine_converged;
    }

    bool align(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const Eigen::Matrix4f&                      init_guess,
        Eigen::Matrix4f&                            final_transform,
        double&                                     fitness_score,
        int                                         registration_index) {
        if (backend_ == "gicp") {
            double elapsed_ms = 0.0;
            const bool converged = runRegistration(
                gicp_, source_cloud, init_guess, use_initial_guess,
                final_transform, fitness_score, elapsed_ms);
            RCLCPP_INFO(
                this->get_logger(),
                "GICP [registration %d/%d]: converged=%d fitness=%.4f "
                "time=%.2f ms",
                registration_index, max_registration_,
                converged ? 1 : 0, fitness_score, elapsed_ms);
            if (final_transform.allFinite()) {
                logTransform("gicp_result (rm->livox)", final_transform);
            }
            return converged;
        }

        if (backend_ == "ndt") {
            const bool converged = runNdtTwoStage(
                source_cloud, init_guess, use_initial_guess,
                final_transform, fitness_score, registration_index,
                "NDT", 0.0);
            if (final_transform.allFinite()) {
                logTransform(
                    "ndt_fine_result (rm->livox)", final_transform);
            }
            return converged;
        }

        Eigen::Matrix4f gicp_transform = init_guess;
        double gicp_fitness = std::numeric_limits<double>::infinity();
        double gicp_elapsed_ms = 0.0;
        const bool gicp_converged = runRegistration(
            gicp_, source_cloud, init_guess, use_initial_guess,
            gicp_transform, gicp_fitness, gicp_elapsed_ms);
        RCLCPP_INFO(
            this->get_logger(),
            "GICP_NDT/GICP [registration %d/%d]: converged=%d "
            "fitness=%.4f time=%.2f ms",
            registration_index, max_registration_,
            gicp_converged ? 1 : 0, gicp_fitness, gicp_elapsed_ms);

        const Eigen::Matrix4f ndt_init =
            gicp_transform.allFinite() ? gicp_transform : init_guess;
        const bool ndt_converged = runNdtTwoStage(
            source_cloud, ndt_init, true,
            final_transform, fitness_score, registration_index,
            "GICP_NDT", gicp_elapsed_ms);
        if (final_transform.allFinite()) {
            logTransform(
                "gicp_ndt_fine_result (rm->livox)", final_transform);
        }
        return ndt_converged;
    }

    double translationDistance(
        const Eigen::Matrix4f& lhs, const Eigen::Matrix4f& rhs) const {
        return (lhs.block<3, 1>(0, 3) - rhs.block<3, 1>(0, 3)).norm();
    }

    double rotationDistanceDeg(
        const Eigen::Matrix4f& lhs, const Eigen::Matrix4f& rhs) const {
        Eigen::Matrix3f delta = lhs.block<3, 3>(0, 0).transpose() *
                                rhs.block<3, 3>(0, 0);
        Eigen::AngleAxisf angle_axis(delta);
        return std::abs(static_cast<double>(angle_axis.angle()) * 180.0 / M_PI);
    }

    Eigen::Matrix4f smoothTransform(
        const Eigen::Matrix4f& previous,
        const Eigen::Matrix4f& current,
        double                 alpha) const {
        if (alpha <= 0.0) {
            return previous;
        }
        if (alpha >= 1.0) {
            return current;
        }

        Eigen::Matrix4f output = Eigen::Matrix4f::Identity();
        output.block<3, 1>(0, 3) =
            previous.block<3, 1>(0, 3) * static_cast<float>(1.0 - alpha) +
            current.block<3, 1>(0, 3) * static_cast<float>(alpha);

        Eigen::Quaternionf q_prev(previous.block<3, 3>(0, 0));
        Eigen::Quaternionf q_curr(current.block<3, 3>(0, 0));
        q_prev.normalize();
        q_curr.normalize();
        Eigen::Quaternionf q_out =
            q_prev.slerp(static_cast<float>(alpha), q_curr);
        q_out.normalize();
        output.block<3, 3>(0, 0) = q_out.toRotationMatrix();
        return output;
    }

    void publishTF(const Eigen::Matrix4f& transform_rm_to_livox) {
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->now();
        transform_stamped.header.frame_id = "rm_frame";
        transform_stamped.child_frame_id = "livox_frame";
        transform_stamped.transform.translation.x = transform_rm_to_livox(0, 3);
        transform_stamped.transform.translation.y = transform_rm_to_livox(1, 3);
        transform_stamped.transform.translation.z = transform_rm_to_livox(2, 3);
        Eigen::Matrix3f rotation = transform_rm_to_livox.block<3, 3>(0, 0);
        Eigen::Quaternionf q(rotation);
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(transform_stamped);
        logTransformThrottled("publish_tf (rm->livox)", transform_rm_to_livox);
    }

    void logTransform(const char* tag, const Eigen::Matrix4f& transform) {
        Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
        tf2::Matrix3x3  rot(
            rotation(0, 0), rotation(0, 1), rotation(0, 2),
            rotation(1, 0), rotation(1, 1), rotation(1, 2),
            rotation(2, 0), rotation(2, 1), rotation(2, 2));
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        rot.getRPY(roll, pitch, yaw);
        RCLCPP_INFO(
            this->get_logger(),
            "%s t=[%.3f %.3f %.3f] rpy_deg=[%.2f %.2f %.2f]",
            tag, transform(0, 3), transform(1, 3), transform(2, 3),
            roll * 180.0 / M_PI, pitch * 180.0 / M_PI,
            yaw * 180.0 / M_PI);
    }

    void logTransformThrottled(const char* tag, const Eigen::Matrix4f& transform) {
        Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
        tf2::Matrix3x3  rot(
            rotation(0, 0), rotation(0, 1), rotation(0, 2),
            rotation(1, 0), rotation(1, 1), rotation(1, 2),
            rotation(2, 0), rotation(2, 1), rotation(2, 2));
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        rot.getRPY(roll, pitch, yaw);
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "%s t=[%.3f %.3f %.3f] rpy_deg=[%.2f %.2f %.2f]",
            tag, transform(0, 3), transform(1, 3), transform(2, 3),
            roll * 180.0 / M_PI, pitch * 180.0 / M_PI,
            yaw * 180.0 / M_PI);
    }
    
    bool has_aligned_ = false;
    bool has_valid_transform_ = false;
    bool has_registered_once_ = false;
    bool target_downsampled_ = false;
    int  stable_count_ = 0;
    Eigen::Matrix4f initial_guess_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f current_transform_ = Eigen::Matrix4f::Identity();
    std::string target_pcd_file_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_filtered_cloud_;
    std_msgs::msg::Header latest_filtered_header_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        registration_reset_service_;
    rclcpp::CallbackGroup::SharedPtr registration_reset_callback_group_;
    // 关闭持久背景过滤时使用的原有滑动点云缓存。
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    int accumulate_time = 20;
    bool use_initial_guess = true;
    std::string backend_ = "gicp";
    bool enable_registration_ = true;
    bool continuous_registration_ = false;
    bool accept_unconverged_result_ = false;
    double registration_interval_sec_ = 0.0;
    int max_registration_ = 100;
    int registration_count_ = 0;
    bool has_last_registration_time_ = false;
    std::chrono::steady_clock::time_point last_registration_time_;
    double fitness_threshold = 0.2;
    double transformation_epsilon_ = 1e-4;
    double source_voxel_leaf_size_ = 0.1;
    double target_voxel_leaf_size_ = 1.0;
    bool enable_background_filter_ = false;
    double background_angular_bin_deg_ = 0.1;
    double background_range_tolerance_ = 0.1;
    int background_candidate_confirm_count_ = 2;
    double background_candidate_tolerance_ = 0.15;
    bool registration_reset_auto_ = true;
    double registration_reset_inlier_distance_ = 0.25;
    double registration_reset_inlier_ratio_threshold_ = 0.3;
    int registration_reset_min_points_ = 200;
    int registration_reset_min_age_frames_ = 10;
    int registration_reset_freeze_frames_ = 5;
    int background_age_frames_ = 0;
    int registration_reset_freeze_frames_remaining_ = 0;
    int stable_required_count_ = 3;
    double max_translation_jump_ = 1.0;
    double max_rotation_jump_deg_ = 20.0;
    double transform_smooth_alpha_ = 0.35;
    double gicp_max_correspondence_distance_ = 1.5;
    int gicp_max_iteration_ = 80;
    double gicp_fitness_epsilon_ = 1e-3;
    double gicp_rotation_epsilon_ = 2e-3;
    int gicp_correspondence_randomness_ = 20;
    int gicp_maximum_optimizer_iterations_ = 20;
    double ndt_resolution_ = 1.0;
    int ndt_max_iteration_ = 60;
    double ndt_step_size_ = 0.1;
    double ndt_fine_resolution_ = 0.2;
    int ndt_fine_max_iteration_ = 20;
    double ndt_fine_step_size_ = 0.05;
    double ndt_outlier_ratio_ = 0.35;
    int ndt_num_threads_ = 4;
    std::string ndt_neighborhood_search_ = "direct7";
    pcl::GeneralizedIterativeClosestPoint<
        pcl::PointXYZ, pcl::PointXYZ>::Ptr gicp_;
    NdtRegistration::Ptr ndt_coarse_;
    NdtRegistration::Ptr ndt_fine_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        registration_source_publisher_;
    // 为map_pub设置计时器。
    rclcpp::TimerBase::SharedPtr timer_;
    const double legacy_grid_size_degrees_ = 0.1;
    GridMap legacy_grid_map_;
    GridMap background_grid_;
    CandidateMap farther_candidates_;
    std::mutex state_mutex_;
    std::condition_variable registration_condition_;
    std::thread registration_worker_;
    RegistrationJob registration_job_;
    bool registration_job_pending_ = false;
    bool registration_in_progress_ = false;
    bool stop_registration_worker_ = false;
    std::uint64_t registration_generation_ = 0;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
} // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)
