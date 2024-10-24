#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl/point_cloud.h>
#include <pcl/common/angles.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

// #include <pcl/gpu/features/features.hpp>

#include <Eigen/Core>

#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <omp.h>


rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

namespace tc_livox_utils {
class PcdFilter : public rclcpp::Node
{
public:
    PcdFilter(const rclcpp::NodeOptions &options);
    PcdFilter(
        const std::string &name_space,
        const rclcpp::NodeOptions &options = rclcpp::NodeOptions()
    );

private:
    void concatenate_pointclouds(
        std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> pointcloud_out,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl1,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl2,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl3
    );

    // voxel grid downsampling (normal)
    template <typename PointT, typename U>
    void voxel_grid_filter(
        const std::shared_ptr<pcl::PointCloud<PointT>> cloud,
        const U leaf_size,
        std::shared_ptr<pcl::PointCloud<PointT>> cloud_filtered
    ) {
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*cloud_filtered);
    }


    // indices による点群の除去
    template <typename PointT>
    void remove_indices(
        const std::shared_ptr<pcl::PointCloud<PointT>> cloud,
        const pcl::PointIndices::Ptr &inliers,
        std::shared_ptr<pcl::PointCloud<PointT>> cloud_removed
    ) {
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_removed);
    }
    

    // 自分が写り込んだ点群の除去
    // front_distance, right_mid_distance, side_distanceから除去
    template <typename PointT>
    void remove_self_pcd(
        const std::shared_ptr<pcl::PointCloud<PointT>> cloud,
        std::shared_ptr<pcl::PointCloud<PointT>> cloud_removed
    ) {
        // RCLCPP_INFO(this->get_logger(), "front_distance: %f, right_mid_distance: %f, side_distance: %f", front_distance, right_mid_distance, side_distance);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        for (size_t i = 0; i < cloud->points.size(); i++) {
            auto point = cloud->points[i];
            if (point.x < front_distance && point.x > -rear_distance && point.y < side_distance && point.y > -side_distance) {
                inliers->indices.push_back(i);
            }
        }
        remove_indices(cloud, inliers, cloud_removed);
        // RCLCPP_INFO(this->get_logger(), "Remove self pcd: %d -> %d", cloud->points.size(), cloud_removed->points.size());
    }

    // x, y, z, roll, pitch, yaw -> Eigen::Affine3f
    Eigen::Affine3f xyzrpy2tf(
        const float x,
        const float y,
        const float z,
        const float roll,
        const float pitch,
        const float yaw
    ) {
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        tf.translation() << x, y, z;
        tf.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
        tf.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
        tf.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
        return tf;
    }



    // ======================================================================
    // PointCloud2 subscriber callback
    void cb_lidar(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // sync callback
    void cb_lidar_sync(
        const sensor_msgs::msg::PointCloud2::SharedPtr front_left_lidar,
        const sensor_msgs::msg::PointCloud2::SharedPtr front_right_lidar,
        const sensor_msgs::msg::PointCloud2::SharedPtr rear_lidar
    );
    // ======================================================================
    std::string front_left_lidar_sub_topic_, front_right_lidar_sub_topic_, rear_lidar_sub_topic_;
    std::string front_left_lidar_pub_topic_, front_right_lidar_pub_topic_, rear_lidar_pub_topic_;
    std::string lidar_frame_id_;
    std::string pcd_pub_topic_head_;
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_front_left_lidar_, sub_front_right_lidar_, sub_rear_lidar_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> front_left_lidar_sub_, front_right_lidar_sub_, rear_lidar_sub_;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    std::shared_ptr<Sync> sync_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pub_lidar_vec_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_laser_;
    std::string laser_pub_topic_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // 地面除去に使用するlidar->base_linkのtf
    bool get_tf_;
    std::shared_ptr<Eigen::Affine3f> tf_lidar2base_;
    std::shared_ptr<Eigen::Affine3f> tf_base2map_;

    // livox->base_linkの変換行列群
    Eigen::Affine3f tf_front_left_lidar2base_;
    Eigen::Affine3f tf_front_right_lidar2base_;
    Eigen::Affine3f tf_rear_lidar2base_;
    
    std::shared_ptr<pcl::search::KdTree<pcl::PointXYZI>> tree_;

    // parameters
    float fl_x_, fl_y_, fl_z_, fl_roll_, fl_pitch_, fl_yaw_;
    float fr_x_, fr_y_, fr_z_, fr_roll_, fr_pitch_, fr_yaw_;
    float rr_x_, rr_y_, rr_z_, rr_roll_, rr_pitch_, rr_yaw_;
    float front_distance, rear_distance, side_distance;
    float ground_angle_threshold_;
    float pcd_height_max_, pcd_height_min_;
    double voxel_leaf_size_;
    bool use_global_angle_ground_removal_;
    bool use_voxel_grid_filter_;
    bool use_z_threshold_ground_removal_;
    bool use_transform_by_topic_name_;
    int normals_mean_k_;
    int num_threads_;
};

} // namespace tc_livox_utils