#include "mid_merger/cloud_merger.hpp"
#include <pcl/filters/passthrough.h>


namespace tc_livox_utils {

PcdFilter::PcdFilter(const rclcpp::NodeOptions &options)
    : Node("pcd_filter", options), get_tf_(false)
{
    this->declare_parameter<float>("front_left_lidar_x", 0.0);
    this->declare_parameter<float>("front_left_lidar_y", 0.0);
    this->declare_parameter<float>("front_left_lidar_z", 0.0);
    this->declare_parameter<float>("front_left_lidar_roll", 0.0);
    this->declare_parameter<float>("front_left_lidar_pitch", 0.0);
    this->declare_parameter<float>("front_left_lidar_yaw", 0.0);
    this->declare_parameter<float>("front_right_lidar_x", 0.0);
    this->declare_parameter<float>("front_right_lidar_y", -26.0);
    this->declare_parameter<float>("front_right_lidar_z", 0.0);
    this->declare_parameter<float>("front_right_lidar_roll", 90.0);
    this->declare_parameter<float>("front_right_lidar_pitch", 90.0);
    this->declare_parameter<float>("front_right_lidar_yaw", 0.0);
    this->declare_parameter<float>("rear_lidar_x", 0.0);
    this->declare_parameter<float>("rear_lidar_y", 26.0);
    this->declare_parameter<float>("rear_lidar_z", 0.0);
    this->declare_parameter<float>("rear_lidar_roll", -90.0);
    this->declare_parameter<float>("rear_lidar_pitch", 90.0);
    this->declare_parameter<float>("rear_lidar_yaw", 0.0);
    this->declare_parameter<float>("front_distance", 0.35);
    this->declare_parameter<float>("rear_distance", 0.60);
    this->declare_parameter<float>("side_distance", 0.38);

    this->declare_parameter<std::string>("front_left_lidar_sub_topic", "front_left_lidar"); // 前
    this->declare_parameter<std::string>("front_right_lidar_sub_topic", "front_right_lidar"); // 左
    this->declare_parameter<std::string>("rear_lidar_sub_topic", "rear_lidar"); // 右
    this->declare_parameter<std::string>("front_left_lidar_pub_topic", "front_left_lidar_filtered");
    this->declare_parameter<std::string>("front_right_lidar_pub_topic", "front_right_lidar_filtered");
    this->declare_parameter<std::string>("rear_lidar_pub_topic", "rear_lidar_filtered");
    this->declare_parameter<std::string>("lidar_frame_id", "livox_frame");
    this->declare_parameter<std::string>("laser_pub_topic", "livox/scan");
    this->declare_parameter<std::string>("pcd_pub_topic_head", "livox");

    this->declare_parameter<double>("voxel_leaf_size", 0.1);
    this->declare_parameter<int>("num_threads", 14);
    this->declare_parameter<float>("pcd_height_max", 1.50);
    this->declare_parameter<float>("pcd_height_min", 0.01);
    this->declare_parameter<int>("normals_mean_k", 50);
    this->declare_parameter<float>("ground_angle_threshold", 10.0);
    this->declare_parameter<bool>("use_global_angle_ground_removal", false);
    this->declare_parameter<bool>("use_voxel_grid_filter", true);
    this->declare_parameter<bool>("use_z_threshold_ground_removal", false);
    this->declare_parameter<bool>("use_transform_by_topic_name", false);

    this->get_parameter("front_left_lidar_x", fl_x_);
    this->get_parameter("front_left_lidar_y", fl_y_);
    this->get_parameter("front_left_lidar_z", fl_z_);
    this->get_parameter("front_left_lidar_roll", fl_roll_);
    this->get_parameter("front_left_lidar_pitch", fl_pitch_);
    this->get_parameter("front_left_lidar_yaw", fl_yaw_);
    this->get_parameter("front_right_lidar_x", fr_x_);
    this->get_parameter("front_right_lidar_y", fr_y_);
    this->get_parameter("front_right_lidar_z", fr_z_);
    this->get_parameter("front_right_lidar_roll", fr_roll_);
    this->get_parameter("front_right_lidar_pitch", fr_pitch_);
    this->get_parameter("front_right_lidar_yaw", fr_yaw_);
    this->get_parameter("rear_lidar_x", rr_x_);
    this->get_parameter("rear_lidar_y", rr_y_);
    this->get_parameter("rear_lidar_z", rr_z_);
    this->get_parameter("rear_lidar_roll", rr_roll_);
    this->get_parameter("rear_lidar_pitch", rr_pitch_);
    this->get_parameter("rear_lidar_yaw", rr_yaw_);
    this->get_parameter("front_left_lidar_sub_topic", front_left_lidar_sub_topic_);
    this->get_parameter("front_right_lidar_sub_topic", front_right_lidar_sub_topic_);
    this->get_parameter("rear_lidar_sub_topic", rear_lidar_sub_topic_);
    this->get_parameter("front_left_lidar_pub_topic", front_left_lidar_pub_topic_);
    this->get_parameter("front_right_lidar_pub_topic", front_right_lidar_pub_topic_);   
    this->get_parameter("rear_lidar_pub_topic", rear_lidar_pub_topic_);
    this->get_parameter("pcd_pub_topic_head", pcd_pub_topic_head_);
    this->get_parameter("lidar_frame_id", lidar_frame_id_);
    this->get_parameter("laser_pub_topic", laser_pub_topic_);
    this->get_parameter("voxel_leaf_size", voxel_leaf_size_);
    this->get_parameter("num_threads", num_threads_);
    this->get_parameter("pcd_height_max", pcd_height_max_);
    this->get_parameter("pcd_height_min", pcd_height_min_);
    this->get_parameter("normals_mean_k", normals_mean_k_);
    this->get_parameter("ground_angle_threshold", ground_angle_threshold_);
    this->get_parameter("use_global_angle_ground_removal", use_global_angle_ground_removal_);
    this->get_parameter("use_voxel_grid_filter", use_voxel_grid_filter_);
    this->get_parameter("use_z_threshold_ground_removal", use_z_threshold_ground_removal_);
    this->get_parameter("use_transform_by_topic_name", use_transform_by_topic_name_);
    this->get_parameter("front_distance", front_distance);
    this->get_parameter("rear_distance", rear_distance);
    this->get_parameter("side_distance", side_distance);

    tree_ = std::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();

    // 点群のサブスクライブとトピックの同期
    front_left_lidar_sub_.subscribe(this, front_left_lidar_sub_topic_, qos_profile_lidar);
    front_right_lidar_sub_.subscribe(this, front_right_lidar_sub_topic_, qos_profile_lidar);
    rear_lidar_sub_.subscribe(this, rear_lidar_sub_topic_, qos_profile_lidar);
    sync_.reset(new Sync(SyncPolicy(10), front_left_lidar_sub_, front_right_lidar_sub_, rear_lidar_sub_));
    sync_->registerCallback(&PcdFilter::cb_lidar_sync, this);

    pub_lidar_vec_.push_back(
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            front_left_lidar_pub_topic_, qos_lidar
        )
    );
    pub_lidar_vec_.push_back(
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            front_right_lidar_pub_topic_, qos_lidar
        )
    );
    pub_lidar_vec_.push_back(
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            rear_lidar_pub_topic_, qos_lidar
        )
    );

    std::string concatendated_pcd = pcd_pub_topic_head_ + "/concatenated_pcd";
    pub_lidar_vec_.push_back(
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            concatendated_pcd, qos_lidar
        )
    );

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // transform
    tf_front_left_lidar2base_ = xyzrpy2tf(fl_x_, fl_y_, fl_z_, fl_roll_, fl_pitch_, fl_yaw_);
    tf_front_right_lidar2base_ = xyzrpy2tf(fr_x_, fr_y_, fr_z_, fr_roll_, fr_pitch_, fr_yaw_);
    tf_rear_lidar2base_ = xyzrpy2tf(rr_x_, rr_y_, rr_z_, rr_roll_, rr_pitch_, rr_yaw_);

    auto omp_max_threads = std::to_string(omp_get_max_threads());
    RCLCPP_INFO(this->get_logger(), "Max threads: %s", omp_max_threads.c_str());
    RCLCPP_INFO(this->get_logger(), "OMP threads: %d", num_threads_);
    omp_set_num_threads(num_threads_);
}

// 複数topicの点群統合 単純に一つの点群に結合するだけ
void PcdFilter::concatenate_pointclouds(
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> pointcloud_out,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl1,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl2,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl3
) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl1_pcl(new pcl::PointCloud<pcl::PointXYZI>(*pcl1_pcl));
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl2_pcl(new pcl::PointCloud<pcl::PointXYZI>(*pcl2_pcl));
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl3_pcl(new pcl::PointCloud<pcl::PointXYZI>(*pcl3_pcl));
    pcl::fromROSMsg(*pcl1, *pcl1_pcl);
    pcl::fromROSMsg(*pcl2, *pcl2_pcl);
    pcl::fromROSMsg(*pcl3, *pcl3_pcl);

    // サイドのmid上半分削除
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl2_pcl_out(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl3_pcl_out(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PassThrough<pcl::PointXYZI> pass;

    pass.setInputCloud(pcl2_pcl);   // 入力点群を設定
    pass.setFilterFieldName("x");      // フィルタリングする軸をx軸に設定
    pass.setFilterLimitsNegative(true); // フィルタの範囲外の値を保持する
    pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // x軸が負の範囲を設定
    pass.filter(*pcl2_pcl_out);

    pass.setInputCloud(pcl3_pcl);   // 入力点群を設定
    pass.setFilterFieldName("x");      // フィルタリングする軸をx軸に設定
    pass.setFilterLimitsNegative(true); // フィルタの範囲外の値を保持する
    pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // x軸が負の範囲を設定
    pass.filter(*pcl3_pcl_out);
    
    *pointcloud_out = *pcl1_pcl + *pcl2_pcl_out + *pcl3_pcl_out;
}

void PcdFilter::cb_lidar_sync(
    const sensor_msgs::msg::PointCloud2::SharedPtr front_left_lidar,
    const sensor_msgs::msg::PointCloud2::SharedPtr front_right_lidar,
    const sensor_msgs::msg::PointCloud2::SharedPtr rear_lidar
) {
    
    if (!get_tf_) {
        try {
            auto tf_lidar2base = tf_buffer_->lookupTransform("base_link", front_left_lidar->header.frame_id, rclcpp::Time(0));
            tf_lidar2base_ = std::make_shared<Eigen::Affine3f>();
            tf_lidar2base_->translation() << tf_lidar2base.transform.translation.x, tf_lidar2base.transform.translation.y, tf_lidar2base.transform.translation.z;
            tf_lidar2base_->linear() = Eigen::Quaternionf(tf_lidar2base.transform.rotation.w, tf_lidar2base.transform.rotation.x, tf_lidar2base.transform.rotation.y, tf_lidar2base.transform.rotation.z).toRotationMatrix();
            get_tf_ = true;
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
            return;
        }
    }
    
    /*
    if (use_global_angle_ground_removal_) {
        try {
            auto tf_base2map = tf_buffer_->lookupTransform("map", "base_link", rclcpp::Time(0));
            tf_base2map_ = std::make_shared<Eigen::Affine3f>();
            tf_base2map_->translation() << tf_base2map.transform.translation.x, tf_base2map.transform.translation.y, tf_base2map.transform.translation.z;
            tf_base2map_->linear() = Eigen::Quaternionf(tf_base2map.transform.rotation.w, tf_base2map.transform.rotation.x, tf_base2map.transform.rotation.y, tf_base2map.transform.rotation.z).toRotationMatrix();
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
            return;
        }
    }*/
    if (front_left_lidar->header.frame_id != lidar_frame_id_ || front_right_lidar->header.frame_id != lidar_frame_id_ || rear_lidar->header.frame_id != lidar_frame_id_) {
        RCLCPP_ERROR(this->get_logger(), "Invalid frame_id: %s, %s, %s", front_left_lidar->header.frame_id.c_str(), front_right_lidar->header.frame_id.c_str(), rear_lidar->header.frame_id.c_str());
        return;
    }

    // 点群の結合
    auto concatenated_pcd_tmp = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (use_transform_by_topic_name_) {
        auto front_left_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto front_right_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto rear_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::fromROSMsg(*front_left_lidar, *front_left_pcd);
        pcl::fromROSMsg(*front_right_lidar, *front_right_pcd);
        pcl::fromROSMsg(*rear_lidar, *rear_pcd);

        // ------------------処理軽くするためにダウンサンプリング--------------------------
        pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;

        voxel_grid_filter.setInputCloud(front_left_pcd);
        voxel_grid_filter.setLeafSize(0.10f, 0.10f, 0.10f);  // mid360の標準誤差<=0.03 ,0.05にしていた
        voxel_grid_filter.filter(*front_left_pcd);

        voxel_grid_filter.setInputCloud(front_right_pcd);
        voxel_grid_filter.setLeafSize(0.10f, 0.10f, 0.10f);  // mid360の標準誤差<=0.03 ,0.05にしていた
        voxel_grid_filter.filter(*front_right_pcd);

        voxel_grid_filter.setInputCloud(rear_pcd);
        voxel_grid_filter.setLeafSize(0.10f, 0.10f, 0.10f);  // mid360の標準誤差<=0.03 ,0.05にしていた
        voxel_grid_filter.filter(*rear_pcd);

        pcl::transformPointCloud(*front_left_pcd, *front_left_pcd, tf_front_left_lidar2base_);
        pcl::transformPointCloud(*front_right_pcd, *front_right_pcd, tf_front_right_lidar2base_);
        pcl::transformPointCloud(*rear_pcd, *rear_pcd, tf_rear_lidar2base_);

        // サイドのmid上半分削除
        pcl::PointCloud<pcl::PointXYZI>::Ptr left_mid_pcd_out(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr right_mid_pcd_out(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PassThrough<pcl::PointXYZI> pass;

        
        pass.setInputCloud(front_right_pcd);   // 入力点群を設定
        pass.setFilterFieldName("z");      // フィルタリングする軸をx軸に設定
        //pass.setFilterLimitsNegative(true); // フィルタの範囲外の値を保持する
        pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // x軸が負の範囲を設定
        pass.filter(*left_mid_pcd_out);

        pass.setInputCloud(rear_pcd);   // 入力点群を設定
        pass.setFilterFieldName("z");      // フィルタリングする軸をx軸に設定
        //pass.setFilterLimitsNegative(true); // フィルタの範囲外の値を保持する
        pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // x軸が負の範囲を設定
        pass.filter(*right_mid_pcd_out);
        

        *concatenated_pcd_tmp = *front_left_pcd + *left_mid_pcd_out + *right_mid_pcd_out;

    } else {
        concatenate_pointclouds(concatenated_pcd_tmp, front_left_lidar, front_right_lidar, rear_lidar);

    }
    auto concatenated_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    remove_self_pcd(concatenated_pcd_tmp, concatenated_pcd);
    auto concatenated_pcd_ros = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*concatenated_pcd, *concatenated_pcd_ros);
    concatenated_pcd_ros->header = front_left_lidar->header;
    pub_lidar_vec_[3]->publish(*concatenated_pcd_ros);

    // 座標変換
    auto transformed_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::transformPointCloud(*concatenated_pcd, *transformed_pcd, tf_lidar2base_->matrix());

}

} // namespace tc_livox_utils

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tc_livox_utils::PcdFilter)