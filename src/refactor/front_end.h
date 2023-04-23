#pragma once

#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <eigen3/Eigen/Eigen>

namespace lio_sam {

class FrontEnd {
 public:
  FrontEnd();
  ~FrontEnd() {}

 private:
  void HandleCloudData(const sensor_msgs::PointCloud2ConstPtr& msg);
  void HandleDepthImageData(const sensor_msgs::ImageConstPtr& msg);
  void HandleOdometryData(const nav_msgs::OdometryConstPtr& msg);
  void HandleImuData(const sensor_msgs::ImuConstPtr& msg);

  inline float distance(const pcl::PointXYZL& point1, const pcl::PointXYZL& point2 = pcl::PointXYZL()) {
    const float dx = point1.x - point2.x;
    const float dy = point1.y - point2.y;
    const float dz = point1.z - point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

 private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_subscriber_;
  ros::Subscriber depth_subscriber_;
  ros::Subscriber odom_subscriber_;
  ros::Subscriber imu_subscriber_;

  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_;
  Eigen::Matrix<float, 160, 120> range_matrix_;
};

}  // namespace lio_sam