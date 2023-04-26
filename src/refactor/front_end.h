#pragma once

#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <eigen3/Eigen/Eigen>
#include <mutex>
#include <unordered_map>

namespace lio_sam {

struct Parameters {
  int width = 160;
  int height = 120;
};

class FrontEnd {
 public:
  FrontEnd();
  ~FrontEnd() {}

 private:
  void HandleCloudData(const sensor_msgs::PointCloud2ConstPtr& msg);
  void HandleDepthImageData(const sensor_msgs::ImageConstPtr& msg);
  void HandleCameraInfoData(const sensor_msgs::CameraInfoConstPtr& msg);
  void HandleOdometryData(const nav_msgs::OdometryConstPtr& msg);
  void HandleImuData(const sensor_msgs::ImuConstPtr& msg);

  void ExtractFeatures(const ros::WallTimerEvent& event);
  void EstimateLidarPose(const ros::WallTimerEvent& event);

  void InitializeUnits(const int& width, const int& height);

  inline float distance(const pcl::PointXYZL& point1, const pcl::PointXYZL& point2 = pcl::PointXYZL()) {
    const float dx = point1.x - point2.x;
    const float dy = point1.y - point2.y;
    const float dz = point1.z - point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

 private:
  const std::unique_ptr<Parameters> parameters_;

  ros::NodeHandle nh_;
  std::vector<ros::Subscriber> subscribers_;
  std::vector<ros::WallTimer> timers_;

  std::mutex mutex_;
  std::mutex camera_info_mutex_;

  std::unordered_map<std::string, sensor_msgs::CameraInfoConstPtr> camera_info_;
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_;
  Eigen::Matrix<float, 160, 120> range_matrix_;
  std::vector<Eigen::Vector2f> units_;
};

}  // namespace lio_sam