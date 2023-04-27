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
  int height_down_sampling_ratio = 10;
  int subregion_num = 4;
  int side_points_for_curvature_calculation = 5;
  float threshold_for_corner = 1.;
  float threshold_for_surface = 0.;
};

struct Feature {
  enum Type { kUnknown = 0, kSurface = -1, kCorner = 1 };
  Type type = kUnknown;
  bool excluded = false;
  int index = -1;
  float curvature = -1.;
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
  void InitializeUnits(const int& width, const int& height);

  void ExtractFeatures(const ros::WallTimerEvent& event);
  void EstimateLidarPose(const ros::WallTimerEvent& event);

  void ComputeSmoothness(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& cloud, std::vector<Feature>& features) const;

  inline float distance(const pcl::PointXYZL& point1, const pcl::PointXYZL& point2 = pcl::PointXYZL()) const {
    const float dx = point1.x - point2.x;
    const float dy = point1.y - point2.y;
    const float dz = point1.z - point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

 private:
  const std::unique_ptr<Parameters> parameters_;

  uint64_t last_cloud_stamp_ = 0;

  ros::NodeHandle nh_;
  std::vector<ros::Subscriber> subscribers_;
  std::vector<ros::WallTimer> timers_;
  ros::Publisher cloud_publisher_;

  std::mutex mutex_;
  std::unordered_map<std::string, sensor_msgs::CameraInfoConstPtr> camera_info_;
  std::vector<Eigen::Vector2f> units_;
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_;
};

}  // namespace lio_sam