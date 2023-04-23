#include "front_end.h"

namespace lio_sam {

FrontEnd::FrontEnd() : nh_("~") {
  cloud_subscriber_ = nh_.subscribe<sensor_msgs::PointCloud2>("cloud", 2, &FrontEnd::HandleCloudData, this);
  depth_subscriber_ = nh_.subscribe<sensor_msgs::Image>("depth_frame", 2, &FrontEnd::HandleDepthImageData, this);
  odom_subscriber_ = nh_.subscribe<nav_msgs::Odometry>("odom", 2, &FrontEnd::HandleOdometryData, this);
  imu_subscriber_ = nh_.subscribe<sensor_msgs::Imu>("imu", 2, &FrontEnd::HandleImuData, this);
}

void FrontEnd::HandleCloudData(const sensor_msgs::PointCloud2ConstPtr& msg) {
  sensor_msgs::PointCloud2 temp = std::move(*msg);
  pcl::moveFromROSMsg(temp, *cloud_);
  for (size_t i = 0; i < cloud_->points.size(); ++i) {
    const auto& point = cloud_->points[i];
    const float range = distance(point);
    if (range < 0.1 || range > 2.0) {
      continue;
    }

    range_matrix_(i / 120, i % 120) = range;
  }
}

void FrontEnd::HandleDepthImageData(const sensor_msgs::ImageConstPtr& msg) {}

void FrontEnd::HandleOdometryData(const nav_msgs::OdometryConstPtr& msg) {}

void FrontEnd::HandleImuData(const sensor_msgs::ImuConstPtr& msg) {}

}  // namespace lio_sam