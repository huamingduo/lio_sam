#include "front_end.h"

#include <cv_bridge/cv_bridge.h>

#include <thread>

namespace lio_sam {

FrontEnd::FrontEnd() : parameters_(std::make_unique<Parameters>()), nh_("~"), cloud_(new pcl::PointCloud<pcl::PointXYZL>()) {
  subscribers_.push_back(nh_.subscribe<sensor_msgs::PointCloud2>("cloud", 2, &FrontEnd::HandleCloudData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::Image>("/ground_camera_node/depth_frame", 2, &FrontEnd::HandleDepthImageData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::CameraInfo>("/ground_camera_node/camera_info", 2, &FrontEnd::HandleCameraInfoData, this));
  subscribers_.push_back(nh_.subscribe<nav_msgs::Odometry>("odom", 2, &FrontEnd::HandleOdometryData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::Imu>("imu", 2, &FrontEnd::HandleImuData, this));
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::ExtractFeatures, this));
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::EstimateLidarPose, this));

  auto initialization = std::thread(std::bind(&FrontEnd::InitializeUnits, this, parameters_->width, parameters_->height));
  initialization.detach();
}

void FrontEnd::HandleCloudData(const sensor_msgs::PointCloud2ConstPtr& msg) {
  sensor_msgs::PointCloud2 temp = std::move(*msg);
  std::lock_guard<std::mutex> lock(mutex_);
  pcl::moveFromROSMsg(temp, *cloud_);
  for (size_t i = 0; i < cloud_->points.size(); ++i) {
    const auto& point = cloud_->points[i];
    const float range = distance(point);
    if (range < 0.1 || range > 2.0) {
      continue;
    }
    range_matrix_(i % 160, i / 160) = range;
  }
}

void FrontEnd::HandleDepthImageData(const sensor_msgs::ImageConstPtr& msg) {
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  if (units_.empty()) {
    return;
  }

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  const cv::Mat& image = cv_ptr->image;
  cloud_->clear();
  cloud_->is_dense = true;
  cloud_->width = image.cols;
  cloud_->height = image.rows;
  cloud_->resize(image.cols * image.rows);
  for (int j = 0; j < image.rows; ++j) {
    for (int i = 0; i < image.cols; ++i) {
      const float range = static_cast<float>(image.at<int16_t>(j, i) * 0.001);
      if (range < 0.1 || range > 5.0) {
        continue;
      }
      range_matrix_(j, i) = range;
      const int flat_index = i + j * image.cols;
      pcl::PointXYZL temp;
      temp.x = units_[flat_index].x() * range;
      temp.y = units_[flat_index].y() * range;
      temp.z = range;
      cloud_->points[flat_index] = temp;
    }
  }
  ROS_INFO("Generate cloud with %ld points", cloud_->size());
}

void FrontEnd::HandleCameraInfoData(const sensor_msgs::CameraInfoConstPtr& msg) {
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  if (camera_info_.size() == 2) {
    return;
  }
  const auto it = camera_info_.find(msg->header.frame_id);
  if (it != camera_info_.end()) {
    return;
  }

  ROS_INFO("Insert %s intrinsic parameters", msg->header.frame_id.c_str());
  camera_info_[msg->header.frame_id] = msg;
}

void FrontEnd::HandleOdometryData(const nav_msgs::OdometryConstPtr& msg) {}

void FrontEnd::HandleImuData(const sensor_msgs::ImuConstPtr& msg) {}

void FrontEnd::ExtractFeatures(const ros::WallTimerEvent& event) {
  std::unique_lock<std::mutex> lock(mutex_);
  pcl::PointCloud<pcl::PointXYZL>::Ptr surface_features;
  pcl::PointCloud<pcl::PointXYZL>::Ptr corner_features;
  lock.unlock();
}

void FrontEnd::EstimateLidarPose(const ros::WallTimerEvent& event) {}

void FrontEnd::InitializeUnits(const int& width, const int& height) {
  ros::Time start = ros::Time::now();
  while (ros::ok()) {
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      if (camera_info_.size() == 2) {
        ROS_INFO("All camera info received");
        for (auto& subscriber : subscribers_) {
          if (subscriber.getTopic() == "/ground_camera_node/camera_info") {
            subscriber.shutdown();
            break;
          }
        }
        break;
      }
    }

    if (ros::Time::now() - start > ros::Duration(10.)) {
      ROS_ERROR("Failed to obtain camera info");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  const float factor_x = static_cast<float>(parameters_->width) / static_cast<float>(camera_info_["depth"]->width);
  const float factor_y = static_cast<float>(parameters_->height) / static_cast<float>(camera_info_["depth"]->height);
  const float fx = factor_x * camera_info_["depth"]->K[0];
  const float fy = factor_y * camera_info_["depth"]->K[4];
  const float cx = factor_x * camera_info_["depth"]->K[2];
  const float cy = factor_y * camera_info_["depth"]->K[5];
  ROS_INFO("Depth intrinsics: fx - %f, fy - %f, cx - %f, cy - %f", fx, fy, cx, cy);

  units_.resize(parameters_->height * parameters_->width, Eigen::Vector2f::Zero());
  for (int j = 0; j < parameters_->height; ++j) {
    for (int i = 0; i < parameters_->width; ++i) {
      const float x = (i - cx) / fx;
      const float y = (j - cy) / fy;
      units_[i + j * parameters_->width] = Eigen::Vector2f(x, y);
    }
  }
  ROS_INFO("Initialized");
}

}  // namespace lio_sam