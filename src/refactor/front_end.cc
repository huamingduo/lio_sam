#include "front_end.h"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <thread>

namespace lio_sam {

FrontEnd::FrontEnd() : parameters_(std::make_unique<Parameters>()), nh_("~"), cloud_(boost::make_shared<pcl::PointCloud<pcl::PointXYZL>>()) {
  subscribers_.push_back(nh_.subscribe<sensor_msgs::PointCloud2>("cloud", 2, &FrontEnd::HandleCloudData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::Image>("/ground_camera_node/depth_frame", 2, &FrontEnd::HandleDepthImageData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::CameraInfo>("/ground_camera_node/camera_info", 2, &FrontEnd::HandleCameraInfoData, this));
  subscribers_.push_back(nh_.subscribe<nav_msgs::Odometry>("odom", 2, &FrontEnd::HandleOdometryData, this));
  subscribers_.push_back(nh_.subscribe<sensor_msgs::Imu>("imu", 2, &FrontEnd::HandleImuData, this));
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::ExtractFeatures, this));
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::EstimateLidarPose, this));
  cloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("result", 2);

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
  }
}

void FrontEnd::HandleDepthImageData(const sensor_msgs::ImageConstPtr& msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& units = units_;
  if (units.empty()) {
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
  cloud_->header = pcl_conversions::toPCL(msg->header);
  cloud_->is_dense = true;
  cloud_->width = image.cols;
  cloud_->height = image.rows / parameters_->height_down_sampling_ratio;
  cloud_->resize(cloud_->width * cloud_->height);
  for (int j = 0, k = 0; j < image.rows; j += parameters_->height_down_sampling_ratio, ++k) {
    for (int i = 0; i < image.cols; ++i) {
      const float range = static_cast<float>(image.at<int16_t>(j, i) * 0.001);
      if (range < 0.1 || range > 5.0) {
        continue;
      }
      const int flat_index = i + j * image.cols;
      pcl::PointXYZL temp;
      temp.x = units.at(flat_index).x() * range;
      temp.y = units.at(flat_index).y() * range;
      temp.z = range;
      cloud_->points[i + k * image.cols] = temp;
    }
  }
  ROS_INFO("Generate cloud with %ld points", cloud_->size());

  sensor_msgs::PointCloud2 message;
  pcl::toROSMsg(*cloud_, message);
  message.header.frame_id = "camera_link";
  cloud_publisher_.publish(message);
}

void FrontEnd::HandleCameraInfoData(const sensor_msgs::CameraInfoConstPtr& msg) {
  std::lock_guard<std::mutex> lock(mutex_);
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
  static constexpr uint64_t kTimeIntervalMicroSeconds = 1e4;
  std::lock_guard<std::mutex> lock(mutex_);
  if (cloud_->header.stamp - last_cloud_stamp_ < kTimeIntervalMicroSeconds) {
    return;
  }
  last_cloud_stamp_ = cloud_->header.stamp;

  std::vector<std::vector<Feature>> features;
  ComputeSmoothness(features);

  auto surface_features = boost::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
  surface_features->header.stamp = cloud_->header.stamp;
  surface_features->is_dense = true;
  surface_features->width = cloud_->width;
  surface_features->height = cloud_->height;
  surface_features->resize(cloud_->width * cloud_->height);

  auto corner_features = boost::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
  corner_features->header.stamp = cloud_->header.stamp;
  corner_features->is_dense = true;
  corner_features->width = cloud_->width;
  corner_features->height = cloud_->height;
  corner_features->resize(cloud_->width * cloud_->height);

  for (size_t j = 0; j < cloud_->height; ++j) {
    const int start_index = parameters_->side_points_for_curvature_calculation + j * cloud_->width;
    const int end_index = (j + 1) * cloud_->width - parameters_->side_points_for_curvature_calculation - 1;
    for (int i = 0; i < parameters_->subregion_num; ++i) {
      const int subregion_start_index = (start_index * (parameters_->subregion_num - i) + end_index * i) / parameters_->subregion_num;
      const int subregion_end_index = (start_index * (parameters_->subregion_num - i - 1) + end_index * (i + 1)) / parameters_->subregion_num;
      if (subregion_start_index >= subregion_end_index) {
        continue;
      }
    }
  }
}

void FrontEnd::EstimateLidarPose(const ros::WallTimerEvent& event) {}

void FrontEnd::InitializeUnits(const int& width, const int& height) {
  ros::Time start = ros::Time::now();
  while (ros::ok()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
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

  std::lock_guard<std::mutex> lock(mutex_);
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

void FrontEnd::ComputeSmoothness(std::vector<std::vector<Feature>>& features) const {
  const int& offset = parameters_->side_points_for_curvature_calculation;
  const float num = static_cast<float>(offset) * 2.;

  features.clear();
  features.reserve(cloud_->height);
  for (size_t j = 0; j < cloud_->height; ++j) {
    std::vector<Feature> features_per_row;
    features_per_row.reserve(cloud_->width);
    for (size_t i = 0; i < cloud_->width; ++i) {
      const int flat_index = i + j * cloud_->width;
      const float range = distance(cloud_->points.at(flat_index));
      if (range < 0.1 || range > 5.0) {
        continue;
      }
      Feature feature;
      feature.index = flat_index;
      feature.range = range;
      features_per_row.push_back(feature);
    }
    features_per_row.resize(features_per_row.size());
    features.push_back(features_per_row);
  }

  for (auto& feature_per_row : features) {
    for (int i = offset; i < static_cast<int>(feature_per_row.size()) - offset; ++i) {
      float temp = num * feature_per_row.at(i).range;
      for (int j = -offset; j <= offset; ++j) {
        temp -= feature_per_row.at(i + j).range;
      }
      feature_per_row[i].curvature = temp * temp;
    }
  }
}

}  // namespace lio_sam