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

  std::vector<std::vector<Smoothness>> smoothness;
  std::vector<std::vector<Feature>> features;
  if (!ComputeSmoothness(smoothness, features)) {
    ROS_ERROR("Failed to compute smoothness");
    return;
  }

  std::vector<Feature> corner_features;
  ExtractCornerFeatures(smoothness, features, corner_features);

  std::vector<Feature> surface_features;
  surface_features.reserve(cloud_->height * cloud_->width);
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

bool FrontEnd::ComputeSmoothness(std::vector<std::vector<Smoothness>>& smoothness, std::vector<std::vector<Feature>>& features) const {
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

  smoothness.clear();
  smoothness.reserve(features.size());
  const int& offset = parameters_->side_points_for_curvature_calculation;
  const float num = static_cast<float>(offset) * 2.;
  for (auto& feature_per_row : features) {
    std::vector<Smoothness> smoothness_per_row;
    smoothness_per_row.reserve(feature_per_row.size());
    for (int i = offset; i < static_cast<int>(feature_per_row.size()) - offset; ++i) {
      auto& feature = feature_per_row[i];
      float temp = num * feature.range;
      for (int j = -offset; j <= offset; ++j) {
        temp -= feature_per_row.at(i + j).range;
      }
      feature.curvature = temp * temp;
      smoothness_per_row.push_back({i, feature.curvature});
    }
    smoothness_per_row.resize(smoothness_per_row.size());
    smoothness.push_back(smoothness_per_row);
  }

  return true;
}

bool FrontEnd::ExtractCornerFeatures(std::vector<std::vector<Smoothness>>& smoothness, std::vector<std::vector<Feature>>& features,
                                     std::vector<Feature>& corner_features) const {
  auto by_smoothness = [](const Smoothness& left, const Smoothness& right) { return left.curvature < right.curvature; };

  corner_features.clear();
  corner_features.reserve(cloud_->height * cloud_->width);
  for (size_t i = 0; i < smoothness.size(); ++i) {
    auto& smoothness_per_row = smoothness[i];
    auto& features_per_row = features[i];
    const int subregion_num =
        std::min(parameters_->subregion_num, static_cast<int>(smoothness_per_row.size()) / parameters_->min_points_per_subregion);
    const int index_increment = smoothness_per_row.size() / subregion_num;
    for (int j = 0; j < subregion_num; ++j) {
      const int start_index = j * index_increment;
      const int end_index = (j + 1) * index_increment - 1;
      std::sort(smoothness_per_row.begin() + start_index, smoothness_per_row.begin() + end_index, by_smoothness);

      int feature_count = 0;
      for (int k = end_index; k >= start_index; --k) {
        const int& feature_index = smoothness_per_row[k].index;
        auto& feature = features_per_row[feature_index];
        if (feature.excluded || feature.curvature < parameters_->threshold_for_corner) {
          continue;
        }

        if (++feature_count >= parameters_->max_corner_feature_count_per_subregion) {
          break;
        }

        feature.type = Feature::kCorner;
        feature.excluded = true;
        corner_features.push_back(feature);

        for (int offset = 1; offset <= parameters_->side_points_for_curvature_calculation; ++offset) {
          features_per_row[feature_index + offset].excluded = true;
          features_per_row[feature_index - offset].excluded = true;
        }
      }
    }
  }

  return true;
}

bool FrontEnd::ExtractSurfaceFeatures(std::vector<std::vector<Smoothness>>& smoothness, std::vector<std::vector<Feature>>& features,
                                      std::vector<Feature>& surface_features) const {
  return true;
}

}  // namespace lio_sam