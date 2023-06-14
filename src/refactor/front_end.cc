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
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::ProcessPointCloud, this));
  timers_.push_back(nh_.createWallTimer(ros::WallDuration(0.05), &FrontEnd::EstimateLidarPose, this));
  cloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("result", 2);
  ranges_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("ranges", 2);

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
      if (range < 0.3 || range > 5.0) {
        continue;
      }
      const int flat_index = i + j * image.cols;
      pcl::PointXYZL temp;
      temp.x = units.at(flat_index).x() * range;
      temp.y = units.at(flat_index).y() * range;
      temp.z = range;
      temp.label = 160;
      cloud_->points[i + k * image.cols] = temp;
    }
  }
  ROS_INFO("Generate cloud with %ld points", cloud_->size());
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

void FrontEnd::ProcessPointCloud(const ros::WallTimerEvent& event) {
  static constexpr uint64_t kTimeIntervalMicroSeconds = 1e4;
  std::lock_guard<std::mutex> lock(mutex_);
  if (cloud_->header.stamp - last_cloud_stamp_ < kTimeIntervalMicroSeconds) {
    return;
  }
  last_cloud_stamp_ = cloud_->header.stamp;

  std::vector<std::vector<Smoothness>> smoothness;
  std::vector<std::vector<Feature>> possible_features;
  if (!ComputeSmoothness(smoothness, possible_features)) {
    ROS_ERROR("Failed to compute smoothness");
    return;
  }

  if (!ExcludeBeamPoints(possible_features)) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZL> ranges;
  ranges.is_dense = true;
  ranges.width = cloud_->width;
  ranges.height = cloud_->height;
  ranges.resize(cloud_->width * cloud_->height);
  for (int i = 0; i < possible_features.size(); ++i) {
    const auto& features_per_row = possible_features.at(i);
    for (int j = 0; j < features_per_row.size(); ++j) {
      const auto& feature = features_per_row[j];
      auto point = cloud_->points[feature.index];
      point.z = feature.range;
      point.label = feature.excluded ? 0 : 255;
      ranges.points[feature.index] = point;
    }
  }
  sensor_msgs::PointCloud2 debug;
  pcl::toROSMsg(ranges, debug);
  debug.header.frame_id = "map";
  ranges_publisher_.publish(debug);

  std::vector<Feature> corner_features;
  std::vector<Feature> surface_features;
  if (!ExtractFeatures(smoothness, possible_features, corner_features, surface_features)) {
    ROS_ERROR("Failed to extract features");
    return;
  }

  ROS_INFO("Extracted %ld corner features and %ld surface features", corner_features.size(), surface_features.size());

  int exclude_count = 0, feature_count = 0;
  for (const auto& features_per_row : possible_features) {
    feature_count += features_per_row.size();
    for (const auto& feature : features_per_row) {
      if (feature.excluded) {
        cloud_->points[feature.index].label = 66;
        ++exclude_count;
      }
    }
  }
  ROS_INFO("Excluded %d features in %d candidates", exclude_count, feature_count);

  for (const auto& feature : corner_features) {
    cloud_->points[feature.index].label = 0;
  }

  for (const auto& feature : surface_features) {
    cloud_->points[feature.index].label = 100;
  }

  sensor_msgs::PointCloud2 message;
  pcl::toROSMsg(*cloud_, message);
  message.header.frame_id = "map";
  cloud_publisher_.publish(message);
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

bool FrontEnd::ComputeSmoothness(std::vector<std::vector<Smoothness>>& smoothness, std::vector<std::vector<Feature>>& possible_features) const {
  possible_features.clear();
  possible_features.reserve(cloud_->height);
  for (size_t j = 0; j < cloud_->height; ++j) {
    std::vector<Feature> features_per_row;
    features_per_row.reserve(cloud_->width);
    for (size_t i = 0; i < cloud_->width; ++i) {
      const int flat_index = i + j * cloud_->width;
      const float& range = cloud_->points.at(flat_index).z;
      if (range < 0.3 || range > 5.0) {
        continue;
      }
      Feature feature;
      feature.index = flat_index;
      feature.range = range;
      features_per_row.push_back(feature);
    }
    features_per_row.resize(features_per_row.size());
    if (features_per_row.empty()) {
      ROS_WARN("No possible features in row %ld", j);
    }
    possible_features.push_back(features_per_row);
  }

  smoothness.clear();
  smoothness.reserve(possible_features.size());
  const int& offset = parameters_->side_points_for_curvature_calculation;
  const float num = static_cast<float>(offset) * 2.;
  for (auto& features_per_row : possible_features) {
    std::vector<Smoothness> smoothness_per_row;
    smoothness_per_row.reserve(features_per_row.size());
    for (int i = offset; i < static_cast<int>(features_per_row.size()) - offset; ++i) {
      auto& feature = features_per_row[i];
      float temp = num * feature.range;
      for (int j = -offset; j <= offset; ++j) {
        if (j == 0) {
          continue;
        }
        temp -= features_per_row.at(i + j).range;
      }
      feature.curvature = temp * temp;
      smoothness_per_row.push_back({i, feature.curvature});
    }
    smoothness_per_row.resize(smoothness_per_row.size());
    smoothness.push_back(smoothness_per_row);
  }

  return true;
}

bool FrontEnd::ExcludeBeamPoints(std::vector<std::vector<Feature>>& possible_features) const {
  const int& offset = parameters_->side_points_for_curvature_calculation;
  for (auto& features_per_row : possible_features) {
    for (int i = offset; i < static_cast<int>(features_per_row.size()) - offset - 1; ++i) {
      const float depth0 = features_per_row[i - 1].range;
      const float depth1 = features_per_row[i].range;
      const float depth2 = features_per_row[i + 1].range;
      const float diff1 = depth0 - depth1;
      const float diff2 = depth1 - depth2;
      if (std::abs(features_per_row[i].index - features_per_row[i + 1].index) < 10 && std::abs(diff2) > 0.03) {
        for (int j = -offset; j <= offset + 1; ++j) {
          features_per_row[i + j].excluded = true;
        }
      }

      // if (std::abs(diff1) > 0.02 * depth1 && std::abs(diff2) > 0.02 * depth1) {
      //   features_per_row[i].excluded = true;
      // }
    }
  }

  return true;
}

bool FrontEnd::ExtractFeatures(std::vector<std::vector<Smoothness>>& smoothness, std::vector<std::vector<Feature>>& possible_features,
                               std::vector<Feature>& corner_features, std::vector<Feature>& surface_features) const {
  const int min_points_per_subregion = parameters_->side_points_for_curvature_calculation * 2 + 1;
  auto by_smoothness = [](const Smoothness& left, const Smoothness& right) { return left.curvature < right.curvature; };

  corner_features.clear();
  corner_features.reserve(cloud_->height * cloud_->width);
  surface_features.clear();
  surface_features.reserve(cloud_->height * cloud_->width);
  for (size_t i = 0; i < smoothness.size(); ++i) {
    auto& smoothness_per_row = smoothness[i];
    auto& features_per_row = possible_features[i];
    if (smoothness_per_row.empty() || features_per_row.empty()) {
      continue;
    }
    const int subregion_num = std::min(parameters_->subregion_num, static_cast<int>(smoothness_per_row.size()) / min_points_per_subregion);
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

      for (int k = start_index; k <= end_index; ++k) {
        const int& feature_index = smoothness_per_row[k].index;
        auto& feature = features_per_row[feature_index];
        if (feature.excluded || feature.curvature > parameters_->threshold_for_surface) {
          continue;
        }

        feature.type = Feature::kSurface;
        feature.excluded = true;
        surface_features.push_back(feature);

        for (int offset = 1; offset <= parameters_->side_points_for_curvature_calculation; ++offset) {
          features_per_row[feature_index + offset].excluded = true;
          features_per_row[feature_index - offset].excluded = true;
        }
      }
    }
  }

  corner_features.resize(corner_features.size());
  surface_features.resize(surface_features.size());
  return true;
}

}  // namespace lio_sam