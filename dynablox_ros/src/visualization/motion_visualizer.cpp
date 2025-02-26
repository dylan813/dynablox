#include "dynablox_ros/visualization/motion_visualizer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dynablox {

void MotionVisualizer::Config::checkParams() const {
  checkParamGT(static_point_scale, 0.f, "static_point_scale");
  checkParamGT(dynamic_point_scale, 0.f, "dynamic_point_scale");
  checkParamGT(sensor_scale, 0.f, "sensor_scale");
  checkParamGT(cluster_line_width, 0.f, "cluster_line_width");
  checkParamGT(color_wheel_num_colors, 0, "color_wheel_num_colors");
  checkColor(static_point_color, "static_point_color");
  checkColor(dynamic_point_color, "dynamic_point_color");
  checkColor(sensor_color, "sensor_color");
  checkColor(true_positive_color, "true_positive_color");
  checkColor(false_positive_color, "false_positive_color");
  checkColor(true_negative_color, "true_negative_color");
  checkColor(false_negative_color, "false_negative_color");
  checkColor(out_of_bounds_color, "out_of_bounds_color");
  checkColor(ever_free_color, "ever_free_color");
  checkColor(never_free_color, "never_free_color");
  checkColor(point_level_slice_color, "point_level_slice_color");
  checkColor(cluster_level_slice_color, "cluster_level_slice_color");
}

void MotionVisualizer::Config::setupParamsAndPrinting() {
  setupParam("global_frame_name", &global_frame_name);
  setupParam("static_point_color", &static_point_color);
  setupParam("dynamic_point_color", &dynamic_point_color);
  setupParam("sensor_color", &sensor_color);
  setupParam("static_point_scale", &static_point_scale, "m");
  setupParam("dynamic_point_scale", &dynamic_point_scale, "m");
  setupParam("sensor_scale", &sensor_scale, "m");
  setupParam("cluster_line_width", &cluster_line_width, "m");
  setupParam("color_wheel_num_colors", &color_wheel_num_colors);
  setupParam("color_clusters", &color_clusters);
  setupParam("true_positive_color", &true_positive_color);
  setupParam("false_positive_color", &false_positive_color);
  setupParam("true_negative_color", &true_negative_color);
  setupParam("false_negative_color", &false_negative_color);
  setupParam("out_of_bounds_color", &out_of_bounds_color);
  setupParam("ever_free_color", &ever_free_color);
  setupParam("never_free_color", &never_free_color);
  setupParam("point_level_slice_color", &point_level_slice_color);
  setupParam("cluster_level_slice_color", &cluster_level_slice_color);
  setupParam("slice_height", &slice_height, "m");
  setupParam("slice_relative_to_sensor", &slice_relative_to_sensor);
  setupParam("visualization_max_z", &visualization_max_z, "m");
}

void MotionVisualizer::Config::checkColor(const std::vector<float>& color,
                                          const std::string& name) const {
  checkParamEq(static_cast<int>(color.size()), 4, name + ".size");
  for (size_t i = 0; i < std::min(color.size(), std::size_t(4u)); ++i) {
    checkParamGE(color[i], 0.f, name + "[" + std::to_string(i) + "]");
    checkParamLE(color[i], 1.f, name + "[" + std::to_string(i) + "]");
  }
}

MotionVisualizer::MotionVisualizer(ros::NodeHandle nh,
                                   std::shared_ptr<TsdfLayer> tsdf_layer)
    : config_(config_utilities::getConfigFromRos<MotionVisualizer::Config>(nh)
                 .checkValid()),
      nh_(std::move(nh)),
      tsdf_layer_(std::move(tsdf_layer)) {
  // Initialize color map
  color_map_.setItemsPerRevolution(config_.color_wheel_num_colors);

  // Ensure frame_id is consistent
  if (config_.global_frame_name != "map") {
    ROS_WARN_STREAM("Frame ID mismatch. RViz expects 'map' but config specifies '" 
                    << config_.global_frame_name << "'. This may cause visualization issues.");
  }

  // Setup mesh integrator.
  mesh_layer_ = std::make_shared<voxblox::MeshLayer>(tsdf_layer_->block_size());
  voxblox::MeshIntegratorConfig mesh_config;
  mesh_integrator_ = std::make_shared<voxblox::MeshIntegrator<TsdfVoxel>>(
      mesh_config, tsdf_layer_.get(), mesh_layer_.get());

  // Advertise topics.
  setupRos();
}

void MotionVisualizer::setupRos() {
  ROS_DEBUG("Setting up ROS publishers");
  
  // Initialize publishers with queue size
  cluster_vis_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("visualization/clusters", 1);
  
  ROS_DEBUG_STREAM("Cluster visualization publisher created with topic: " 
                   << cluster_vis_pub_.getTopic());
  
  // Advertise all topics.
  const int queue_size = 10;
  sensor_pose_pub_ =
      nh_.advertise<visualization_msgs::Marker>("lidar_pose", queue_size);
  sensor_points_pub_ =
      nh_.advertise<visualization_msgs::Marker>("lidar_points", queue_size);
  detection_points_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/point/dynamic", queue_size);
  detection_points_comp_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/point/static", queue_size);
  detection_cluster_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/cluster/dynamic", queue_size);
  detection_cluster_comp_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/cluster/static", queue_size);
  detection_object_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/object/dynamic", queue_size);
  detection_object_comp_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "detections/object/static", queue_size);
  gt_point_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "ground_truth/point", queue_size);
  gt_cluster_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "ground_truth/cluster", queue_size);
  gt_object_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "ground_truth/object", queue_size);
  ever_free_pub_ =
      nh_.advertise<visualization_msgs::Marker>("ever_free", queue_size);
  never_free_pub_ =
      nh_.advertise<visualization_msgs::Marker>("never_free", queue_size);
  mesh_pub_ = nh_.advertise<voxblox_msgs::Mesh>("mesh", queue_size);
  ever_free_slice_pub_ =
      nh_.advertise<visualization_msgs::Marker>("slice/ever_free", queue_size);
  never_free_slice_pub_ =
      nh_.advertise<visualization_msgs::Marker>("slice/never_free", queue_size);
  tsdf_slice_pub_ =
      nh_.advertise<pcl::PointCloud<pcl::PointXYZI>>("slice/tsdf", queue_size);
  point_slice_pub_ =
      nh_.advertise<visualization_msgs::Marker>("slice/points", queue_size);
}

void MotionVisualizer::visualizeAll(const Cloud& cloud,
                                    const CloudInfo& cloud_info,
                                    const Clusters& clusters) {
  current_stamp_.fromNSec(cloud_info.timestamp);
  time_stamp_set_ = true;
  visualizeLidarPose(cloud_info);
  visualizeLidarPoints(cloud);
  visualizePointDetections(cloud, cloud_info);
  visualizeClusterDetections(cloud, cloud_info, clusters);
  visualizeObjectDetections(cloud, cloud_info, clusters);
  visualizeGroundTruth(cloud, cloud_info);
  visualizeMesh();
  visualizeEverFree();
  const float slice_height =
      config_.slice_relative_to_sensor
          ? config_.slice_height + cloud_info.sensor_position.z
          : config_.slice_height;
  visualizeEverFreeSlice(slice_height);
  visualizeTsdfSlice(slice_height);
  visualizeSlicePoints(cloud, cloud_info);
  visualizeClusters(clusters);
  time_stamp_set_ = false;
}

void MotionVisualizer::visualizeClusters(const Clusters& clusters,
                                        const std::string& ns) const {
  if (cluster_vis_pub_.getNumSubscribers() == 0u) {
    ROS_DEBUG("No subscribers to cluster visualization");
    return;
  }

  // Add debug logging
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }

  ROS_DEBUG_STREAM("Visualizing " << clusters.size() << " clusters");
  
  visualization_msgs::MarkerArray array_msg;
  size_t id = 0;
  size_t valid_clusters = 0;

  for (const Cluster& cluster : clusters) {
    ROS_DEBUG_STREAM("Cluster " << cluster.id << " has " << cluster.points.size() << " points");
    
    if (!cluster.points.empty()) {
      if (!cluster.aabb.isValid()) {
        ROS_WARN_STREAM("Cluster " << cluster.id << " has invalid AABB");
        continue;
      }

      const auto& min = cluster.aabb.min_corner;
      const auto& max = cluster.aabb.max_corner;
      const Eigen::Vector3f min_vec = min.getVector3fMap();
      const Eigen::Vector3f max_vec = max.getVector3fMap();

      ROS_DEBUG_STREAM("Cluster " << cluster.id 
                      << "\nMin corner: (" << min_vec.transpose() << ")"
                      << "\nMax corner: (" << max_vec.transpose() << ")"
                      << "\nNum points: " << cluster.points.size()
                      << "\nFrame ID: " << config_.global_frame_name);

      // Additional validation before visualization
      if (!isValidAABB(min_vec, max_vec)) {
        ROS_WARN_STREAM("Invalid AABB detected before visualization: min(" 
                       << min_vec.transpose() << ") max(" << max_vec.transpose() << ")");
        continue;
      }

      valid_clusters++;
      // Create visualization marker
      visualization_msgs::Marker box_msg;
      box_msg.header.frame_id = config_.global_frame_name;  // Use configured frame
      box_msg.header.stamp = getStamp();
      box_msg.ns = ns;
      box_msg.id = id++;
      box_msg.type = visualization_msgs::Marker::LINE_LIST;
      box_msg.action = visualization_msgs::Marker::ADD;
      box_msg.pose.orientation.w = 1.0;
      box_msg.scale.x = config_.cluster_line_width;
      box_msg.color = setColor(color_map_.colorLookup(cluster.id));

      // Validate transformed points
      const Eigen::Vector3f base = min_vec;
      const Eigen::Vector3f delta = max_vec - base;
      
      if (delta.minCoeff() < 0) {
        ROS_WARN("Invalid box dimensions after transform");
        continue;
      }

      // Box corners calculation
      const Eigen::Vector3f dx = delta.cwiseProduct(Eigen::Vector3f::UnitX());
      const Eigen::Vector3f dy = delta.cwiseProduct(Eigen::Vector3f::UnitY());
      const Eigen::Vector3f dz = delta.cwiseProduct(Eigen::Vector3f::UnitZ());

      // Add box lines
      box_msg.points.push_back(setPoint(base));
      box_msg.points.push_back(setPoint(base + dx));
      box_msg.points.push_back(setPoint(base));
      box_msg.points.push_back(setPoint(base + dy));
      box_msg.points.push_back(setPoint(base + dx));
      box_msg.points.push_back(setPoint(base + dx + dy));
      box_msg.points.push_back(setPoint(base + dy));
      box_msg.points.push_back(setPoint(base + dx + dy));

      box_msg.points.push_back(setPoint(base + dz));
      box_msg.points.push_back(setPoint(base + dx + dz));
      box_msg.points.push_back(setPoint(base + dz));
      box_msg.points.push_back(setPoint(base + dy + dz));
      box_msg.points.push_back(setPoint(base + dx + dz));
      box_msg.points.push_back(setPoint(base + dx + dy + dz));
      box_msg.points.push_back(setPoint(base + dy + dz));
      box_msg.points.push_back(setPoint(base + dx + dy + dz));

      box_msg.points.push_back(setPoint(base));
      box_msg.points.push_back(setPoint(base + dz));
      box_msg.points.push_back(setPoint(base + dx));
      box_msg.points.push_back(setPoint(base + dx + dz));
      box_msg.points.push_back(setPoint(base + dy));
      box_msg.points.push_back(setPoint(base + dy + dz));
      box_msg.points.push_back(setPoint(base + dx + dy));
      box_msg.points.push_back(setPoint(base + dx + dy + dz));
      array_msg.markers.push_back(box_msg);

      // Text label
      visualization_msgs::Marker text_msg;
      text_msg.action = visualization_msgs::Marker::ADD;
      text_msg.id = id++;
      text_msg.ns = ns;
      text_msg.header.stamp = getStamp();
      text_msg.header.frame_id = config_.global_frame_name;
      text_msg.color = setColor(color_map_.colorLookup(cluster.id));
      text_msg.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_msg.scale.z = 0.5;
      text_msg.pose.position = setPoint(max_vec);  // Use validated max vector
      text_msg.pose.orientation.w = 1.f;
      
      std::stringstream stream;
      stream << cluster.points.size() << "pts - " << std::fixed
             << std::setprecision(1) << delta.norm() << "m";
      text_msg.text = stream.str();
      array_msg.markers.push_back(text_msg);
    }
  }

  ROS_DEBUG_STREAM("Publishing " << array_msg.markers.size() 
                  << " markers for " << valid_clusters << " valid clusters");

  if (!array_msg.markers.empty()) {
    cluster_vis_pub_.publish(array_msg);
  } else {
    ROS_WARN("No valid markers to publish");
  }
}

void MotionVisualizer::visualizeEverFree() const {
  const bool ever_free = ever_free_pub_.getNumSubscribers() > 0u;
  const bool never_free = never_free_pub_.getNumSubscribers() > 0u;

  if (!ever_free && !never_free) {
    return;
  }

  visualization_msgs::Marker result;
  visualization_msgs::Marker result_never;

  if (ever_free) {
    // Common properties.
    result.action = visualization_msgs::Marker::ADD;
    result.id = 0;
    result.header.stamp = getStamp();
    result.header.frame_id = config_.global_frame_name;
    result.type = visualization_msgs::Marker::CUBE_LIST;
    result.color = setColor(config_.ever_free_color);
    result.scale = setScale(tsdf_layer_->voxel_size());
    result.pose.orientation.w = 1.f;
  }

  if (never_free) {
    result_never.action = visualization_msgs::Marker::ADD;
    result_never.id = 0;
    result_never.header.stamp = getStamp();
    result_never.header.frame_id = config_.global_frame_name;
    result_never.type = visualization_msgs::Marker::CUBE_LIST;
    result_never.color = setColor(config_.never_free_color);
    result_never.scale = setScale(tsdf_layer_->voxel_size());
    result_never.pose.orientation.w = 1.f;
  }

  voxblox::BlockIndexList block_list;
  tsdf_layer_->getAllAllocatedBlocks(&block_list);
  for (const auto& index : block_list) {
    const TsdfBlock& block = tsdf_layer_->getBlockByIndex(index);
    for (size_t linear_index = 0; linear_index < block.num_voxels();
         ++linear_index) {
      const TsdfVoxel& voxel = block.getVoxelByLinearIndex(linear_index);

      if (voxel.weight < 1e-6) {
        continue;  // Unknown voxel.
      }

      const voxblox::Point coords =
          block.computeCoordinatesFromLinearIndex(linear_index);
      if (coords.z() > config_.visualization_max_z) {
        continue;
      }

      if (voxel.ever_free && ever_free) {
        result.points.push_back(setPoint(coords));
      } else if (!voxel.ever_free && never_free) {
        result_never.points.push_back(setPoint(coords));
      }
    }
  }

  if (!result.points.empty()) {
    ever_free_pub_.publish(result);
  }

  if (!result_never.points.empty()) {
    never_free_pub_.publish(result_never);
  }
}

void MotionVisualizer::visualizeEverFreeSlice(const float slice_height) const {
  const bool ever_free = ever_free_slice_pub_.getNumSubscribers() > 0u;
  const bool never_free = never_free_slice_pub_.getNumSubscribers() > 0u;

  if (!ever_free && !never_free) {
    return;
  }

  visualization_msgs::Marker result;
  visualization_msgs::Marker result_never;

  if (ever_free) {
    // Common properties.
    result.action = visualization_msgs::Marker::ADD;
    result.id = 0;
    result.header.stamp = getStamp();
    result.header.frame_id = config_.global_frame_name;
    result.type = visualization_msgs::Marker::CUBE_LIST;
    result.color = setColor(config_.ever_free_color);
    result.scale = setScale(tsdf_layer_->voxel_size());
    result.scale.z = 0.01f;
    result.pose.orientation.w = 1.f;
  }

  if (never_free) {
    result_never.action = visualization_msgs::Marker::ADD;
    result_never.id = 0;
    result_never.header.stamp = getStamp();
    result_never.header.frame_id = config_.global_frame_name;
    result_never.type = visualization_msgs::Marker::CUBE_LIST;
    result_never.color = setColor(config_.never_free_color);
    result_never.scale = setScale(tsdf_layer_->voxel_size());
    result_never.scale.z = 0.01f;
    result_never.pose.orientation.w = 1.f;
  }

  // Setup the slice.
  const voxblox::Point slice_coords(0, 0, slice_height);
  const BlockIndex slice_block_index =
      tsdf_layer_->computeBlockIndexFromCoordinates(slice_coords);
  const VoxelIndex slice_voxel_index =
      voxblox::getGridIndexFromPoint<VoxelIndex>(
          slice_coords - voxblox::getOriginPointFromGridIndex(
                             slice_block_index, tsdf_layer_->block_size()),
          tsdf_layer_->voxel_size_inv());

  // Visualize.
  voxblox::BlockIndexList block_list;
  tsdf_layer_->getAllAllocatedBlocks(&block_list);
  const float offset = tsdf_layer_->voxel_size() / 2.f;
  for (const auto& index : block_list) {
    if (index.z() != slice_block_index.z()) {
      continue;
    }
    const TsdfBlock& block = tsdf_layer_->getBlockByIndex(index);
    for (size_t x = 0; x < block.voxels_per_side(); ++x) {
      for (size_t y = 0; y < block.voxels_per_side(); ++y) {
        const VoxelIndex index(x, y, slice_voxel_index.z());
        const TsdfVoxel& voxel = block.getVoxelByVoxelIndex(index);

        if (voxel.weight < 1e-6) {
          continue;  // Unknown voxel.
        }

        voxblox::Point coords = block.computeCoordinatesFromVoxelIndex(index);
        coords.z() -= offset;

        if (voxel.ever_free && ever_free) {
          result.points.push_back(setPoint(coords));
        } else if ((!voxel.ever_free) && never_free) {
          result_never.points.push_back(setPoint(coords));
        }
      }
    }
  }

  if (!result.points.empty()) {
    ever_free_slice_pub_.publish(result);
  }

  if (!result_never.points.empty()) {
    never_free_slice_pub_.publish(result_never);
  }
}

void MotionVisualizer::visualizeTsdfSlice(const float slice_height) const {
  if (tsdf_slice_pub_.getNumSubscribers() == 0u) {
    return;
  }
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  voxblox::createDistancePointcloudFromTsdfLayerSlice(
      *tsdf_layer_, 2u, slice_height, &pointcloud);

  pointcloud.header.frame_id = config_.global_frame_name;
  pointcloud.header.stamp = getStamp().toNSec();
  tsdf_slice_pub_.publish(pointcloud);
}

void MotionVisualizer::visualizeSlicePoints(const Cloud& cloud,
                                            const CloudInfo& cloud_info) const {
  if (point_slice_pub_.getNumSubscribers() == 0u) {
    return;
  }

  visualization_msgs::Marker result;
  result.action = visualization_msgs::Marker::ADD;
  result.id = 0;
  result.header.stamp = getStamp();
  result.header.frame_id = config_.global_frame_name;
  result.type = visualization_msgs::Marker::POINTS;
  result.scale = setScale(config_.dynamic_point_scale);

  visualization_msgs::Marker result_comp;
  result_comp.action = visualization_msgs::Marker::ADD;
  result_comp.id = 1;
  result_comp.header.stamp = getStamp();
  result_comp.header.frame_id = config_.global_frame_name;
  result_comp.type = visualization_msgs::Marker::POINTS;
  result_comp.color = setColor(config_.static_point_color);
  result_comp.scale = setScale(config_.static_point_scale);

  const float slice_height =
      config_.slice_relative_to_sensor
          ? config_.slice_height + cloud_info.sensor_position.z
          : config_.slice_height;
  const float slice_center =
      std::round((slice_height * tsdf_layer_->voxel_size_inv()) + 0.5) *
      tsdf_layer_->voxel_size();
  const float min_z = slice_center - tsdf_layer_->voxel_size() / 2.f;
  const float max_z = slice_center + tsdf_layer_->voxel_size() / 2.f;

  // Get all points.
  int i = -1;
  for (const auto& point : cloud.points) {
    ++i;
    if (point.z < min_z || point.z > max_z) {
      continue;
    }
    if (cloud_info.points[i].ever_free_level_dynamic) {
      result.points.push_back(setPoint(point));
      result.colors.push_back(setColor(config_.point_level_slice_color));
    } else if (cloud_info.points[i].cluster_level_dynamic) {
      result.points.push_back(setPoint(point));
      result.colors.push_back(setColor(config_.cluster_level_slice_color));
    } else {
      result_comp.points.push_back(setPoint(point));
    }
  }
  if (!result.points.empty()) {
    point_slice_pub_.publish(result);
  }
  if (!result_comp.points.empty()) {
    point_slice_pub_.publish(result_comp);
  }
}

void MotionVisualizer::visualizeGroundTruth(const Cloud& cloud,
                                            const CloudInfo& cloud_info,
                                            const std::string& ns) const {
  if (!cloud_info.has_labels) {
    return;
  }
  // Go through all levels if it has subscribers.
  if (gt_point_pub_.getNumSubscribers() > 0) {
    visualizeGroundTruthAtLevel(
        cloud, cloud_info,
        [](const PointInfo& point) { return point.ever_free_level_dynamic; },
        gt_point_pub_, ns);
  }
  if (gt_cluster_pub_.getNumSubscribers() > 0) {
    visualizeGroundTruthAtLevel(
        cloud, cloud_info,
        [](const PointInfo& point) { return point.cluster_level_dynamic; },
        gt_cluster_pub_, ns);
  }
  if (gt_object_pub_.getNumSubscribers() > 0) {
    visualizeGroundTruthAtLevel(
        cloud, cloud_info,
        [](const PointInfo& point) { return point.object_level_dynamic; },
        gt_object_pub_, ns);
  }
}

void MotionVisualizer::visualizeGroundTruthAtLevel(
    const Cloud& cloud, const CloudInfo& cloud_info,
    const std::function<bool(const PointInfo&)>& check_level,
    const ros::Publisher& pub, const std::string& ns) const {
  // Common properties.
  visualization_msgs::Marker result;
  result.action = visualization_msgs::Marker::ADD;
  result.id = 0;
  result.ns = ns;
  result.header.stamp = getStamp();
  result.header.frame_id = config_.global_frame_name;
  result.type = visualization_msgs::Marker::POINTS;
  result.scale = setScale(config_.dynamic_point_scale);

  visualization_msgs::Marker comp = result;
  comp.scale = setScale(config_.static_point_scale);
  comp.id = 1;

  // Get all points.
  size_t i = 0;
  for (const auto& point : cloud.points) {
    const PointInfo& info = cloud_info.points[i];
    ++i;
    if (point.z > config_.visualization_max_z) {
      continue;
    }
    if (!info.ready_for_evaluation) {
      comp.points.push_back(setPoint(point));
      comp.colors.push_back(setColor(config_.out_of_bounds_color));
    } else if (check_level(info) && info.ground_truth_dynamic) {
      result.points.push_back(setPoint(point));
      result.colors.push_back(setColor(config_.true_positive_color));
    } else if (check_level(info) && !info.ground_truth_dynamic) {
      result.points.push_back(setPoint(point));
      result.colors.push_back(setColor(config_.false_positive_color));
    } else if (!check_level(info) && info.ground_truth_dynamic) {
      result.points.push_back(setPoint(point));
      result.colors.push_back(setColor(config_.false_negative_color));
    } else {
      comp.points.push_back(setPoint(point));
      comp.colors.push_back(setColor(config_.true_negative_color));
    }
  }
  pub.publish(result);
  pub.publish(comp);
}

void MotionVisualizer::visualizeLidarPose(const CloudInfo& cloud_info) const {
  if (sensor_pose_pub_.getNumSubscribers() == 0u) {
    return;
  }
  visualization_msgs::Marker result;
  result.action = visualization_msgs::Marker::ADD;
  result.id = 0;
  result.header.stamp = getStamp();
  result.header.frame_id = config_.global_frame_name;
  result.type = visualization_msgs::Marker::SPHERE;
  result.color = setColor(config_.sensor_color);
  result.scale = setScale(config_.sensor_scale);
  result.pose.position = setPoint(cloud_info.sensor_position);
  result.pose.orientation.w = 1.0;
  sensor_pose_pub_.publish(result);
}

void MotionVisualizer::visualizeLidarPoints(const Cloud& cloud) const {
  if (sensor_points_pub_.getNumSubscribers() == 0u) {
    return;
  }
  visualization_msgs::Marker result;
  result.points.reserve(cloud.points.size());

  // Common properties.
  result.action = visualization_msgs::Marker::ADD;
  result.id = 0;
  result.header.stamp = getStamp();
  result.header.frame_id = config_.global_frame_name;
  result.type = visualization_msgs::Marker::POINTS;
  result.color = setColor(config_.static_point_color);
  result.scale = setScale(config_.static_point_scale);

  // Get all points.
  for (const auto& point : cloud.points) {
    if (point.z > config_.visualization_max_z) {
      continue;
    }
    result.points.push_back(setPoint(point));
  }
  if (!result.points.empty()) {
    sensor_points_pub_.publish(result);
  }
}

void MotionVisualizer::visualizePointDetections(
    const Cloud& cloud, const CloudInfo& cloud_info) const {
  const bool dynamic = detection_points_pub_.getNumSubscribers() > 0u;
  const bool comp = detection_points_comp_pub_.getNumSubscribers() > 0u;

  if (!dynamic && !comp) {
    return;
  }

  visualization_msgs::Marker result;
  visualization_msgs::Marker result_comp;

  if (dynamic) {
    // Common properties.
    result.points.reserve(cloud.points.size());
    result.action = visualization_msgs::Marker::ADD;
    result.id = 0;
    result.header.stamp = getStamp();
    result.header.frame_id = config_.global_frame_name;
    result.type = visualization_msgs::Marker::POINTS;
    result.color = setColor(config_.dynamic_point_color);
    result.scale = setScale(config_.dynamic_point_scale);
  }

  if (comp) {
    result_comp.points.reserve(cloud.points.size());
    result_comp.action = visualization_msgs::Marker::ADD;
    result_comp.id = 0;
    result_comp.header.stamp = getStamp();
    result_comp.header.frame_id = config_.global_frame_name;
    result_comp.type = visualization_msgs::Marker::POINTS;
    result_comp.color = setColor(config_.static_point_color);
    result_comp.scale = setScale(config_.static_point_scale);
  }

  // Get all points.
  int i = -1;
  for (const auto& point : cloud.points) {
    ++i;
    if (point.z > config_.visualization_max_z) {
      continue;
    }
    if (cloud_info.points[i].ever_free_level_dynamic) {
      if (!dynamic) {
        continue;
      }
      result.points.push_back(setPoint(point));
    } else {
      if (!comp) {
        continue;
      }
      result_comp.points.push_back(setPoint(point));
    }
  }
  if (!result.points.empty()) {
    detection_points_pub_.publish(result);
  }
  if (!result_comp.points.empty()) {
    detection_points_comp_pub_.publish(result_comp);
  }
}

void MotionVisualizer::visualizeClusterDetections(
    const Cloud& cloud, const CloudInfo& cloud_info,
    const Clusters& clusters) const {
  const bool dynamic = detection_cluster_pub_.getNumSubscribers() > 0u;
  const bool comp = detection_cluster_comp_pub_.getNumSubscribers() > 0u;

  if (!dynamic && !comp) {
    return;
  }

  visualization_msgs::Marker result;
  visualization_msgs::Marker result_comp;

  if (dynamic) {
    // We just reserve too much space to save compute.
    result.points.reserve(cloud.points.size());
    result.action = visualization_msgs::Marker::ADD;
    result.id = 0;
    result.header.stamp = getStamp();
    result.header.frame_id = config_.global_frame_name;
    result.type = visualization_msgs::Marker::POINTS;
    result.scale = setScale(config_.dynamic_point_scale);
  }

  if (comp) {
    result_comp.points.reserve(cloud.points.size());
    result_comp.action = visualization_msgs::Marker::ADD;
    result_comp.id = 0;
    result_comp.header.stamp = getStamp();
    result_comp.header.frame_id = config_.global_frame_name;
    result_comp.type = visualization_msgs::Marker::POINTS;
    result_comp.color = setColor(config_.static_point_color);
    result_comp.scale = setScale(config_.static_point_scale);
  }

  // Get all cluster points.
  int i = 0;
  for (const Cluster& cluster : clusters) {
    if (!cluster.points.empty() && cluster.aabb.isValid()) {
        const auto& min = cluster.aabb.min_corner;
        const auto& max = cluster.aabb.max_corner;
        const Eigen::Vector3f min_vec = min.getVector3fMap();
        const Eigen::Vector3f max_vec = max.getVector3fMap();
        
        if (isValidAABB(min_vec, max_vec)) {
            std_msgs::ColorRGBA color;
            if (config_.color_clusters) {
                color = setColor(color_map_.colorLookup(i));
                ++i;
            } else {
                color = setColor(config_.dynamic_point_color);
            }
            for (int index : cluster.points) {
                if (          cloud[index].z > config_.visualization_max_z) {
                    continue;
                }
                result.points.push_back(setPoint(cloud[index]));
                result.colors.push_back(color);
            }
        } else {
            ROS_WARN("Invalid AABB detected in cluster detection");
        }
    }
  }

  // Get all other points.
  if (comp) {
    size_t i = 0;
    for (const auto& point : cloud.points) {
      if (point.z > config_.visualization_max_z) {
        ++i;
        continue;
      }
      if (!cloud_info.points[i].cluster_level_dynamic) {
        result_comp.points.push_back(setPoint(point));
      }
      ++i;
    }
  }

  if (!result.points.empty()) {
    detection_cluster_pub_.publish(result);
  }
  if (!result_comp.points.empty()) {
    detection_cluster_comp_pub_.publish(result_comp);
  }
}

void MotionVisualizer::visualizeObjectDetections(
    const Cloud& cloud, const CloudInfo& cloud_info,
    const Clusters& clusters) const {
  // TODO(schmluk): This is currently copied from the clusters, it simply tries
  // to do color associations for a bit more consistency during visualization.
  const bool dynamic = detection_object_pub_.getNumSubscribers() > 0u;
  const bool comp = detection_object_comp_pub_.getNumSubscribers() > 0u;

  if (!dynamic && !comp) {
    return;
  }

  visualization_msgs::Marker result;
  visualization_msgs::Marker result_comp;

  if (dynamic) {
    // We just reserve too much space to save compute.
    result.points.reserve(cloud.points.size());
    result.action = visualization_msgs::Marker::ADD;
    result.id = 0;
    result.header.stamp = getStamp();
    result.header.frame_id = config_.global_frame_name;
    result.type = visualization_msgs::Marker::POINTS;
    result.scale = setScale(config_.dynamic_point_scale);
  }

  if (comp) {
    result_comp = result;
    result_comp.color = setColor(config_.static_point_color);
    result_comp.scale = setScale(config_.static_point_scale);
  }

  // Get all cluster points.
  for (const Cluster& cluster : clusters) {
    if (cluster.valid && !cluster.points.empty() && cluster.aabb.isValid()) {
        const auto& min = cluster.aabb.min_corner;
        const auto& max = cluster.aabb.max_corner;
        const Eigen::Vector3f min_vec = min.getVector3fMap();
        const Eigen::Vector3f max_vec = max.getVector3fMap();
        
        if (isValidAABB(min_vec, max_vec)) {
            std_msgs::ColorRGBA color;
            if (config_.color_clusters) {
                color = setColor(color_map_.colorLookup(cluster.id));
            } else {
                color = setColor(config_.dynamic_point_color);
            }
            for (int index : cluster.points) {
                if (          cloud[index].z > config_.visualization_max_z) {
                    continue;
                }
                result.points.push_back(setPoint(cloud[index]));
                result.colors.push_back(color);
            }
        } else {
            ROS_WARN("Invalid AABB detected in object detection");
        }
    }
  }

  // Get all other points.
  if (comp) {
    size_t i = 0;
    for (const auto& point : cloud.points) {
      if (point.z > config_.visualization_max_z) {
        ++i;
        continue;
      }
      if (!cloud_info.points[i].object_level_dynamic) {
        result_comp.points.push_back(setPoint(point));
      }
      ++i;
    }
  }

  if (!result.points.empty()) {
    detection_object_pub_.publish(result);
  }
  if (!result_comp.points.empty()) {
    detection_object_comp_pub_.publish(result_comp);
  }
}

void MotionVisualizer::visualizeMesh() const {
  if (mesh_pub_.getNumSubscribers() == 0u) {
    return;
  }
  mesh_integrator_->generateMesh(true, true);
  voxblox_msgs::Mesh mesh_msg;
  voxblox::generateVoxbloxMeshMsg(mesh_layer_, voxblox::ColorMode::kLambert,
                                  &mesh_msg);
  mesh_msg.header.frame_id = config_.global_frame_name;
  mesh_msg.header.stamp = getStamp();
  mesh_pub_.publish(mesh_msg);
}

geometry_msgs::Vector3 MotionVisualizer::setScale(const float scale) {
  geometry_msgs::Vector3 msg;
  msg.x = scale;
  msg.y = scale;
  msg.z = scale;
  return msg;
}

std_msgs::ColorRGBA MotionVisualizer::setColor(
    const std::vector<float>& color) {
  std_msgs::ColorRGBA msg;
  msg.r = color[0];
  msg.g = color[1];
  msg.b = color[2];
  msg.a = color[3];
  return msg;
}

std_msgs::ColorRGBA MotionVisualizer::setColor(const voxblox::Color& color) {
  std_msgs::ColorRGBA msg;
  msg.r = static_cast<float>(color.r) / 255.f;
  msg.g = static_cast<float>(color.g) / 255.f;
  msg.b = static_cast<float>(color.b) / 255.f;
  msg.a = static_cast<float>(color.a) / 255.f;
  return msg;
}

geometry_msgs::Point MotionVisualizer::setPoint(const Point& point) {
  geometry_msgs::Point msg;
  msg.x = point.x;
  msg.y = point.y;
  msg.z = point.z;
  return msg;
}

geometry_msgs::Point MotionVisualizer::setPoint(const voxblox::Point& point) {
  geometry_msgs::Point msg;
  msg.x = point.x();
  msg.y = point.y();
  msg.z = point.z();
  return msg;
}

ros::Time MotionVisualizer::getStamp() const {
  if (time_stamp_set_) {
    return current_stamp_;
  } else {
    return ros::Time::now();
  }
}

bool MotionVisualizer::isValidAABB(const Eigen::Vector3f& min_vec, const Eigen::Vector3f& max_vec) const {
    return min_vec.x() <= max_vec.x() && 
           min_vec.y() <= max_vec.y() && 
           min_vec.z() <= max_vec.z();
}

}  // namespace dynablox
