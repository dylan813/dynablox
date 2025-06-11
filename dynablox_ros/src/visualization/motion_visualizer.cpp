#include <algorithm>
#include "dynablox_ros/visualization/motion_visualizer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

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
  color_map_.setItemsPerRevolution(config_.color_wheel_num_colors);
  // Setup mesh integrator.
  mesh_layer_ = std::make_shared<voxblox::MeshLayer>(tsdf_layer_->block_size());
  voxblox::MeshIntegratorConfig mesh_config;
  mesh_integrator_ = std::make_shared<voxblox::MeshIntegrator<TsdfVoxel>>(
      mesh_config, tsdf_layer_.get(), mesh_layer_.get());

  // Advertise topics.
  setupRos();
}

void MotionVisualizer::setupRos() {
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
  cluster_vis_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("clusters", queue_size);
}

void MotionVisualizer::visualizeAll(const Cloud& cloud,
                                    const CloudInfo& cloud_info,
                                    const Clusters& clusters) {
  current_stamp_.fromNSec(cloud_info.timestamp);
  time_stamp_set_ = true;
  visualizeLidarPose(cloud_info);
  visualizeLidarPoints(cloud);
  visualizePointDetections(cloud, cloud_info, clusters);
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
    return;
  }

  // Visualize Bbox.
  visualization_msgs::MarkerArray array_msg;

  size_t id = 0;
  for (const Cluster& cluster : clusters) {
    // Check if the cluster should be visualized
    bool visualize = true;
    if (classifications_.count(cluster.id)) {
      if (classifications_.at(cluster.id) == "false") {
        visualize = false;
      }
    }

    if (visualize) {
      // Bounding Box (only for 'human' clusters)
      if (classifications_.count(cluster.id) &&
          classifications_.at(cluster.id) == "human") {
        visualization_msgs::Marker cube_msg;
        cube_msg.action = visualization_msgs::Marker::ADD;
        cube_msg.id = id++;
        cube_msg.ns = ns;
        cube_msg.header.stamp = getStamp();
        cube_msg.header.frame_id = config_.global_frame_name;
        cube_msg.type = visualization_msgs::Marker::CUBE;

        // Get corners as Eigen vectors to perform math.
        const Eigen::Vector3f min_vec =
            cluster.aabb.min_corner.getVector3fMap();
        const Eigen::Vector3f max_vec =
            cluster.aabb.max_corner.getVector3fMap();

        // VITAL CHECK: Ensure min <= max to prevent RViz crash.
        const Eigen::Vector3f dimensions = max_vec - min_vec;
        if (dimensions.x() < 0.f || dimensions.y() < 0.f || dimensions.z() < 0.f) {
           ROS_WARN_THROTTLE(1.0, "Skipping visualization of invalid cluster %d: min_corner > max_corner.", cluster.id);
           continue;
        }
        
        // Set the pose of the CUBE to the center of the AABB
        const Eigen::Vector3f center = min_vec + dimensions / 2.0;

        ROS_INFO(
            "Visualizing Cluster ID %d with dimensions (x: %f, y: %f, z: %f)",
            cluster.id, dimensions.x(), dimensions.y(), dimensions.z());

        cube_msg.pose.position = setPoint(center);
        cube_msg.pose.orientation.w = 1.0;

        // Set the scale of the CUBE to the dimensions of the AABB, ensuring no
        // zero values.
        const float kMinDimension = 1e-4;
        cube_msg.scale.x = std::max(dimensions.x(), kMinDimension);
        cube_msg.scale.y = std::max(dimensions.y(), kMinDimension);
        cube_msg.scale.z = std::max(dimensions.z(), kMinDimension);

        // Set the color and transparency
        cube_msg.color = setColor(color_map_.colorLookup(cluster.id));
        cube_msg.color.a = 0.4;  // Make it semi-transparent

        array_msg.markers.push_back(cube_msg);
      }

      // Text Label
      visualization_msgs::Marker text_msg;
      text_msg.action = visualization_msgs::Marker::ADD;
      text_msg.id = id++;
      text_msg.ns = ns;
      text_msg.header.stamp = getStamp();
      text_msg.header.frame_id = config_.global_frame_name;
      if (classifications_.count(cluster.id)) {
        // Simple color scheme: green for classified, default otherwise.
        std_msgs::ColorRGBA color;
        color.r = 0.f;
        color.g = 1.f;
        color.b = 0.f;
        color.a = 0.5f;  // Make it slightly transparent
        text_msg.color = color;
      } else {
        text_msg.color = setColor(color_map_.colorLookup(cluster.id));
      }
      text_msg.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_msg.scale.z = 0.5;
      text_msg.pose.position = setPoint(cluster.aabb.max_corner);
      text_msg.pose.orientation.w = 1.f;
      const float extent = cluster.aabb.extent();
      std::stringstream stream;
      if (classifications_.count(cluster.id)) {
        stream << classifications_.at(cluster.id) << " - ";
      }
      stream << cluster.points.size() << "pts - " << std::fixed
             << std::setprecision(1) << extent << "m";
      text_msg.text = stream.str();
      array_msg.markers.push_back(text_msg);
    }
  }
  if (!array_msg.markers.empty()) {
    cluster_vis_pub_.publish(array_msg);
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
                                            const CloudInfo& cloud_info) const {
  if (gt_point_pub_.getNumSubscribers() == 0u) {
    return;
  }
  // Create a marker for all dynamic ground truth points.
  visualization_msgs::Marker points_msg;
  points_msg.action = visualization_msgs::Marker::ADD;
  points_msg.id = 0;
  points_msg.ns = "ground_truth_points";
  points_msg.header.stamp = getStamp();
  points_msg.header.frame_id = config_.global_frame_name;
  points_msg.type = visualization_msgs::Marker::POINTS;
  points_msg.scale = setScale(config_.dynamic_point_scale);
  points_msg.color = setColor(config_.true_positive_color);

  for (size_t i = 0; i < cloud.points.size(); ++i) {
    const PointInfo& point_info = cloud_info.points[i];
    const Point& point = cloud.points[i];
    if (point_info.ground_truth_dynamic) {
      points_msg.points.push_back(setPoint(point));
    }
  }
  gt_point_pub_.publish(points_msg);
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
    const Cloud& cloud, const CloudInfo& cloud_info,
    const Clusters& clusters) const {
  if (detection_points_pub_.getNumSubscribers() == 0u &&
      detection_points_comp_pub_.getNumSubscribers() == 0u) {
    return;
  }
  // All points that are not part of any cluster are considered static.
  visualization_msgs::Marker msg;
  msg.action = visualization_msgs::Marker::ADD;
  msg.ns = "detections_static_points";
  msg.id = 0;
  msg.header.stamp = getStamp();
  msg.header.frame_id = config_.global_frame_name;
  msg.type = visualization_msgs::Marker::SPHERE_LIST;
  msg.scale = setScale(config_.static_point_scale);
  msg.color = setColor(config_.static_point_color);
  msg.pose.orientation.w = 1.0;
  for (size_t i = 0; i < cloud.size(); ++i) {
    if (!cloud_info.points[i].cluster_level_dynamic) {
      msg.points.push_back(setPoint(cloud[i]));
    }
  }
  detection_points_comp_pub_.publish(msg);

  // Dynamic points are all points that are part of a cluster.
  for (const auto& cluster : clusters) {
    // Only visualize points for clusters that are not classified as false.
    if (classifications_.count(cluster.id) &&
        classifications_.at(cluster.id) == "false") {
      continue;
    }
    visualization_msgs::Marker c_msg;
    c_msg.action = visualization_msgs::Marker::ADD;
    c_msg.ns = "detections_dynamic_points";
    c_msg.id = cluster.id;
    c_msg.header.stamp = getStamp();
    c_msg.header.frame_id = config_.global_frame_name;
    c_msg.type = visualization_msgs::Marker::SPHERE_LIST;
    c_msg.scale = setScale(config_.dynamic_point_scale);
    c_msg.color = setColor(config_.dynamic_point_color);
    c_msg.pose.orientation.w = 1.0;
    for (const auto& point_idx : cluster.points) {
      c_msg.points.push_back(setPoint(cloud[point_idx]));
    }
    detection_points_pub_.publish(c_msg);
  }
}

void MotionVisualizer::visualizeClusterDetections(const Cloud& cloud,
                                                const Clusters& clusters) {
  if (detection_cluster_pub_.getNumSubscribers() == 0u) {
    return;
  }

  visualization_msgs::Marker result;
  result.action = visualization_msgs::Marker::ADD;
  result.id = 0;
  result.header.stamp = getStamp();
  result.header.frame_id = config_.global_frame_name;
  result.type = visualization_msgs::Marker::POINTS;
  result.scale = setScale(config_.dynamic_point_scale);

  // Get all cluster points.
  int i = 0;
  for (const Cluster& cluster : clusters) {
    std_msgs::ColorRGBA color;
    if (config_.color_clusters) {
      color = setColor(color_map_.colorLookup(i));
      ++i;
    } else {
      color = setColor(config_.dynamic_point_color);
    }
    for (int index : cluster.points) {
      if (index < 0 || index >= static_cast<int>(cloud.points.size()) || 
          cloud[index].z > config_.visualization_max_z) {
        continue;
      }
      result.points.push_back(setPoint(cloud[index]));
      result.colors.push_back(color);
    }
  }

  detection_cluster_pub_.publish(result);
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
    if (!cluster.valid || cluster.points.empty()) {
      continue;
    }
    std_msgs::ColorRGBA color;
    if (config_.color_clusters) {
      color = setColor(color_map_.colorLookup(cluster.id));
    } else {
      color = setColor(config_.dynamic_point_color);
    }

    for (int index : cluster.points) {
      if (index < 0 || index >= static_cast<int>(cloud.points.size()) || 
          cloud[index].z > config_.visualization_max_z) {
        continue;
      }
      result.points.push_back(setPoint(cloud[index]));
      result.colors.push_back(color);
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
      if (i < cloud_info.points.size() && !cloud_info.points[i].object_level_dynamic) {
        geometry_msgs::Point p = setPoint(point);
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
          result_comp.points.push_back(p);
        }
      }
      ++i;
    }
  }

  if (!result.points.empty() && dynamic) {
    detection_object_pub_.publish(result);
  }
  if (!result_comp.points.empty() && comp) {
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

void MotionVisualizer::setClassifications(
    const std::unordered_map<int, std::string>& classifications) {
  classifications_ = classifications;
}

}  // namespace dynablox
