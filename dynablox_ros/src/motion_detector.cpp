#include "dynablox_ros/motion_detector.h"

#include <math.h>

#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <minkindr_conversions/kindr_tf.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>

namespace dynablox {

using Timer = voxblox::timing::Timer;

void MotionDetector::Config::checkParams() const {
  checkParamCond(!global_frame_name.empty(),
                 "'global_frame_name' may not be empty.");
  checkParamGE(num_threads, 1, "num_threads");
  checkParamGE(queue_size, 0, "queue_size");
}

void MotionDetector::Config::setupParamsAndPrinting() {
  setupParam("global_frame_name", &global_frame_name);
  setupParam("sensor_frame_name", &sensor_frame_name);
  setupParam("queue_size", &queue_size);
  setupParam("evaluate", &evaluate);
  setupParam("visualize", &visualize);
  setupParam("verbose", &verbose);
  setupParam("num_threads", &num_threads);
  setupParam("shutdown_after", &shutdown_after);

  setupParam("use_latest_transform", &use_latest_transform);
  setupParam("transform_lookup_timeout", &transform_lookup_timeout);
  setupParam("max_cluster_topics", &max_cluster_topics);
  setupParam("use_filtered_clusters", &use_filtered_clusters);
  setupParam("filtered_topic_prefix", &filtered_topic_prefix);
  setupParam("filtered_trigger_topic", &filtered_trigger_topic);
}

MotionDetector::MotionDetector(const ros::NodeHandle& nh,
                               const ros::NodeHandle& nh_private)
    : config_(
          config_utilities::getConfigFromRos<MotionDetector::Config>(nh_private)
              .checkValid()),
      nh_(nh),
      nh_private_(nh_private) {
  setupMembers();

  // Cache frequently used constants.
  voxels_per_side_ = tsdf_layer_->voxels_per_side();
  voxels_per_block_ = voxels_per_side_ * voxels_per_side_ * voxels_per_side_;

  // Advertise and subscribe to topics.
  setupRos();

  // Print current configuration of all components.
  LOG_IF(INFO, config_.verbose) << "Configuration:\n"
                                << config_utilities::Global::printAllConfigs();
}

void MotionDetector::setupMembers() {
  // Voxblox. Overwrite dependent config parts. Note that this TSDF layer is
  // shared with all other processing components and is mutable for processing.
  ros::NodeHandle nh_voxblox(nh_private_, "voxblox");
  nh_voxblox.setParam("world_frame", config_.global_frame_name);
  nh_voxblox.setParam("update_mesh_every_n_sec", 0);
  nh_voxblox.setParam("voxel_carving_enabled",
                      true);  // Integrate whole ray not only truncation band.
  nh_voxblox.setParam("allow_clear",
                      true);  // Integrate rays up to max_ray_length.
  nh_voxblox.setParam("integrator_threads", config_.num_threads);

  tsdf_server_ = std::make_shared<voxblox::TsdfServer>(nh_voxblox, nh_voxblox);
  tsdf_layer_.reset(tsdf_server_->getTsdfMapPtr()->getTsdfLayerPtr());

  // Preprocessing.
  preprocessing_ = std::make_shared<Preprocessing>(
      config_utilities::getConfigFromRos<Preprocessing::Config>(
          ros::NodeHandle(nh_private_, "preprocessing")));

  // Clustering.
  clustering_ = std::make_shared<Clustering>(
      config_utilities::getConfigFromRos<Clustering::Config>(
          ros::NodeHandle(nh_private_, "clustering")),
      tsdf_layer_);

  // Tracking.
  tracking_ = std::make_shared<Tracking>(
      config_utilities::getConfigFromRos<Tracking::Config>(
          ros::NodeHandle(nh_private_, "tracking")));

  // Ever-Free Integrator.
  ros::NodeHandle nh_ever_free(nh_private_, "ever_free_integrator");
  nh_ever_free.setParam("num_threads", config_.num_threads);
  ever_free_integrator_ = std::make_shared<EverFreeIntegrator>(
      config_utilities::getConfigFromRos<EverFreeIntegrator::Config>(
          nh_ever_free),
      tsdf_layer_);

  // Evaluation.
  if (config_.evaluate) {
    // NOTE(schmluk): These will be uninitialized if not requested, but then no
    // config files need to be set.
    evaluator_ = std::make_shared<Evaluator>(
        config_utilities::getConfigFromRos<Evaluator::Config>(
            ros::NodeHandle(nh_private_, "evaluation")));
  }

  // Visualization.
  visualizer_ = std::make_shared<MotionVisualizer>(
      ros::NodeHandle(nh_private_, "visualization"), tsdf_layer_);
}

void MotionDetector::setupRos() {
  // Always subscribe to lidar data to generate initial clusters
  lidar_pcl_sub_ = nh_.subscribe("pointcloud", config_.queue_size,
                                 &MotionDetector::pointcloudCallback, this);
  cluster_batch_pub_ =
      nh_private_.advertise<std_msgs::Header>("cluster_batch", 10);
  
  cluster_pubs_.clear();
  cluster_pubs_.reserve(config_.max_cluster_topics);
  for (int i = 0; i < config_.max_cluster_topics; ++i) {
      const std::string topic_name = "cluster_" + std::to_string(i);
      cluster_pubs_.push_back(nh_.advertise<sensor_msgs::PointCloud2>(topic_name, 10));
  }

  if (config_.use_filtered_clusters) {
    // Setup filtered cluster mode
    ROS_INFO("Using filtered cluster mode with prefix '%s'", 
             config_.filtered_topic_prefix.c_str());
    
    // Subscribe to filtered completion trigger (after classification is done)
    std::string completion_trigger_topic = config_.filtered_trigger_topic;
    size_t pos = completion_trigger_topic.find("cluster_batch");
    if (pos != std::string::npos) {
      completion_trigger_topic.replace(pos, 13, "filtered_complete"); // 13 = length of "cluster_batch"
    }
    filtered_trigger_sub_ = nh_.subscribe(completion_trigger_topic, 10,
                                        &MotionDetector::filteredTriggerCallback, this);
    ROS_INFO("Subscribing to completion trigger on '%s'", completion_trigger_topic.c_str());
    
    // Subscribe to filtered cluster topics
    filtered_cluster_subs_.reserve(config_.max_cluster_topics);
    for (int i = 0; i < config_.max_cluster_topics; ++i) {
      std::string topic_name = config_.filtered_topic_prefix + std::to_string(i);
      filtered_cluster_subs_.push_back(
        nh_.subscribe<sensor_msgs::PointCloud2>(
          topic_name, 10,
          [this, i](const sensor_msgs::PointCloud2::ConstPtr& msg) {
            filteredClusterCallback(msg, i);
          }
        )
      );
    }
  }
}

void MotionDetector::pointcloudCallback(
    const sensor_msgs::PointCloud2::Ptr& msg) {
  Timer frame_timer("frame");
  Timer detection_timer("motion_detection");

  // Lookup cloud transform T_M_S of sensor (S) to map (M).
  // If different sensor frame is required, update the message.
  Timer tf_lookup_timer("motion_detection/tf_lookup");
  const std::string sensor_frame_name = config_.sensor_frame_name.empty()
                                            ? msg->header.frame_id
                                            : config_.sensor_frame_name;

  tf::StampedTransform T_M_S;
  if (!lookupTransform(config_.global_frame_name, sensor_frame_name,
                       msg->header.stamp.toNSec(), T_M_S)) {
    // Getting transform failed, need to skip.
    return;
  }
  tf_lookup_timer.Stop();

  // Preprocessing.
  Timer preprocessing_timer("motion_detection/preprocessing");
  frame_counter_++;
  CloudInfo cloud_info;
  Cloud cloud;
  preprocessing_->processPointcloud(msg, T_M_S, cloud, cloud_info);
  preprocessing_timer.Stop();

  // Build a mapping of all blocks to voxels to points for the scan.
  Timer setup_timer("motion_detection/indexing_setup");
  BlockToPointMap point_map;
  std::vector<voxblox::VoxelKey> occupied_ever_free_voxel_indices;
  setUpPointMap(cloud, point_map, occupied_ever_free_voxel_indices, cloud_info);
  setup_timer.Stop();

  // Clustering.
  Timer clustering_timer("motion_detection/clustering");
  Clusters clusters = clustering_->performClustering(
      point_map, occupied_ever_free_voxel_indices, frame_counter_, cloud,
      cloud_info);
  clustering_timer.Stop();

  if (!config_.use_filtered_clusters) {
    // Normal mode: do tracking, evaluation, and visualization here
    // Tracking.
    Timer tracking_timer("motion_detection/tracking");
    tracking_->track(cloud, clusters, cloud_info);
    tracking_timer.Stop();

    // Evaluation if requested.
    if (config_.evaluate) {
      Timer eval_timer("evaluation");
      evaluator_->evaluateFrame(cloud, cloud_info, clusters);
      eval_timer.Stop();
      if (config_.shutdown_after > 0 &&
          evaluator_->getNumberOfEvaluatedFrames() >= config_.shutdown_after) {
        LOG(INFO) << "Evaluated " << config_.shutdown_after
                  << " frames, shutting down";
        ros::shutdown();
      }
    }

    // Visualization if requested.
    if (config_.visualize) {
      Timer vis_timer("visualizations");
      visualizer_->visualizeAll(cloud, cloud_info, clusters);
      vis_timer.Stop();
    }
  } else {
    // Filtered mode: only do basic visualization (lidar points, static elements)
    // The dynamic cluster visualization will be done in processFilteredClusters
    if (config_.visualize) {
      Timer vis_timer("visualizations/basic");
      // Create empty clusters for basic visualization (only lidar points, etc.)
      Clusters empty_clusters;
      visualizer_->visualizeAll(cloud, cloud_info, empty_clusters);
      vis_timer.Stop();
    }
  }

  // Always update ever-free and TSDF (needed for both modes)
  // Integrate ever-free information.
  Timer update_ever_free_timer("motion_detection/update_ever_free");
  ever_free_integrator_->updateEverFreeVoxels(frame_counter_);
  update_ever_free_timer.Stop();

  // Integrate the pointcloud into the voxblox TSDF map.
  Timer tsdf_timer("motion_detection/tsdf_integration");
  voxblox::Transformation T_G_C;
  tf::transformTFToKindr(T_M_S, &T_G_C);
  tsdf_server_->processPointCloudMessageAndInsert(msg, T_G_C, false);
  tsdf_timer.Stop();
  detection_timer.Stop();

  size_t num_clusters_to_publish = clusters.size();
  if (clusters.size() > static_cast<size_t>(config_.max_cluster_topics)) {
    ROS_WARN_THROTTLE(5.0, "Number of detected clusters (%zu) exceeds 'max_cluster_topics' (%d). "
                           "Only publishing the first %d clusters.",
                           clusters.size(), config_.max_cluster_topics, config_.max_cluster_topics);
    num_clusters_to_publish = config_.max_cluster_topics;
  }

  for (size_t i = 0; i < num_clusters_to_publish; ++i) {
    const auto& cluster = clusters[i];
    
    if (cluster.points.empty()) {
      continue;
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    for (const auto& point_idx : cluster.points) {
      pcl::PointXYZI point;
      point.x = cloud[point_idx].x;
      point.y = cloud[point_idx].y;
      point.z = cloud[point_idx].z;
      point.intensity = cloud[point_idx].intensity;
      
      cluster_cloud->points.push_back(point);
    }
    
    if (!cluster_cloud->empty()) {
      cluster_cloud->width = cluster_cloud->points.size();
      cluster_cloud->height = 1;
      cluster_cloud->is_dense = true;
      
      sensor_msgs::PointCloud2 output_msg;
      pcl::toROSMsg(*cluster_cloud, output_msg);
      output_msg.header = msg->header; 
      
      cluster_pubs_[i].publish(output_msg);
    }
  }

  std_msgs::Header batch_manifest_msg;
  batch_manifest_msg.stamp = msg->header.stamp;
  batch_manifest_msg.frame_id = msg->header.frame_id;
  batch_manifest_msg.seq = num_clusters_to_publish;
  cluster_batch_pub_.publish(batch_manifest_msg);

  frame_timer.Stop();
}

bool MotionDetector::lookupTransform(const std::string& target_frame,
                                     const std::string& source_frame,
                                     uint64_t timestamp,
                                     tf::StampedTransform& result) const {
  ros::Time timestamp_ros;
  timestamp_ros.fromNSec(timestamp);

  try {
    if (tf_listener_.waitForTransform(target_frame, source_frame, timestamp_ros, 
                                     ros::Duration(config_.transform_lookup_timeout))) {
      tf_listener_.lookupTransform(target_frame, source_frame, timestamp_ros, result);
      return true;
    } else if (config_.use_latest_transform) {
      if (config_.verbose) {
        ROS_WARN_STREAM("Could not get transform at exact timestamp, using latest available transform");
      }
      tf_listener_.lookupTransform(target_frame, source_frame, ros::Time(0), result);
      return true;
    } else {
      LOG(WARNING) << "Could not get sensor transform within timeout, skipping pointcloud";
      return false;
    }
  } catch (tf::TransformException& ex) {
    LOG(WARNING) << "Could not get sensor transform, skipping pointcloud: " << ex.what();
    return false;
  }
}

void MotionDetector::setUpPointMap(
    const Cloud& cloud, BlockToPointMap& point_map,
    std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
    CloudInfo& cloud_info) const {
  // Identifies for any LiDAR point the block it falls in and constructs the
  // hash-map block2points_map mapping each block to the LiDAR points that
  // fall into the block.
  const voxblox::HierarchicalIndexIntMap block2points_map =
      buildBlockToPointsMap(cloud);

  // Builds the voxel2point-map in parallel blockwise.
  std::vector<BlockIndex> block_indices(block2points_map.size());
  size_t i = 0;
  for (const auto& block : block2points_map) {
    block_indices[i] = block.first;
    ++i;
  }
  IndexGetter<BlockIndex> index_getter(block_indices);
  std::vector<std::future<void>> threads;
  std::mutex aggregate_results_mutex;
  for (int i = 0; i < config_.num_threads; ++i) {
    threads.emplace_back(std::async(std::launch::async, [&]() {
      // Data to store results.
      BlockIndex block_index;
      std::vector<voxblox::VoxelKey> local_occupied_indices;
      BlockToPointMap local_point_map;

      // Process until no more blocks.
      while (index_getter.getNextIndex(&block_index)) {
        VoxelToPointMap result;
        this->blockwiseBuildPointMap(cloud, block_index,
                                     block2points_map.at(block_index), result,
                                     local_occupied_indices, cloud_info);
        local_point_map.insert(std::pair(block_index, result));
      }

      // After processing is done add data to the output map.
      std::lock_guard<std::mutex> lock(aggregate_results_mutex);
      occupied_ever_free_voxel_indices.insert(
          occupied_ever_free_voxel_indices.end(),
          local_occupied_indices.begin(), local_occupied_indices.end());
      point_map.merge(local_point_map);
    }));
  }

  for (auto& thread : threads) {
    thread.get();
  }
}

voxblox::HierarchicalIndexIntMap MotionDetector::buildBlockToPointsMap(
    const Cloud& cloud) const {
  voxblox::HierarchicalIndexIntMap result;

  int i = 0;
  for (const Point& point : cloud) {
    voxblox::Point coord(point.x, point.y, point.z);
    const BlockIndex blockindex =
        tsdf_layer_->computeBlockIndexFromCoordinates(coord);
    result[blockindex].push_back(i);
    i++;
  }
  return result;
}

void MotionDetector::blockwiseBuildPointMap(
    const Cloud& cloud, const BlockIndex& block_index,
    const voxblox::AlignedVector<size_t>& points_in_block,
    VoxelToPointMap& voxel_map,
    std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
    CloudInfo& cloud_info) const {
  // Get the block.
  TsdfBlock::Ptr tsdf_block = tsdf_layer_->getBlockPtrByIndex(block_index);
  if (!tsdf_block) {
    return;
  }

  // Create a mapping of each voxel index to the points it contains.
  for (size_t i : points_in_block) {
    const Point& point = cloud[i];
    const voxblox::Point coords(point.x, point.y, point.z);
    const VoxelIndex voxel_index =
        tsdf_block->computeVoxelIndexFromCoordinates(coords);
    if (!tsdf_block->isValidVoxelIndex(voxel_index)) {
      continue;
    }
    voxel_map[voxel_index].push_back(i);

    // EverFree detection flag at the same time, since we anyways lookup
    // voxels.
    if (tsdf_block->getVoxelByVoxelIndex(voxel_index).ever_free) {
      cloud_info.points.at(i).ever_free_level_dynamic = true;
    }
  }

  // Update the voxel status of the currently occupied voxels.
  for (const auto& voxel_points_pair : voxel_map) {
    TsdfVoxel& tsdf_voxel =
        tsdf_block->getVoxelByVoxelIndex(voxel_points_pair.first);
    tsdf_voxel.last_lidar_occupied = frame_counter_;

    // This voxel attribute is used in the voxel clustering method: it
    // signalizes that a currently occupied voxel has not yet been clustered
    tsdf_voxel.clustering_processed = false;

    // The set of occupied_ever_free_voxel_indices allows for fast access of
    // the seed voxels in the voxel clustering
    if (tsdf_voxel.ever_free) {
      occupied_ever_free_voxel_indices.push_back(
          std::make_pair(block_index, voxel_points_pair.first));
    }
  }
}

void MotionDetector::filteredTriggerCallback(const std_msgs::Header::ConstPtr& msg) {
   std::lock_guard<std::mutex> lock(filtered_buffer_lock_);
   
   ros::Time stamp = msg->stamp;
   int expected_count = msg->seq;
   
   // Store expected cluster count for this timestamp
   filtered_trigger_buffer_[stamp] = expected_count;
   
   ROS_DEBUG("Classification complete for stamp %f with %d filtered clusters", stamp.toSec(), expected_count);
   
   // Check if we already have all expected clusters buffered.
   auto clusters_it = filtered_cluster_buffer_.find(stamp);
   if (clusters_it != filtered_cluster_buffer_.end()) {
     size_t have = clusters_it->second.size();
     if (have >= static_cast<size_t>(expected_count)) {
       ROS_DEBUG("Processing %zu buffered filtered clusters for stamp %f (expected %d)", have, stamp.toSec(), expected_count);
       processFilteredClusters(clusters_it->second, stamp);
       filtered_cluster_buffer_.erase(clusters_it);
       filtered_trigger_buffer_.erase(stamp);
     } else {
       ROS_DEBUG("Waiting for remaining %zu clusters for stamp %f", static_cast<size_t>(expected_count) - have, stamp.toSec());
     }
   } else {
     ROS_DEBUG("No clusters buffered yet for stamp %f, waiting for them to arrive", stamp.toSec());
   }
   
   pruneInflightBuffers();
 }

void MotionDetector::filteredClusterCallback(const sensor_msgs::PointCloud2::ConstPtr& msg, int cluster_index) {
   std::lock_guard<std::mutex> lock(filtered_buffer_lock_);
   
   ros::Time stamp = msg->header.stamp;
   
   ROS_DEBUG("Received filtered cluster %d for stamp %f", cluster_index, stamp.toSec());
   
   // Buffer this cluster
   auto& cluster_map = filtered_cluster_buffer_[stamp];
   cluster_map[cluster_index] = msg;
   
   // Check if we already know how many clusters to expect for this stamp
   auto trig_it = filtered_trigger_buffer_.find(stamp);
   if (trig_it != filtered_trigger_buffer_.end()) {
     int expected_count = trig_it->second;
     if (cluster_map.size() >= static_cast<size_t>(expected_count)) {
       ROS_DEBUG("All %d filtered clusters received for stamp %f. Processing now.", expected_count, stamp.toSec());
       processFilteredClusters(cluster_map, stamp);
       filtered_cluster_buffer_.erase(stamp);
       filtered_trigger_buffer_.erase(trig_it);
     }
   }
   
   pruneInflightBuffers();
 }

void MotionDetector::processFilteredClusters(
    const std::unordered_map<int, sensor_msgs::PointCloud2::ConstPtr>& cluster_msgs,
    const ros::Time& stamp) {
  
  Timer frame_timer("filtered_frame");
  
  // Convert filtered clusters to dynablox format
  Cloud combined_cloud;
  CloudInfo cloud_info;
  cloud_info.timestamp = stamp.toNSec();
  cloud_info.sensor_position.x = 0.f;
  cloud_info.sensor_position.y = 0.f;
  cloud_info.sensor_position.z = 0.f;
  Clusters clusters;
  
  // Process each cluster message
  for (const auto& pair : cluster_msgs) {
    const auto& msg = pair.second;
    
    // Convert PointCloud2 to PCL
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    
    if (pcl_cloud.empty()) {
      continue;
    }
    
    // Create cluster object
    Cluster cluster;
    cluster.id = next_cluster_id_++;
    cluster.valid = true;  // These are already filtered as valid
    cluster.track_length = 0;  // Will be updated by tracking
    
    // Add points to combined cloud and set up cluster
    Point min_point, max_point;
    bool first_point = true;
    
    for (const auto& pcl_point : pcl_cloud.points) {
      // Add to combined cloud
      Point point;
      point.x = pcl_point.x;
      point.y = pcl_point.y;
      point.z = pcl_point.z;
      point.intensity = pcl_point.intensity;
      combined_cloud.push_back(point);
      
      // Add point info
      PointInfo point_info;
      point_info.cluster_level_dynamic = true;  // These are already classified as dynamic
      point_info.ever_free_level_dynamic = false;
      point_info.object_level_dynamic = false;  // Will be set by tracking
      point_info.ground_truth_dynamic = false;
      point_info.ready_for_evaluation = true;
      cloud_info.points.push_back(point_info);
      
      // Add to cluster
      cluster.points.push_back(combined_cloud.size() - 1);
      
      // Update bounding box
      if (first_point) {
        min_point = max_point = point;
        first_point = false;
      } else {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);
        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
      }
    }
    
    // Set bounding box
    cluster.aabb.min_corner = min_point;
    cluster.aabb.max_corner = max_point;
    
    clusters.push_back(cluster);
  }
  
  if (clusters.empty()) {
    ROS_INFO("No valid filtered clusters to process for stamp %f", stamp.toSec());
    return;
  }
  
  ROS_DEBUG("Processing %zu filtered clusters with %zu total points", 
           clusters.size(), combined_cloud.size());
  
  // Run tracking
  Timer tracking_timer("filtered_frame/tracking");
  tracking_->track(combined_cloud, clusters, cloud_info);
  tracking_timer.Stop();
  
  // Run visualization if enabled
  if (config_.visualize) {
    Timer vis_timer("filtered_frame/visualization");
    ROS_DEBUG("Visualizing %zu filtered clusters with %zu total points", 
             clusters.size(), combined_cloud.size());
    visualizer_->visualizeAll(combined_cloud, cloud_info, clusters);
    vis_timer.Stop();
  }
  
  // Evaluation if requested
  if (config_.evaluate) {
    Timer eval_timer("filtered_frame/evaluation");
    evaluator_->evaluateFrame(combined_cloud, cloud_info, clusters);
    eval_timer.Stop();
  }
  
  frame_timer.Stop();
}

void MotionDetector::pruneInflightBuffers() {
  // If too many trigger stamps are buffered, drop oldest until within limit.
  while (filtered_trigger_buffer_.size() > kMaxInFlight) {
    dropFrame(filtered_trigger_buffer_.begin()->first, "trigger overflow");
  }

  // Same for cluster sets.
  while (filtered_cluster_buffer_.size() > kMaxInFlight) {
    dropFrame(filtered_cluster_buffer_.begin()->first, "cluster overflow");
  }
}

void MotionDetector::dropFrame(const ros::Time& stamp, const std::string& reason) {
  size_t have = 0;
  auto cl_it = filtered_cluster_buffer_.find(stamp);
  if (cl_it != filtered_cluster_buffer_.end()) {
    have = cl_it->second.size();
    filtered_cluster_buffer_.erase(cl_it);
  }

  int expected = 0;
  auto tr_it = filtered_trigger_buffer_.find(stamp);
  if (tr_it != filtered_trigger_buffer_.end()) {
    expected = tr_it->second;
    filtered_trigger_buffer_.erase(tr_it);
  }

  ROS_WARN("Dropping incomplete frame %f — %s (expected %d, received %zu)",
           stamp.toSec(), reason.c_str(), expected, have);
}

}  // namespace dynablox
