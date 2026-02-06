#include "dynablox_ros/motion_detector.h"

#include <math.h>

#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <minkindr_conversions/kindr_tf.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree_flann.h>
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
  setupParam("use_cluster_array", &use_cluster_array);
  setupParam("cluster_array_topic", &cluster_array_topic);
  setupParam("use_filtered_clusters", &use_filtered_clusters);
  setupParam("filtered_cluster_array_topic", &filtered_cluster_array_topic);
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

MotionDetector::~MotionDetector() {
  // Save latencies on shutdown
  saveLatenciesToFile();
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
    
    //publish timestamped output directory for other nodes
    nh_private_.setParam("evaluation/actual_output_directory", evaluator_->getOutputDirectory());
  }

  // Visualization.
  visualizer_ = std::make_shared<MotionVisualizer>(
      ros::NodeHandle(nh_private_, "visualization"), tsdf_layer_);
}

void MotionDetector::setupRos() {
  lidar_pcl_sub_ = nh_.subscribe("pointcloud", config_.queue_size,
                                 &MotionDetector::pointcloudCallback, this);
  
  // Setup cluster publishing based on mode
  if (config_.use_cluster_array) {
    ROS_INFO("Using ClusterArray mode on topic '%s' (optimized for latency)", 
             config_.cluster_array_topic.c_str());
    cluster_array_pub_ = nh_.advertise<dynablox_ros::ClusterArray>(
        config_.cluster_array_topic, 100);
  } else {
    ROS_INFO("Using legacy multi-topic mode (32 separate topics)");
    cluster_batch_pub_ =
        nh_private_.advertise<std_msgs::Header>("cluster_batch", 10);
    
    cluster_pubs_.clear();
    cluster_pubs_.reserve(config_.max_cluster_topics);
    for (int i = 0; i < config_.max_cluster_topics; ++i) {
        const std::string topic_name = "cluster_" + std::to_string(i);
        cluster_pubs_.push_back(nh_.advertise<sensor_msgs::PointCloud2>(topic_name, 10));
    }
  }

  // FilteredClusterArray mode - subscribe to classified human clusters
  if (config_.use_filtered_clusters) {
    ROS_INFO("🚀 FilteredClusterArray mode enabled on '%s' (TCP_NODELAY enabled)", 
             config_.filtered_cluster_array_topic.c_str());
    ros::TransportHints hints;
    hints.tcpNoDelay(true);
    filtered_cluster_array_sub_ = nh_.subscribe(
      config_.filtered_cluster_array_topic, 100,
      &MotionDetector::filteredClusterArrayCallback, this, hints);
  }
}

void MotionDetector::pointcloudCallback(
    const sensor_msgs::PointCloud2::Ptr& msg) {
  Timer frame_timer("frame");
  Timer detection_timer("motion_detection");

  // UNIFIED TIMING: Record when PointCloud2 arrived
  ros::WallTime t0_pointcloud_received = ros::WallTime::now();

  // Lookup cloud transform T_M_S of sensor (S) to map (M).
  // If different sensor frame is required, update the message.
  Timer tf_lookup_timer("motion_detection/tf_lookup");
  const std::string sensor_frame_name = config_.sensor_frame_name.empty()
                                            ? msg->header.frame_id
                                            : config_.sensor_frame_name;

  tf::StampedTransform T_M_S;
  if (!lookupTransform(config_.global_frame_name, sensor_frame_name,
                       msg->header.stamp.toNSec(), T_M_S)) {
    // Getting transform failed - log warning but try to continue with identity or previous transform
    if (have_last_good_transform_) {
      ROS_WARN_STREAM_THROTTLE(1.0, "TF lookup failed for stamp " << msg->header.stamp.toSec() 
                                    << ", using last known transform to avoid dropping frame");
      T_M_S = last_good_T_M_S_;
    } else {
      // Last resort: use identity transform (assumes sensor and map are aligned)
      ROS_ERROR_STREAM_THROTTLE(1.0, "TF lookup failed for stamp " << msg->header.stamp.toSec() 
                                     << " and no previous transform available. Using identity transform.");
      T_M_S.setIdentity();
      T_M_S.frame_id_ = config_.global_frame_name;
      T_M_S.child_frame_id_ = sensor_frame_name;
      T_M_S.stamp_ = msg->header.stamp;
    }
    // Continue processing instead of skipping frame
  }
  tf_lookup_timer.Stop();

  // Preprocessing.
  Timer preprocessing_timer("motion_detection/preprocessing");
  frame_counter_++;
  CloudInfo cloud_info;
  Cloud cloud;
  preprocessing_->processPointcloud(msg, T_M_S, cloud, cloud_info);
  preprocessing_timer.Stop();
  
  // UNIFIED TIMING: Record preprocessing done
  ros::WallTime t1_preprocessing_done = ros::WallTime::now();

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
  
  // UNIFIED TIMING: Record clustering done
  ros::WallTime t2_clustering_done = ros::WallTime::now();

  if (!config_.use_filtered_clusters) {
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
        saveLatenciesToFile();
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
    if (config_.visualize) {
      Timer vis_timer("visualizations/basic");
      Clusters empty_clusters;
      visualizer_->visualizeAll(cloud, cloud_info, empty_clusters);
      vis_timer.Stop();
    }
  }

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

  //store the raw pointcloud for evaluation
  // Buffer is pruned dynamically in FilteredClusterArrayCallback when frames are consumed
  if (config_.use_filtered_clusters) {
    std::lock_guard<std::mutex> lock(raw_buffer_lock_);
    raw_cloud_buffer_[msg->header.stamp] = std::make_pair(cloud, cloud_info);
  }

  size_t num_clusters_to_publish = clusters.size();
  if (clusters.size() > static_cast<size_t>(config_.max_cluster_topics)) {
    ROS_WARN_THROTTLE(5.0, "Number of detected clusters (%zu) exceeds 'max_cluster_topics' (%d). "
                           "Only publishing the first %d clusters.",
                           clusters.size(), config_.max_cluster_topics, config_.max_cluster_topics);
    num_clusters_to_publish = config_.max_cluster_topics;
  }

  if (config_.use_cluster_array) {
    // NEW: Optimized single-message publishing
    dynablox_ros::ClusterArray array_msg;
    array_msg.header = msg->header;
    array_msg.num_clusters = 0;  // Will count valid clusters
    
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
        
        sensor_msgs::PointCloud2 cluster_msg;
        pcl::toROSMsg(*cluster_cloud, cluster_msg);
        cluster_msg.header = msg->header;
        
        // Add to array
        array_msg.clusters.push_back(cluster_msg);
        array_msg.cluster_ids.push_back(i);
        
        // Calculate centroid
        geometry_msgs::Point centroid;
        centroid.x = 0.0;
        centroid.y = 0.0;
        centroid.z = 0.0;
        for (const auto& point_idx : cluster.points) {
          centroid.x += cloud[point_idx].x;
          centroid.y += cloud[point_idx].y;
          centroid.z += cloud[point_idx].z;
        }
        centroid.x /= cluster.points.size();
        centroid.y /= cluster.points.size();
        centroid.z /= cluster.points.size();
        
        array_msg.centroids.push_back(centroid);
        array_msg.cluster_sizes.push_back(cluster.points.size());
        array_msg.num_clusters++;
      }
    }
    
    size_t total_bytes = 0;
    for (const auto& cluster_msg : array_msg.clusters) {
      total_bytes += cluster_msg.data.size();
    }
    
    // Record wall-clock time for component latency tracking
    if (config_.use_filtered_clusters) {
      std::lock_guard<std::mutex> lock(latency_mutex_);
      ComponentTiming timing;
      timing.pointcloud_received = t0_pointcloud_received;
      timing.preprocessing_done = t1_preprocessing_done;
      timing.clustering_done = t2_clustering_done;
      timing.cluster_array_published = ros::WallTime::now();
      
      // Store sub-component durations
      timing.dynablox_preprocessing_ms = (t1_preprocessing_done - t0_pointcloud_received).toSec() * 1000.0;
      timing.dynablox_clustering_ms = (t2_clustering_done - t1_preprocessing_done).toSec() * 1000.0;
      
      timing_data_[msg->header.stamp] = timing;
    }
    
    // Single publish operation!
    cluster_array_pub_.publish(array_msg);
    
    // Log message size for debugging communication latency
    size_t total_points = 0;
    for (const auto& cluster_msg : array_msg.clusters) {
      total_points += cluster_msg.width * cluster_msg.height;
    }
    ROS_INFO_THROTTLE(1.0, "ClusterArray published: %u clusters, %zu points | Size: %.2f KB", 
                      array_msg.num_clusters, total_points, total_bytes / 1024.0);
    
  } else {
    // LEGACY: Multi-topic publishing (backwards compatibility)
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
  }

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
      last_good_T_M_S_ = result;
      have_last_good_transform_ = true;
      return true;
    }

    if (config_.use_latest_transform) {
      if (config_.verbose) {
        ROS_WARN_STREAM("Could not get transform at exact timestamp, using latest available transform");
      }
      tf_listener_.lookupTransform(target_frame, source_frame, ros::Time(0), result);
      last_good_T_M_S_ = result;
      have_last_good_transform_ = true;
      return true;
    }

    if (config_.use_previous_transform_on_fail && have_last_good_transform_) {
      if (config_.verbose) {
        ROS_WARN_STREAM("TF lookup failed, reusing previous transform");
      }
      result = last_good_T_M_S_;
      return true;
    }

    // Even if use_previous_transform_on_fail is false, try to use previous transform as last resort
    if (have_last_good_transform_) {
      LOG(WARNING) << "Could not get sensor transform within timeout, reusing previous transform to avoid dropping frame";
      result = last_good_T_M_S_;
      return true;
    }
    
    LOG(WARNING) << "Could not get sensor transform within timeout and no previous transform available";
    return false;
  } catch (tf::TransformException& ex) {
    if (config_.use_previous_transform_on_fail && have_last_good_transform_) {
      if (config_.verbose) {
        ROS_WARN_STREAM("TF exception: " << ex.what() << "; reusing previous transform");
      }
      result = last_good_T_M_S_;
      return true;
    }
    
    // Even if use_previous_transform_on_fail is false, try to use previous transform as last resort
    if (have_last_good_transform_) {
      LOG(WARNING) << "TF exception: " << ex.what() << "; reusing previous transform to avoid dropping frame";
      result = last_good_T_M_S_;
      return true;
    }
    
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

void MotionDetector::filteredClusterArrayCallback(const dynablox_ros::FilteredClusterArray::ConstPtr& msg) {
  ros::Time stamp = msg->header.stamp;
  uint32_t num_clusters = msg->num_clusters;
  
  // UNIFIED TIMING: Record when FilteredClusterArray arrived
  ros::WallTime t4_filtered_received = ros::WallTime::now();
  
  // Calculate component latencies using wall-clock time
    double total_latency_ms = 0.0;
    double pointosr_ms = 0.0;
    double communication_ms = 0.0;
    {
      std::lock_guard<std::mutex> lock(latency_mutex_);
      auto it = timing_data_.find(stamp);
      if (it != timing_data_.end()) {
        // Complete the timing record
        it->second.filtered_array_received = t4_filtered_received;
        it->second.processing_done = ros::WallTime::now();
        it->second.pointosr_processing_ms = msg->processing_time_ms;
        
        // Total pipeline latency (wall-clock)
        total_latency_ms = (t4_filtered_received - it->second.cluster_array_published).toSec() * 1000.0;
        
        // PointOSR processing time from message
        pointosr_ms = msg->processing_time_ms;
        
        // Communication overhead = Total - PointOSR processing
        communication_ms = total_latency_ms - pointosr_ms;
        
        total_pipeline_latencies_.push_back(total_latency_ms);
        pointosr_latencies_.push_back(pointosr_ms);
        communication_latencies_.push_back(communication_ms);
        
        // Save for unified logging
        completed_timings_.push_back(it->second);
        timing_data_.erase(it);
      }
    }
  
  // Log message size for debugging
  size_t filtered_points = 0;
  size_t filtered_bytes = 0;
  for (const auto& cluster_msg : msg->clusters) {
    filtered_points += cluster_msg.width * cluster_msg.height;
    filtered_bytes += cluster_msg.data.size();
  }
  
  if (total_latency_ms > 0.0) {
    ROS_INFO_STREAM_THROTTLE(1.0, "FilteredClusterArray received: " << num_clusters 
                                   << " clusters, " << filtered_points << " points, "
                                   << filtered_bytes << " bytes (" << (filtered_bytes/1024.0) << " KB) | "
                                   << "Pipeline: " << total_latency_ms << " ms "
                                   << "(PointOSR: " << pointosr_ms << " ms, "
                                   << "Comm: " << communication_ms << " ms)");
  }
  
  ROS_DEBUG("Received FilteredClusterArray for stamp %f with %u clusters (total: %.1f ms, pointosr: %.1f ms, comm: %.1f ms)", 
            stamp.toSec(), num_clusters, total_latency_ms, pointosr_ms, communication_ms);
  
  if (num_clusters == 0) {
    ROS_DEBUG("FilteredClusterArray for stamp %f has 0 clusters — skipping processing.", stamp.toSec());
    return;
  }
  
  // Convert FilteredClusterArray to cluster map
  std::unordered_map<int, sensor_msgs::PointCloud2::ConstPtr> cluster_map;
  for (size_t i = 0; i < msg->clusters.size() && i < num_clusters; ++i) {
    auto pc_ptr = boost::make_shared<sensor_msgs::PointCloud2>(msg->clusters[i]);
    cluster_map[static_cast<int>(i)] = pc_ptr;
  }
  
  ROS_DEBUG("FilteredClusterArray: Processing %zu clusters for stamp %f", cluster_map.size(), stamp.toSec());
  
  // Process the filtered clusters directly (no buffering/synchronization needed!)
  processFilteredClusters(cluster_map, stamp);
}

void MotionDetector::processFilteredClusters(
    const std::unordered_map<int, sensor_msgs::PointCloud2::ConstPtr>& cluster_msgs,
    const ros::Time& stamp) {
  
  Timer frame_timer("filtered_frame");
  
  Cloud combined_cloud;
  CloudInfo cloud_info;
  cloud_info.timestamp = stamp.toNSec();
  cloud_info.sensor_position.x = 0.f;
  cloud_info.sensor_position.y = 0.f;
  cloud_info.sensor_position.z = 0.f;
  Clusters clusters;
  
  for (const auto& pair : cluster_msgs) {
    const auto& msg = pair.second;
    
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    
    if (pcl_cloud.empty()) {
      continue;
    }
    
    Cluster cluster;
    cluster.id = next_cluster_id_++;
    cluster.valid = true;
    cluster.track_length = 0;
    
    Point min_point, max_point;
    bool first_point = true;
    
    for (const auto& pcl_point : pcl_cloud.points) {
      Point point;
      point.x = pcl_point.x;
      point.y = pcl_point.y;
      point.z = pcl_point.z;
      point.intensity = pcl_point.intensity;
      combined_cloud.push_back(point);
      
      PointInfo point_info;
      point_info.cluster_level_dynamic = true;
      point_info.ever_free_level_dynamic = false;
      point_info.object_level_dynamic = false;
      point_info.ground_truth_dynamic = false;
      point_info.ready_for_evaluation = true;
      cloud_info.points.push_back(point_info);
      
      cluster.points.push_back(combined_cloud.size() - 1);
      
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
    
    cluster.aabb.min_corner = min_point;
    cluster.aabb.max_corner = max_point;
    
    clusters.push_back(cluster);
  }
  
  if (clusters.empty()) {
    ROS_INFO("No valid filtered clusters for stamp %f — continuing with evaluation on raw cloud.",
             stamp.toSec());
  }
  
  ROS_DEBUG("Processing %zu filtered clusters with %zu total points", 
           clusters.size(), combined_cloud.size());
  
  //retrieve the raw pointcloud for evaluation
  Cloud raw_cloud;
  CloudInfo raw_info;
  bool have_raw_cloud = false;
  {
    std::lock_guard<std::mutex> lock(raw_buffer_lock_);
    auto it = raw_cloud_buffer_.find(stamp);
    if (it != raw_cloud_buffer_.end()) {
      raw_cloud = it->second.first;
      raw_info = it->second.second;
      raw_cloud_buffer_.erase(it);
      have_raw_cloud = true;
      
      // Trigger-based pruning: remove all frames older than the one we just processed
      // These frames will never be used since classification has moved past them
      auto prune_it = raw_cloud_buffer_.begin();
      while (prune_it != raw_cloud_buffer_.end() && prune_it->first < stamp) {
        prune_it = raw_cloud_buffer_.erase(prune_it);
      }
    }
  }

  if (!have_raw_cloud) {
    ROS_WARN_THROTTLE(10.0, "No raw cloud available for stamp %f – skipping evaluation", stamp.toSec());
  }

  if (have_raw_cloud) {
    cloud_info.sensor_position = raw_info.sensor_position;
  }

  //tracking
  Timer tracking_timer("filtered_frame/tracking");
  tracking_->track(combined_cloud, clusters, cloud_info);
  tracking_timer.Stop();

  //propagate cluster and object level dynamic flags to the raw cloud
  if (!raw_cloud.empty()) {
    pcl::KdTreeFLANN<Point> kdtree;
    kdtree.setInputCloud(raw_cloud.makeShared());

    const float radius_sq = 1e-4;  //cm^2 tolerance

    for (size_t c_idx = 0; c_idx < clusters.size(); ++c_idx) {
      const Cluster& filtered_cluster = clusters[c_idx];
      for (int idx_filtered : filtered_cluster.points) {
        const Point& p = combined_cloud[idx_filtered];
        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);
        if (kdtree.nearestKSearch(p, 1, nn_indices, nn_dists) > 0 &&
            nn_dists[0] < radius_sq) {
          int raw_idx = nn_indices[0];
          if (raw_idx >= 0 && raw_idx < static_cast<int>(raw_info.points.size())) {
            raw_info.points[raw_idx].cluster_level_dynamic = true;
            if (cloud_info.points[idx_filtered].object_level_dynamic) {
              raw_info.points[raw_idx].object_level_dynamic = true;
            }
          }
        }
      }
    }
  } else {
    ROS_WARN_THROTTLE(10.0, "Raw cloud is empty for stamp %f - skipping KD-tree propagation", stamp.toSec());
  }

  //evaluation
  if (config_.evaluate) {
    Timer eval_timer("filtered_frame/evaluation_raw");
    Clusters empty_clusters;
    evaluator_->evaluateFrame(raw_cloud, raw_info, empty_clusters);
    eval_timer.Stop();
  }

  // Run visualization if enabled
  if (config_.visualize) {
    Timer vis_timer("filtered_frame/visualization");
    ROS_DEBUG("Visualizing %zu filtered clusters with %zu total points", 
             clusters.size(), combined_cloud.size());
    visualizer_->visualizeAll(combined_cloud, cloud_info, clusters);
    vis_timer.Stop();
  }
  
  frame_timer.Stop();
}

void MotionDetector::saveLatenciesToFile() const{
  if (!config_.evaluate || !evaluator_) {
    return;
  }
  
  if (total_pipeline_latencies_.empty()) {
    LOG(INFO) << "No latency data to save.";
    return;
  }
  
  //get output directory from evaluator
  std::string output_dir = evaluator_->getOutputDirectory();
  if (output_dir.empty()) {
    LOG(WARNING) << "Could not get output directory, skipping latency file.";
    return;
  }
  
  std::string output_file = output_dir + "/component_latencies.txt";
  
  std::lock_guard<std::mutex> lock(latency_mutex_);
  
  // Helper to calculate stats
  auto calc_stats = [](const std::vector<double>& data, double& mean, double& stddev, double& min_val, double& max_val) {
    if (data.empty()) return;
    double sum = 0.0;
    min_val = data[0];
    max_val = data[0];
    for (double val : data) {
      sum += val;
      if (val < min_val) min_val = val;
      if (val > max_val) max_val = val;
    }
    mean = sum / data.size();
    double variance_sum = 0.0;
    for (double val : data) {
      double diff = val - mean;
      variance_sum += diff * diff;
    }
    stddev = std::sqrt(variance_sum / data.size());
  };
  
  double total_mean, total_std, total_min, total_max;
  double pointosr_mean, pointosr_std, pointosr_min, pointosr_max;
  double comm_mean, comm_std, comm_min, comm_max;
  
  calc_stats(total_pipeline_latencies_, total_mean, total_std, total_min, total_max);
  calc_stats(pointosr_latencies_, pointosr_mean, pointosr_std, pointosr_min, pointosr_max);
  calc_stats(communication_latencies_, comm_mean, comm_std, comm_min, comm_max);
  
  std::ofstream file(output_file);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open latency file: " << output_file;
    return;
  }
  
  file << "=================================================================\n";
  file << "COMPONENT LATENCY BREAKDOWN (Wall-Clock Time)\n";
  file << "=================================================================\n\n";
  file << "PURPOSE: Identify bottlenecks in the processing pipeline\n\n";
  file << "COMPONENTS:\n";
  file << "  1. PointOSR Processing: Actual classification time (from message)\n";
  file << "  2. Communication: ROS message passing overhead\n";
  file << "  3. Total Pipeline: End-to-end latency (measured)\n\n";
  file << "NOTE: For Dynablox processing times, see timings.txt\n\n";
  file << "=================================================================\n\n";
  
  file << "TOTAL PIPELINE LATENCY\n";
  file << "============================================\n";
  file << "Number of samples: " << total_pipeline_latencies_.size() << "\n";
  file << "Mean:   " << total_mean << " ms\n";
  file << "StdDev: " << total_std << " ms\n";
  file << "Min:    " << total_min << " ms\n";
  file << "Max:    " << total_max << " ms\n\n";
  
  file << "POINTOSR PROCESSING TIME\n";
  file << "============================================\n";
  file << "Number of samples: " << pointosr_latencies_.size() << "\n";
  file << "Mean:   " << pointosr_mean << " ms\n";
  file << "StdDev: " << pointosr_std << " ms\n";
  file << "Min:    " << pointosr_min << " ms\n";
  file << "Max:    " << pointosr_max << " ms\n\n";
  
  file << "COMMUNICATION OVERHEAD\n";
  file << "============================================\n";
  file << "Number of samples: " << communication_latencies_.size() << "\n";
  file << "Mean:   " << comm_mean << " ms\n";
  file << "StdDev: " << comm_std << " ms\n";
  file << "Min:    " << comm_min << " ms\n";
  file << "Max:    " << comm_max << " ms\n\n";
  
  file << "ANALYSIS\n";
  file << "============================================\n";
  file << "PointOSR %:      " << (pointosr_mean / total_mean * 100.0) << "%\n";
  file << "Communication %: " << (comm_mean / total_mean * 100.0) << "%\n\n";
  
  file << "All total pipeline latencies (ms):\n";
  for (size_t i = 0; i < total_pipeline_latencies_.size(); ++i) {
    file << total_pipeline_latencies_[i];
    if (i < total_pipeline_latencies_.size() - 1) file << ", ";
    if ((i + 1) % 10 == 0) file << "\n";
  }
  file << "\n\n";
  
  file << "All PointOSR latencies (ms):\n";
  for (size_t i = 0; i < pointosr_latencies_.size(); ++i) {
    file << pointosr_latencies_[i];
    if (i < pointosr_latencies_.size() - 1) file << ", ";
    if ((i + 1) % 10 == 0) file << "\n";
  }
  file << "\n\n";
  
  file << "All communication latencies (ms):\n";
  for (size_t i = 0; i < communication_latencies_.size(); ++i) {
    file << communication_latencies_[i];
    if (i < communication_latencies_.size() - 1) file << ", ";
    if ((i + 1) % 10 == 0) file << "\n";
  }
  file << "\n";
  
  file.close();
  LOG(INFO) << "Saved component latencies to: " << output_file;
  LOG(INFO) << "Total pipeline: " << total_mean << " ms (PointOSR: " << pointosr_mean 
            << " ms, Comm: " << comm_mean << " ms)";
  
  // Also save unified timing log
  saveUnifiedTimingLog();
}

void MotionDetector::saveUnifiedTimingLog() const {
  if (!config_.evaluate || !evaluator_) {
    return;
  }
  
  if (completed_timings_.empty()) {
    LOG(INFO) << "No unified timing data to save.";
    return;
  }
  
  std::string output_dir = evaluator_->getOutputDirectory();
  if (output_dir.empty()) {
    LOG(WARNING) << "Could not get output directory, skipping unified timing log.";
    return;
  }
  
  std::string output_file = output_dir + "/unified_timing_log.txt";
  
  std::ofstream file(output_file);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open unified timing file: " << output_file;
    return;
  }
  
  file << "=================================================================\n";
  file << "UNIFIED TIMING LOG - COMPLETE PIPELINE BREAKDOWN\n";
  file << "=================================================================\n\n";
  file << "This log shows the complete end-to-end timing for each frame\n";
  file << "when use_filtered_clusters=true, broken down by component.\n\n";
  file << "TIMELINE:\n";
  file << "  t0: PointCloud2 arrives at motion_detector\n";
  file << "  t1: Preprocessing complete\n";
  file << "  t2: Clustering complete\n";
  file << "  t3: ClusterArray published\n";
  file << "  t4: FilteredClusterArray received back\n";
  file << "  t5: Processing complete\n\n";
  file << "COMPONENTS:\n";
  file << "  - Dynablox Preprocessing: t0 → t1\n";
  file << "  - Dynablox Clustering: t1 → t2\n";
  file << "  - PointOSR Total: (from message)\n";
  file << "  - Communication: (t3 → t4) - PointOSR\n";
  file << "  - Total Pipeline: t3 → t4\n\n";
  file << "=================================================================\n\n";
  
  // Calculate aggregate statistics
  std::vector<double> preprocessing_times, clustering_times;
  std::vector<double> pointosr_times, communication_times, total_times;
  
  for (const auto& timing : completed_timings_) {
    preprocessing_times.push_back(timing.dynablox_preprocessing_ms);
    clustering_times.push_back(timing.dynablox_clustering_ms);
    pointosr_times.push_back(timing.pointosr_processing_ms);
    
    double total_ms = (timing.filtered_array_received - timing.cluster_array_published).toSec() * 1000.0;
    double comm_ms = total_ms - timing.pointosr_processing_ms;
    
    communication_times.push_back(comm_ms);
    total_times.push_back(total_ms);
  }
  
  // Helper to print stats
  auto print_stats = [&file](const std::string& name, const std::vector<double>& data) {
    if (data.empty()) return;
    double sum = 0.0, min_val = data[0], max_val = data[0];
    for (double val : data) {
      sum += val;
      if (val < min_val) min_val = val;
      if (val > max_val) max_val = val;
    }
    double mean = sum / data.size();
    double var_sum = 0.0;
    for (double val : data) {
      double diff = val - mean;
      var_sum += diff * diff;
    }
    double stddev = std::sqrt(var_sum / data.size());
    
    file << name << ":\n";
    file << "  Mean:   " << mean << " ms\n";
    file << "  StdDev: " << stddev << " ms\n";
    file << "  Min:    " << min_val << " ms\n";
    file << "  Max:    " << max_val << " ms\n\n";
  };
  
  file << "AGGREGATE STATISTICS\n";
  file << "============================================\n";
  file << "Samples: " << completed_timings_.size() << "\n\n";
  
  print_stats("Dynablox Preprocessing", preprocessing_times);
  print_stats("Dynablox Clustering", clustering_times);
  print_stats("PointOSR Processing", pointosr_times);
  print_stats("Communication Overhead", communication_times);
  print_stats("Total Pipeline (t3→t4)", total_times);
  
  file << "\n=================================================================\n";
  file << "PER-FRAME DETAILED BREAKDOWN\n";
  file << "=================================================================\n\n";
  file << std::fixed << std::setprecision(3);
  
  for (size_t i = 0; i < completed_timings_.size() && i < 50; ++i) {  // First 50 frames
    const auto& t = completed_timings_[i];
    
    double total_ms = (t.filtered_array_received - t.cluster_array_published).toSec() * 1000.0;
    double comm_ms = total_ms - t.pointosr_processing_ms;
    
    file << "Frame " << i << ":\n";
    file << "  Dynablox Preprocessing: " << t.dynablox_preprocessing_ms << " ms\n";
    file << "  Dynablox Clustering:    " << t.dynablox_clustering_ms << " ms\n";
    file << "  PointOSR Processing:    " << t.pointosr_processing_ms << " ms\n";
    file << "  Communication:          " << comm_ms << " ms\n";
    file << "  Total Pipeline:         " << total_ms << " ms\n";
    file << "  ---\n";
  }
  
  if (completed_timings_.size() > 50) {
    file << "\n... (showing first 50 frames, " << (completed_timings_.size() - 50) << " more frames omitted)\n";
  }
  
  file.close();
  LOG(INFO) << "Saved unified timing log to: " << output_file;
}

}
