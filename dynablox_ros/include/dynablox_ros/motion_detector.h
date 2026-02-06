#ifndef DYNABLOX_ROS_MOTION_DETECTOR_H_
#define DYNABLOX_ROS_MOTION_DETECTOR_H_

#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>
#include <map>
#include <mutex>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <dynablox_ros/ClusterArray.h>
#include <dynablox_ros/FilteredClusterArray.h>
#include <voxblox/core/block_hash.h>
#include <voxblox/core/common.h>
#include <voxblox_ros/tsdf_server.h>

#include "dynablox/3rd_party/config_utilities.hpp"
#include "dynablox/common/index_getter.h"
#include "dynablox/common/types.h"
#include "dynablox/evaluation/evaluator.h"
#include "dynablox/evaluation/ground_truth_handler.h"
#include "dynablox/processing/clustering.h"
#include "dynablox/processing/ever_free_integrator.h"
#include "dynablox/processing/preprocessing.h"
#include "dynablox/processing/tracking.h"
#include "dynablox_ros/visualization/motion_visualizer.h"

namespace dynablox {

class MotionDetector {
 public:
  // Config.
  struct Config : public config_utilities::Config<Config> {
    // If true evaluate the performance against GT.
    bool evaluate = false;

    // Enable helper and debug visualizations.
    bool visualize = true;

    // Print additional information when running.
    bool verbose = true;

    // Frame names.
    std::string global_frame_name = "map";
    std::string sensor_frame_name =
        "";  // Takes msg header if empty, overrides msg header if set.

    // Subscriber queue size.
    int queue_size = 1;

    // Number of threads to use.
    int num_threads = std::thread::hardware_concurrency();

    // If >0, shutdown after this many evaluated frames.
    int shutdown_after = 0;

    // Transform time delay handling
    bool use_latest_transform = true;
    double transform_lookup_timeout = 0.1;  // seconds
    bool use_previous_transform_on_fail = true;

    /// Default max (kept for backwards compatibility with config files)
    int max_cluster_topics = 30;

    // Use ClusterArray message instead of individual topics
    bool use_cluster_array = true;
    std::string cluster_array_topic = "/cluster_array";

    // FilteredClusterArray mode - when enabled, wait for classified human clusters before evaluation
    bool use_filtered_clusters = false;
    std::string filtered_cluster_array_topic = "/filtered_cluster_array";

    Config() { setConfigName("MotionDetector"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  // Constructor.
  MotionDetector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  
  // Destructor.
  ~MotionDetector();

  // Setup.
  void setupMembers();
  void setupRos();

  // Callbacks.
  void pointcloudCallback(const sensor_msgs::PointCloud2::Ptr& msg);
  void filteredClusterArrayCallback(const dynablox_ros::FilteredClusterArray::ConstPtr& msg);

  // Motion detection pipeline.
  bool lookupTransform(const std::string& target_frame,
                       const std::string& source_frame, uint64_t timestamp,
                       tf::StampedTransform& result) const;

  /**
   * @brief Create a mapping of each voxel index to the points it contains. Each
   * point will be checked whether it falls into an ever-free voxel and updates
   * voxel occupancy, since we go through voxels anyways already.
   *
   * @param cloud Complete point cloud to look up positions.
   * @param point_map Resulting map.
   * @param occupied_ever_free_voxel_indices Indices of voxels containing
   * ever-free points.
   * @param cloud_info Cloud info to store ever-free flags of checked points.
   */
  void setUpPointMap(
      const Cloud& cloud, BlockToPointMap& point_map,
      std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
      CloudInfo& cloud_info) const;

  /**
   * @brief Create a mapping of each block to ids of points that fall into it.
   *
   * @param cloud Points to process.
   * @return Mapping of block to point ids in cloud.
   */
  voxblox::HierarchicalIndexIntMap buildBlockToPointsMap(
      const Cloud& cloud) const;

  /**
   * @brief Create a mapping of each voxel index to the points it contains. Each
   * point will be checked whether it falls into an ever-free voxel and updates
   * voxel occupancy, since we go through voxels anyways already. This function
   * operates on a single block for data parallelism.
   *
   * @param cloud Complete point cloud to look up positions.
   * @param block_index Index of the block to be processed.
   * @param points_in_block Indices of all points in the block.
   * @param point_map Where to store the resulting point map for this block.
   * @param occupied_ever_free_voxel_indices Where to store the indices of ever
   * free voxels in this block.
   * @param cloud_info Cloud info to store ever-free flags of checked points.
   */
  void blockwiseBuildPointMap(
      const Cloud& cloud, const BlockIndex& block_index,
      const voxblox::AlignedVector<size_t>& points_in_block,
      VoxelToPointMap& point_map,
      std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
      CloudInfo& cloud_info) const;

  void processFilteredClusters(
      const std::unordered_map<int, sensor_msgs::PointCloud2::ConstPtr>& cluster_msgs,
      const ros::Time& stamp);

 private:
  const Config config_;

  // ROS.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber lidar_pcl_sub_;
  ros::Publisher cluster_batch_pub_;  // Kept for backwards compatibility
  ros::Publisher cluster_array_pub_;  // New unified cluster array publisher
  tf::TransformListener tf_listener_;
  mutable tf::StampedTransform last_good_T_M_S_;
  mutable bool have_last_good_transform_ = false;
  
  // FilteredClusterArray subscriber
  ros::Subscriber filtered_cluster_array_sub_;

  // Voxblox map.
  std::shared_ptr<voxblox::TsdfServer> tsdf_server_;
  std::shared_ptr<TsdfLayer> tsdf_layer_;

  // Processing.
  std::shared_ptr<Preprocessing> preprocessing_;
  std::shared_ptr<EverFreeIntegrator> ever_free_integrator_;
  std::shared_ptr<Clustering> clustering_;
  std::shared_ptr<Tracking> tracking_;
  std::shared_ptr<Evaluator> evaluator_;
  std::shared_ptr<MotionVisualizer> visualizer_;

  // Cached data.
  size_t voxels_per_side_;
  size_t voxels_per_block_;

  // Variables.
  int frame_counter_ = 0;
  int next_cluster_id_ = 0;

  // Legacy individual cluster publishers (kept for backwards compatibility)
  std::vector<ros::Publisher> cluster_pubs_;

  //store the raw pointcloud for evaluation
  std::map<ros::Time, std::pair<Cloud, CloudInfo>> raw_cloud_buffer_;
  std::mutex raw_buffer_lock_;
  
  // Track component-level latencies for pipeline analysis
  struct ComponentTiming {
    // Full pipeline timing (all wall-clock times)
    ros::WallTime pointcloud_received;       // t0: When PointCloud2 arrived
    ros::WallTime preprocessing_done;        // After preprocessing
    ros::WallTime clustering_done;           // After clustering
    ros::WallTime cluster_array_published;   // t1: When we sent ClusterArray
    ros::WallTime filtered_array_received;   // t2: When FilteredClusterArray arrived
    ros::WallTime processing_done;           // Final processing complete
    
    // Sub-component timings (from Dynablox timers)
    double dynablox_preprocessing_ms = 0.0;
    double dynablox_clustering_ms = 0.0;
    
    // PointOSR timing (from message)
    double pointosr_processing_ms = 0.0;
  };
  
  mutable std::vector<double> total_pipeline_latencies_;     // Total wall-clock time
  mutable std::vector<double> pointosr_latencies_;           // PointOSR processing time
  mutable std::vector<double> communication_latencies_;      // Communication overhead
  mutable std::map<ros::Time, ComponentTiming> timing_data_;
  mutable std::vector<ComponentTiming> completed_timings_;   // For unified logging
  mutable std::mutex latency_mutex_;
  void saveLatenciesToFile() const;
  void saveUnifiedTimingLog() const;
};

}  // namespace dynablox

#endif  // DYNABLOX_ROS_MOTION_DETECTOR_H_
