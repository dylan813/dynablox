#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>

class ClusterExtractor {
public:
  ClusterExtractor(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private) 
    : nh_(nh), nh_private_(nh_private), frame_count_(0) {
    
    // Get parameters
    nh_private_.param<std::string>("output_dir", output_dir_, "/tmp/clusters");
    nh_private_.param<std::string>("file_prefix", file_prefix_, "cluster");
    nh_private_.param<bool>("save_as_bin", save_as_bin_, true);
    
    // Create output directory if it doesn't exist
    createDirectory(output_dir_);
    
    // Create subscriber
    clusters_sub_ = nh_.subscribe("eval_clusters", 1, &ClusterExtractor::clusterCallback, this);
    
    ROS_INFO("Cluster extractor initialized. Saving to: %s", output_dir_.c_str());
  }

private:
  bool createDirectory(const std::string& path) {
    // Check if directory exists
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
      // Directory doesn't exist, create it
      if (mkdir(path.c_str(), 0755) == -1) {
        ROS_ERROR("Failed to create directory %s: %s", 
                  path.c_str(), strerror(errno));
        return false;
      }
      ROS_INFO("Created directory: %s", path.c_str());
    }
    return true;
  }

  void clusterCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    // Convert ROS message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    
    // Map to store points by cluster ID (intensity value)
    std::map<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> cluster_map;
    
    // Separate points by cluster ID
    for (const auto& point : cloud->points) {
      int cluster_id = static_cast<int>(point.intensity);
      
      if (cluster_map.find(cluster_id) == cluster_map.end()) {
        cluster_map[cluster_id] = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
      }
      
      cluster_map[cluster_id]->points.push_back(point);
    }
    
    // Save each cluster to a separate file
    for (const auto& cluster_pair : cluster_map) {
      int cluster_id = cluster_pair.first;
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster = cluster_pair.second;
      
      // Set width and height
      cluster->width = cluster->points.size();
      cluster->height = 1;
      cluster->is_dense = true;
      
      // Create filename with frame number and cluster ID
      std::stringstream ss;
      ss << output_dir_ << "/" << file_prefix_ << "_frame_" 
         << std::setw(6) << std::setfill('0') << frame_count_ 
         << "_cluster_" << cluster_id;
      
      if (save_as_bin_) {
        // Save as binary file
        std::string bin_filename = ss.str() + ".bin";
        saveToBinary(cluster, bin_filename);
        ROS_INFO("Saved cluster %d with %zu points to %s", 
                 cluster_id, cluster->points.size(), bin_filename.c_str());
      } else {
        // Save as PCD file
        std::string pcd_filename = ss.str() + ".pcd";
        pcl::io::savePCDFile(pcd_filename, *cluster);
        ROS_INFO("Saved cluster %d with %zu points to %s", 
                 cluster_id, cluster->points.size(), pcd_filename.c_str());
      }
    }
    
    frame_count_++;
  }
  
  void saveToBinary(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const std::string& filename) {
    // Open binary file for writing
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    
    if (!file.is_open()) {
      ROS_ERROR("Could not open file %s for writing", filename.c_str());
      return;
    }
    
    // Write each point as x,y,z,intensity (4 floats = 16 bytes per point)
    for (const auto& point : cloud->points) {
      file.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
      file.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
      file.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
      file.write(reinterpret_cast<const char*>(&point.intensity), sizeof(float));
    }
    
    file.close();
  }
  
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber clusters_sub_;
  
  std::string output_dir_;
  std::string file_prefix_;
  bool save_as_bin_;
  int frame_count_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "cluster_extractor");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  
  ClusterExtractor extractor(nh, nh_private);
  
  ros::spin();
  
  return 0;
} 