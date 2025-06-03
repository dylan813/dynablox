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
#include <vector>
#include <boost/bind.hpp>

class ClusterExtractor {
public:
  ClusterExtractor(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private) 
    : nh_(nh), nh_private_(nh_private), frame_count_(0) {
    
    nh_private_.param<std::string>("output_dir", output_dir_, "/tmp/clusters");
    nh_private_.param<std::string>("file_prefix", file_prefix_, "location");
    nh_private_.param<bool>("save_as_bin", save_as_bin_, true);
    nh_private_.param<int>("num_clusters", num_clusters_, 100);
    
    createDirectory(output_dir_);
    
    for (int i = 0; i < num_clusters_; i++) {
      std::string topic_name = "cluster_" + std::to_string(i);
      cluster_subs_.push_back(
        nh_.subscribe<sensor_msgs::PointCloud2>(
          topic_name, 1, 
          boost::bind(&ClusterExtractor::clusterCallback, this, _1, i)
        )
      );
    }
    
    ROS_INFO("Cluster extractor initialized. Saving to: %s", output_dir_.c_str());
  }

private:
  bool createDirectory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
      if (mkdir(path.c_str(), 0755) == -1) {
        ROS_ERROR("Failed to create directory %s: %s", 
                  path.c_str(), strerror(errno));
        return false;
      }
      ROS_INFO("Created directory: %s", path.c_str());
    }
    return true;
  }

  void clusterCallback(const sensor_msgs::PointCloud2::ConstPtr& msg, int cluster_id) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cluster);
    
    cluster->width = cluster->points.size();
    cluster->height = 1;
    cluster->is_dense = true;
    
    std::stringstream ss;
    ss << output_dir_ << "/" << file_prefix_ << "_frame_" 
       << std::setw(6) << std::setfill('0') << frame_count_ 
       << "_cluster_" << cluster_id;
    
    if (save_as_bin_) {
      std::string bin_filename = ss.str() + ".bin";
      saveToBinary(cluster, bin_filename);
      ROS_INFO("Saved cluster %d with %zu points to %s", 
               cluster_id, cluster->points.size(), bin_filename.c_str());
    } else {
      std::string pcd_filename = ss.str() + ".pcd";
      pcl::io::savePCDFile(pcd_filename, *cluster);
      ROS_INFO("Saved cluster %d with %zu points to %s", 
               cluster_id, cluster->points.size(), pcd_filename.c_str());
    }
    
    if (cluster_id == 0) {
      frame_count_++;
    }
  }
  
  void saveToBinary(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const std::string& filename) {
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    
    if (!file.is_open()) {
      ROS_ERROR("Could not open file %s for writing", filename.c_str());
      return;
    }
    
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
  std::vector<ros::Subscriber> cluster_subs_;
  
  std::string output_dir_;
  std::string file_prefix_;
  bool save_as_bin_;
  int frame_count_;
  int num_clusters_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "cluster_extractor");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  
  ClusterExtractor extractor(nh, nh_private);
  
  ros::spin();
  
  return 0;
} 