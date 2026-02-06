# Array of classified/filtered clusters for efficient batch processing
# Replaces 32 individual filtered cluster topics with single atomic message

# Timing and frame information
std_msgs/Header header

# Number of clusters in this batch
uint32 num_clusters

# Array of filtered cluster point clouds
sensor_msgs/PointCloud2[] clusters

# Original cluster indices (for tracking)
uint32[] cluster_ids

# Processing time (milliseconds) - for latency analysis
float64 processing_time_ms

# Classification results for each cluster
string[] class_names
float32[] confidences
bool[] is_human
