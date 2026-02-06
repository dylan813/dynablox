# Array of clusters for efficient batch processing
# Replaces 32 individual cluster topics with single atomic message

# Timing and frame information
std_msgs/Header header

# Number of clusters in this batch
uint32 num_clusters

# Array of cluster point clouds
sensor_msgs/PointCloud2[] clusters

# Original cluster indices (for tracking/debugging)
uint32[] cluster_ids

# Metadata for each cluster (optional, for optimization)
geometry_msgs/Point[] centroids
uint32[] cluster_sizes
