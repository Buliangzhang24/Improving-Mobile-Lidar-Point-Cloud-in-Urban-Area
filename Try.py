import open3d as o3d
import numpy as np

# Create a sample point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# Try visualizing
o3d.visualization.draw_geometries([pcd])
