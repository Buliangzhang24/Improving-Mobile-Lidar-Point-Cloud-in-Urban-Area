import open3d as o3d
import laspy
import numpy as np

# 加载 LAS 文件并转为 Open3D 点云
def load_las_as_o3d_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd




#tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/LiDAR_Engelseplein/Engelseplein_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/data/Roof_MLS.las")

#o3d.visualization.draw_geometries([tls_pcd], window_name="Denoising Visualization")
o3d.visualization.draw_geometries([mls_pcd], window_name="Denoising Visualization")

