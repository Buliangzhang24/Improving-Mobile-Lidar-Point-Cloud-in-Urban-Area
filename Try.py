import open3d as o3d
import numpy as np
import laspy
from sklearn.neighbors import NearestNeighbors

def calculate_rmse(denoised_points, ground_truth_points):
    """
    计算 RMSE（均方根误差），通过最近邻搜索匹配点
    """
    # 使用最近邻搜索来匹配点云
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(ground_truth_points)
    # 找到每个 denoised_points 的最近邻点
    distances, _ = neigh.kneighbors(denoised_points)

    # 计算 RMSE
    rmse = np.sqrt(np.mean(distances**2))
    return rmse

# 点云下采样
def downsample_points(points, voxel_size):
    """
    对点云数据进行下采样
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

# 读取点云数据（PLY 格式）
def read_point_cloud(file_path):
    """
    读取 PLY 格式的点云文件
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

# 读取 LAS 格式的点云数据
def read_las_to_o3d(file_path):
    """
    将 LAS 文件读取为 Open3D 点云
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# 读取并处理点云数据
denoised_file_path = "D:/E_2024_Thesis/Data/denoised_point_cloud.ply"  # 你的去噪点云文件路径
ground_truth_file_path = "D:/E_2024_Thesis/Data/roof/roof_TLS.las"  # 参考点云文件路径

# 读取点云
denoised_points = read_point_cloud(denoised_file_path)
ground_truth_points = read_las_to_o3d(ground_truth_file_path)

# 下采样
voxel_size = 0.05  # 根据数据情况调整
denoised_points_down = downsample_points(denoised_points, voxel_size)
ground_truth_points_down = downsample_points(np.asarray(ground_truth_points.points), voxel_size)

# 计算 RMSE
print("Calculating RMSE...")
rmse = calculate_rmse(denoised_points_down, ground_truth_points_down)
print(f"RMSE: {rmse}")
