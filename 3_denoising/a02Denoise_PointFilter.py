import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import laspy

def read_las_to_o3d(file_path):
    """
    将 LAS 文件读取为 Open3D 点云
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def estimate_normals(points, k=20):
    """
    使用 k 近邻估计每个点的法向量
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = []
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)  # 协方差矩阵
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals.append(eigvecs[:, 0])  # 最小特征值对应的特征向量为法向量

    normals = np.array(normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def point_filter(noisy_points, ground_truth_normals, k=20, iterations=5):
    """
    PointFilter 算法实现
    """
    denoised_points = noisy_points.copy()

    for _ in range(iterations):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(denoised_points)
        _, indices = nbrs.kneighbors(denoised_points)

        new_points = []
        for i in range(denoised_points.shape[0]):
            neighbors = denoised_points[indices[i]]
            normal = ground_truth_normals[i]
            # 投影到法向量平面
            projected_points = neighbors - np.dot(neighbors - denoised_points[i], normal)[:, None] * normal
            new_points.append(np.mean(projected_points, axis=0))

        denoised_points = np.array(new_points)

    return denoised_points

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

def downsample_points(points, voxel_size):
    """
    对点云数据进行下采样
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

def calculate_noise_removal_rate(original_points, denoised_points):
    """
    Calculate noise removal rate as the percentage of points removed during 3_denoising.

    Parameters:
    original_points (np.ndarray): Original noisy point cloud points.
    denoised_points (np.ndarray): Denoised point cloud points.

    Returns:
    float: Noise removal rate in percentage.
    """
    original_count = len(original_points)
    denoised_count = len(denoised_points)
    removed_points = original_count - denoised_count
    noise_removal_rate = (removed_points / original_count) * 100
    return noise_removal_rate
# 加载点云数据
print("Loading LAS point cloud data...")
file_path_mls = "D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las"
file_path_tls = "D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las"

noisy_pcd = read_las_to_o3d(file_path_mls)
ground_truth_pcd = read_las_to_o3d(file_path_tls)

noisy_points = np.asarray(noisy_pcd.points)
ground_truth_points = np.asarray(ground_truth_pcd.points)

# Step 1: 估计法向量
print("Estimating ground truth normals...")
ground_truth_normals = estimate_normals(ground_truth_points)

# Step 2: 执行 PointFilter 算法
print("Running PointFilter 3_denoising...")
denoised_points = point_filter(noisy_points, ground_truth_normals)

# Step 3: 保存去噪后的点云
print("Saving denoised point cloud...")
denoised_pcd = o3d.geometry.PointCloud()
denoised_pcd.points = o3d.utility.Vector3dVector(denoised_points)
o3d.io.write_point_cloud("D:/E_2024_Thesis/Data/Output/Road/PointFilter.ply", denoised_pcd)
#o3d.io.write_point_cloud("D:/E_2024_Thesis/Output/Roof_Denoised_HighRes.pcd", denoised_pcd)

# Step 4: 计算 RMSE
# 下采样
#voxel_size = 0.05  # 根据数据情况调整
#denoised_points_down = downsample_points(denoised_points, voxel_size)
#ground_truth_points_down = downsample_points(np.asarray(ground_truth_points), voxel_size)

# 计算 RMSE
#print("Calculating RMSE...")
#rmse = calculate_rmse(denoised_points_down, ground_truth_points_down)
#print(f"RMSE: {rmse}")

# 可视化结果
#print("Visualizing point clouds...")
#noisy_pcd.paint_uniform_color([1, 0, 0])  # 红色表示有噪声的点云
#denoised_pcd.paint_uniform_color([0, 0, 1])  # 绿色表示去噪点云
#ground_truth_pcd.paint_uniform_color([0, 0, 1])  # 蓝色表示 Ground Truth 点云
#o3d.visualization.draw_geometries([noisy_pcd, denoised_pcd])

# Step 6: Calculate and display 3_denoising rate
# Function to calculate 3_denoising rate based on noise removal

# Step: Calculate and display noise removal rate
print("Calculating noise removal rate...")
#noise_removal_rate = calculate_noise_removal_rate(noisy_points, denoised_points)

# Display results
#print(f"Noise Removal Rate: {noise_removal_rate:.2f}%")