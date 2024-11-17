import open3d as o3d
import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans
import laspy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# 加载LAS文件为Open3D点云对象
def load_las_as_o3d_point_cloud(file_path):
    # 使用 laspy 加载 .las 文件
    las_data = laspy.read(file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd

# 统计流形去噪 (通过流形重建)
def manifold_reconstruction_denoising(pcd, num_neighbors=20):
    # 基于KNN计算流形特征
    pcd_points = np.asarray(pcd.points)
    tree = scipy.spatial.KDTree(pcd_points)

    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=num_neighbors)
        neighbor_points = pcd_points[neighbors[1], :]
        denoised_point = np.mean(neighbor_points, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd


# K均值聚类与统计流形结合去噪
def kmeans_statistical_manifold_denoising(pcd, num_clusters=5):
    pcd_points = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pcd_points)

    # 根据簇分配去噪
    denoised_points = []
    for i in range(num_clusters):
        cluster_points = pcd_points[kmeans.labels_ == i]
        cluster_center = np.mean(cluster_points, axis=0)
        denoised_points.extend([cluster_center] * len(cluster_points))

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd


# 流体几何特征截断去噪
def manifold_distance_truncation_denoising(pcd, distance_threshold=0.05):
    pcd_points = np.asarray(pcd.points)
    tree = scipy.spatial.KDTree(pcd_points)

    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=20)
        neighbor_points = pcd_points[neighbors[1], :]

        distances = np.linalg.norm(neighbor_points - point, axis=1)
        close_neighbors = neighbor_points[distances < distance_threshold]

        # 使用较近邻点的平均值
        denoised_point = np.mean(close_neighbors, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd


# 流体启发式去噪（通过概率分布去噪）
def fluid_inspired_denoising(pcd):
    pcd_points = np.asarray(pcd.points)

    # 对点云进行一定的平滑操作，这里假设流体分布可以通过均值平滑近邻点来表示
    tree = scipy.spatial.KDTree(pcd_points)
    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=20)
        neighbor_points = pcd_points[neighbors[1], :]

        # 使用均值平滑
        denoised_point = np.mean(neighbor_points, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd

def compute_rmse(denoised_pcd, reference_pcd):
    # 可选：对点云进行轻微的随机旋转和/或平移，避免完全匹配
    random_rotation = R.random().as_matrix()  # 随机旋转矩阵
    denoised_points = np.asarray(denoised_pcd.points)
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(
        denoised_points, axis=0)

    reference_points = np.asarray(reference_pcd.points)

    # 使用最近邻算法找到每个去噪点云点的最近参考点云点
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # 计算每个去噪点与参考点云中最接近点之间的距离的平方
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def compute_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)
    removal_rate = (len(original_points) - len(denoised_points)) / len(original_points) * 100
    return removal_rate

# 载入数据
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

# 选择去噪方法并应用
denoised_mls_reconstruction = manifold_reconstruction_denoising(mls_pcd)
denoised_mls_statistical = kmeans_statistical_manifold_denoising(mls_pcd)
denoised_mls_truncation = manifold_distance_truncation_denoising(mls_pcd)
denoised_mls_fluid = fluid_inspired_denoising(mls_pcd)


denoised_rmse_reconstruction = compute_rmse(denoised_mls_reconstruction, tls_pcd)
denoised_rmse_statistical = compute_rmse(denoised_mls_statistical, tls_pcd)
denoised_rmse_truncation = compute_rmse(denoised_mls_truncation, tls_pcd)
denoised_rmse_fluid = compute_rmse(denoised_mls_fluid, tls_pcd)

print(f"Reconstruction RMSE: {denoised_rmse_reconstruction:.4f}")
print(f"Reconstruction Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_rmse_reconstruction):.2f}%")
print(f"Statisical RMSE: {denoised_rmse_statistical:.4f}")
print(f"Statisical Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_rmse_statistical):.2f}%")
print(f"Truncation RMSE: {denoised_rmse_truncation:.4f}")
print(f"Truncation Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_rmse_truncation):.2f}%")
print(f"Fluid RMSE: {denoised_rmse_fluid:.4f}")
print(f"Fluid Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_rmse_fluid):.2f}%")