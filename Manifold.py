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

def visualize_denoising_fast(pcd_original, pcd_denoised):
    # 获取原始点云和去噪后点云的点
    original_points = np.asarray(pcd_original.points)
    denoised_points = np.asarray(pcd_denoised.points)

    # 构建去噪点云的 k-d 树以加速查找
    kdtree = o3d.geometry.KDTreeFlann(pcd_denoised)

    # 初始化保留的掩码
    retained_mask = np.zeros(len(original_points), dtype=bool)

    # 在 k-d 树中查找每个点是否存在
    for i, point in enumerate(original_points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)  # 查找最近邻点
        if len(idx) > 0 and np.linalg.norm(denoised_points[idx[0]] - point) <= 1e-6:
            retained_mask[i] = True

    # 创建一个颜色数组
    colors = np.zeros_like(original_points)
    colors[~retained_mask] = [1, 0, 0]  # 去掉的点为红色
    colors[retained_mask] = [0, 0, 1]  # 保留的点为蓝色

    # 将颜色添加到点云
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    # 显示结果
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")


# 载入数据
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/data/Roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/data/Roof_MLS.las")

# 选择去噪方法并应用
denoised_mls_reconstruction = manifold_reconstruction_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_reconstruction)
denoised_mls_statistical = kmeans_statistical_manifold_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_statistical)
denoised_mls_truncation = manifold_distance_truncation_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_truncation)
denoised_mls_fluid = fluid_inspired_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_fluid)

denoised_rmse_reconstruction = compute_rmse(denoised_mls_reconstruction, tls_pcd)
denoised_rmse_statistical = compute_rmse(denoised_mls_statistical, tls_pcd)
denoised_rmse_truncation = compute_rmse(denoised_mls_truncation, tls_pcd)
denoised_rmse_fluid = compute_rmse(denoised_mls_fluid, tls_pcd)

print(f"Reconstruction RMSE: {denoised_rmse_reconstruction:.4f}")
#print(f"Reconstruction Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_mls_reconstruction):.2f}%")

print(f"Statisical RMSE: {denoised_rmse_statistical:.4f}")
#print(f"Statisical Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_mls_statistical):.2f}%")

print(f"Truncation RMSE: {denoised_rmse_truncation:.4f}")
#print(f"Truncation Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_mls_truncation):.2f}%")

print(f"Fluid RMSE: {denoised_rmse_fluid:.4f}")
#print(f"Fluid Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_mls_fluid):.2f}%")
