import open3d as o3d
import numpy as np
import laspy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R


# 加载 LAS 文件为 Open3D 格式的点云
def load_las_as_o3d_point_cloud(file_path):
    # 使用 laspy 加载 .las 文件
    las_data = laspy.read(file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd


# 基于导向滤波的去噪 (Zhou et al., 2022)
def guided_filtering(pcd, iterations=5, filter_strength=0.1):
    # 创建 KDTree 用于邻域搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)

    for _ in range(iterations):
        filtered_normals = normals.copy()
        for i in range(len(normals)):
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)  # 搜索10个最近邻
            avg_normal = np.mean(normals[neighbors], axis=0)
            filtered_normals[i] = (1 - filter_strength) * normals[i] + filter_strength * avg_normal
        normals = filtered_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# 双边滤波迭代方法 (Hurtado et al., 2023)
def bilateral_filtering(pcd, iterations=5, spatial_sigma=0.5, normal_sigma=0.1):
    # 创建 KDTree 用于邻域搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    for _ in range(iterations):
        filtered_normals = normals.copy()
        for i in range(len(normals)):
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(points[i], 10)  # 搜索10个最近邻
            neighbor_normals = normals[neighbors]
            neighbor_points = points[neighbors]
            weights_spatial = np.exp(
                -np.linalg.norm(neighbor_points - points[i], axis=1) ** 2 / (2 * spatial_sigma ** 2))
            weights_normal = np.exp(
                -np.linalg.norm(neighbor_normals - normals[i], axis=1) ** 2 / (2 * normal_sigma ** 2))
            weights = weights_spatial * weights_normal
            filtered_normals[i] = np.sum(weights[:, None] * neighbor_normals, axis=0) / np.sum(weights)
        normals = filtered_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# 各向异性扩散方法 (扩展法向量去噪)
def anisotropic_diffusion(pcd, iterations=5, diffusion_factor=0.1):
    # 创建 KDTree 用于邻域搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    for _ in range(iterations):
        updated_normals = normals.copy()
        for i in range(len(normals)):
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(points[i], 10)  # 搜索10个最近邻
            for neighbor_idx in neighbors:
                diff = normals[neighbor_idx] - normals[i]
                weight = np.exp(-np.linalg.norm(points[neighbor_idx] - points[i]) ** 2)
                updated_normals[i] += diffusion_factor * weight * diff
            updated_normals[i] = updated_normals[i] / np.linalg.norm(updated_normals[i])  # 单位化
        normals = updated_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# 计算 RMSE
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


# 计算去噪率
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


# 加载点云文件
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/roof/Roof_MLS.las")
output_dir = "D:/E_2024_Thesis/Data/Output/Roof/"
# 估算法向量
tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

# 计算原始 RMSE
#original_rmse = compute_rmse(mls_pcd, tls_pcd)
#print(f"Original RMSE: {original_rmse:.4f}")

# 使用基于导向滤波的去噪方法
denoised_pcd_guided = guided_filtering(mls_pcd)
visualize_denoising_fast(mls_pcd, denoised_pcd_guided)
o3d.io.write_point_cloud(output_dir + "mls_guided.ply", denoised_pcd_guided)

#denoised_rmse_guided = compute_rmse(denoised_pcd_guided, tls_pcd)
#print(f"Guided Filtering RMSE: {denoised_rmse_guided:.4f}")
#print(f"Guided Filtering Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_pcd_guided):.2f}%")

# 使用双边滤波迭代方法
denoised_pcd_bilateral = bilateral_filtering(mls_pcd)
visualize_denoising_fast(mls_pcd, denoised_pcd_bilateral)
o3d.io.write_point_cloud(output_dir + "mls_bilateral.ply", denoised_pcd_bilateral)

#denoised_rmse_bilateral = compute_rmse(denoised_pcd_bilateral, tls_pcd)
#print(f"Bilateral Filtering RMSE: {denoised_rmse_bilateral:.4f}")
#print(f"Bilateral Filtering Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_pcd_bilateral):.2f}%")

# 使用各向异性扩散方法
denoised_pcd_anisotropic = anisotropic_diffusion(mls_pcd)
visualize_denoising_fast(mls_pcd, denoised_pcd_anisotropic)
o3d.io.write_point_cloud(output_dir + "mls_anisotropic.ply", denoised_pcd_anisotropic)


#denoised_rmse_anisotropic = compute_rmse(denoised_pcd_anisotropic, tls_pcd)
#print(f"Anisotropic Diffusion RMSE: {denoised_rmse_anisotropic:.4f}")
#print(f"Anisotropic Diffusion Denoising Rate: {compute_denoising_rate(mls_pcd, denoised_pcd_anisotropic):.2f}%")
