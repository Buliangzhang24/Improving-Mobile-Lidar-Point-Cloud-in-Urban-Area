import open3d as o3d
import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import random

# 设置随机种子
random.seed(42)  # 设置 Python random 模块的种子
np.random.seed(42)  # 设置 numpy 模块的种子

# ICP 配准函数
def align_point_clouds(source_pcd, target_pcd, threshold=0.02, trans_init=np.eye(4)):
    # 计算法线
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用点到平面的 ICP 配准
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 应用变换将点云对齐
    aligned_pcd = source_pcd.transform(reg_p2l.transformation)
    return aligned_pcd

def knn_denoise_patch(point_cloud, k=25, distance_threshold=0.2):
    points = np.asarray(point_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    denoised_points = []
    for i, point in enumerate(points):
        avg_distance = np.mean(distances[i])
        if avg_distance < distance_threshold:
            denoised_points.append(point)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised

def knn_denoise_manifold(point_cloud, k=25, curvature_threshold=0.2):
    points = np.asarray(point_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    denoised_points = []
    for i, point in enumerate(points):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        curvature = min(eigenvalues) / sum(eigenvalues)

        if curvature < curvature_threshold:
            denoised_points.append(point)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised


def knn_denoise_voxel(point_cloud, voxel_size=0.05, k=25, distance_threshold=0.2):
    # 下采样点云到体素网格，获取每个体素的中心点
    voxel_grid = point_cloud.voxel_down_sample(voxel_size)
    voxel_centers = np.asarray(voxel_grid.points)  # 获取体素中心点

    denoised_points = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(voxel_centers)
    distances, _ = nbrs.kneighbors(voxel_centers)

    for i, center in enumerate(voxel_centers):
        avg_distance = np.mean(distances[i])
        if avg_distance < distance_threshold:
            denoised_points.append(center)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised

# 读取 .las 文件并转换为 Open3D 点云格式
def read_las_to_o3d(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def compute_rmse(denoised_points, reference_points):
    # 可选：对点云进行轻微的随机旋转和/或平移，避免完全匹配
    random_rotation = R.random().as_matrix()  # 随机旋转矩阵
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(
        denoised_points, axis=0)

    # 使用最近邻算法找到每个去噪点云点的最近参考点云点
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # 计算每个去噪点与参考点云中最接近点之间的距离的平方
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse


# 计算去噪率
def compute_denoising_rate(original_point_cloud, denoised_point_cloud):
    original_points = np.asarray(original_point_cloud.points)
    denoised_points = np.asarray(denoised_point_cloud.points)
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


# 载入点云并运行去噪函数
file_path = "D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las"
pcd = read_las_to_o3d(file_path)
file_path_1 = "D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las"
reference_pcd = read_las_to_o3d(file_path_1)

# 这里使用前面的knn_denoise_patch函数
pcd_denoised_patch = knn_denoise_patch(pcd)
#o3d.visualization.draw_geometries([pcd_denoised_patch])

pcd_denoised_manifold = knn_denoise_manifold(pcd)
#o3d.visualization.draw_geometries([pcd_denoised_manifold])

pcd_denoised_voxel = knn_denoise_voxel(pcd)
#o3d.visualization.draw_geometries([pcd_denoised_voxel])

output_dir = "D:/E_2024_Thesis/Data/Output/Road/"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

# 保存去噪后的点云
o3d.io.write_point_cloud(output_dir + "pcd_denoised_patch.ply", pcd_denoised_patch)
o3d.io.write_point_cloud(output_dir + "pcd_denoised_manifold.ply", pcd_denoised_manifold)
o3d.io.write_point_cloud(output_dir + "pcd_denoised_voxel.ply", pcd_denoised_voxel)

print("Point clouds have been saved to the output directory.")

# 使用去噪后的点云调用可视化函数
#visualize_denoising_fast(pcd, pcd_denoised_voxel)
#visualize_denoising_fast(pcd, pcd_denoised_manifold)
#visualize_denoising_fast(pcd, pcd_denoised_patch)

# 对齐每个去噪后的点云到参考点云
#aligned_pcd_patch = align_point_clouds(pcd_denoised_patch, reference_pcd)
#aligned_pcd_manifold = align_point_clouds(pcd_denoised_manifold, reference_pcd)
#aligned_pcd_voxel = align_point_clouds(pcd_denoised_voxel, reference_pcd)
#aligned_pcd_origin = align_point_clouds(pcd, reference_pcd)

# 计算不同方法的 RMSE
#rmse_patch = compute_rmse(np.asarray(aligned_pcd_patch.points), np.asarray(reference_pcd.points))
#rmse_manifold = compute_rmse(np.asarray(aligned_pcd_manifold.points), np.asarray(reference_pcd.points))
#rmse_voxel = compute_rmse(np.asarray(aligned_pcd_voxel.points), np.asarray(reference_pcd.points))
#original_rmse = compute_rmse(np.asarray(aligned_pcd_origin.points), np.asarray(reference_pcd.points))

# 计算去噪率
#denoising_rate_patch = compute_denoising_rate(pcd, pcd_denoised_patch)
#denoising_rate_manifold = compute_denoising_rate(pcd, pcd_denoised_manifold)
#denoising_rate_voxel = compute_denoising_rate(pcd, pcd_denoised_voxel)

# 输出结果
#print(f"Original RMSE: {original_rmse:.4f}")
#print("KNN Patch RMSE:", rmse_patch)
#print("KNN Patch Denoising Rate:", denoising_rate_patch, "%")
#print("KNN Manifold RMSE:", rmse_manifold)
#print("KNN Manifold Denoising Rate:", denoising_rate_manifold, "%")
#print("KNN Voxel RMSE:", rmse_voxel)
#print("KNN Voxel Denoising Rate:", denoising_rate_voxel, "%")
