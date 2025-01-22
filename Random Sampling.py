import open3d as o3d
import laspy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import os

# 加载 LAS 文件并转为 Open3D 点云
def load_las_as_o3d_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 计算 RMSE
def compute_rmse(denoised_points, reference_points):
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

    # 计算去噪率
    denoising_rate = len(denoised_points) / len(original_points) * 100
    return denoising_rate


# RANSAC去噪
def ransac_denoise(pcd, distance_threshold=0.05):
    # 估计平面模型并移除离群点
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3,
                                                    num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inlier_indices)
    return inlier_cloud


# 贝叶斯去噪
def bayesian_denoise(pcd, prior_prob_real=0.9, prior_prob_noise=0.1):
    points = np.asarray(pcd.points)
    # 使用贝叶斯估计进行噪声点识别
    # 这里只是一个示例，实际的贝叶斯方法可能需要更多的概率建模
    # 假设根据某种规则选择点，这里进行简单的噪声点过滤
    filtered_points = points[points[:, 2] > 0]  # 假设 Z 小于零的是噪声
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return denoised_pcd


# 密度估计去噪
def density_denoise(pcd, bandwidth=0.1):
    points = np.asarray(pcd.points)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    log_density = kde.score_samples(points)
    threshold = np.percentile(log_density, 20)  # 选择低密度点作为噪声
    denoised_points = points[log_density > threshold]
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(denoised_points)
    return denoised_pcd

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
    colors[~retained_mask] = [0, 0, 1]  # 去掉的点为红色
    colors[retained_mask] = [1, 0, 0]  # 保留的点为蓝色

    # 将颜色添加到点云
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    # 显示结果
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")

# 加载 TLS 和 MLS 点云
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/roof/Roof_MLS.las")

# 对 MLS 点云进行去噪
#mls_denoised_ransac = ransac_denoise(mls_pcd)
#mls_denoised_bayes = bayesian_denoise(mls_pcd)
mls_denoised_density = density_denoise(mls_pcd)

#visualize_denoising_fast(mls_pcd, mls_denoised_ransac)
#visualize_denoising_fast(mls_pcd, mls_denoised_bayes)
visualize_denoising_fast(mls_pcd, mls_denoised_density)
# 设置输出目录
output_dir = "D:/E_2024_Thesis/Data/Output/Roof/"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

# 保存去噪后的点云
#o3d.io.write_point_cloud(output_dir + "mls_denoised_ransac.ply", mls_denoised_ransac)
#o3d.io.write_point_cloud(output_dir + "mls_denoised_bayes.ply", mls_denoised_bayes)
o3d.io.write_point_cloud(output_dir + "mls_denoised_density.ply", mls_denoised_density)

print("Point clouds have been saved to the output directory.")

# 计算 RMSE
#rmse_mls_ransac = compute_rmse(np.asarray(mls_denoised_ransac.points), np.asarray(tls_pcd.points))
#rmse_mls_bayes = compute_rmse(np.asarray(mls_denoised_bayes.points), np.asarray(tls_pcd.points))
#rmse_mls_density = compute_rmse(np.asarray(mls_denoised_density.points), np.asarray(tls_pcd.points))

#print(f"RANSAC去噪RMSE: {rmse_mls_ransac}")
#print(f"贝叶斯去噪RMSE: {rmse_mls_bayes}")
#print(f"密度估计去噪RMSE: {rmse_mls_density}")

# 计算去噪率
#denoising_rate_ransac= compute_denoising_rate(mls_pcd, mls_denoised_ransac)
#denoising_rate_bayes = compute_denoising_rate(mls_pcd, mls_denoised_bayes)
#denoising_rate_density = compute_denoising_rate(mls_pcd, mls_denoised_density)

# 打印 RMSE 和去噪率
#print(f"去噪率: {denoising_rate_ransac}%")
#print(f" 去噪率: {denoising_rate_bayes}%")
#print(f"去噪率: {denoising_rate_density}%")
