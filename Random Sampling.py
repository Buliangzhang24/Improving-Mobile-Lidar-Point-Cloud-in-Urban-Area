import open3d as o3d
import laspy
import numpy as np
from sklearn.neighbors import KernelDensity
import cupy as cp

# 加载 LAS 文件并转为 Open3D 点云
def load_las_as_o3d_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 计算 RMSE
# 使用 GPU 计算 RMSE
# 计算 RMSE，使用稀疏矩阵
def compute_rmse(reference_pcd, denoised_pcd):
    # 将点云数据转为稀疏矩阵
    reference_points = cp.asarray(np.asarray(reference_pcd.points))  # 转为 GPU 数组
    denoised_points = cp.asarray(np.asarray(denoised_pcd.points))  # 转为 GPU 数组

    # 确保两个点云数量相同，否则无法计算 RMSE
    if reference_points.shape[0] != denoised_points.shape[0]:
        raise ValueError("点云数量不同，无法计算 RMSE")

    # 转为稀疏矩阵
    reference_sparse = cp.sparse.COO.from_data(reference_points)
    denoised_sparse = cp.sparse.COO.from_data(denoised_points)

    # 计算点之间的欧几里得距离
    distances = cp.linalg.norm(reference_sparse - denoised_sparse, axis=1)

    # 计算 RMSE
    rmse = cp.sqrt(cp.mean(distances ** 2))
    return rmse.get()  # 将结果从 GPU 转回 CPU


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


# 加载 TLS 和 MLS 点云
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

# 对 MLS 点云进行去噪
mls_denoised_ransac = ransac_denoise(mls_pcd)
mls_denoised_bayes = bayesian_denoise(mls_pcd)
mls_denoised_density = density_denoise(mls_pcd)

# 计算 RMSE
#rmse_mls_ransac = compute_rmse(tls_pcd, mls_denoised_ransac)
rmse_mls_bayes = compute_rmse(tls_pcd, mls_denoised_bayes)
rmse_mls_density = compute_rmse(tls_pcd, mls_denoised_density)

# 计算去噪率
#denoising_rate_ransac = compute_denoising_rate(mls_pcd, mls_denoised_ransac)
denoising_rate_bayes = compute_denoising_rate(mls_pcd, mls_denoised_bayes)
denoising_rate_density = compute_denoising_rate(mls_pcd, mls_denoised_density)

# 打印 RMSE 和去噪率
#print(f"RANSAC去噪RMSE: {rmse_mls_ransac}, 去噪率: {denoising_rate_ransac}%")
print(f"贝叶斯去噪RMSE: {rmse_mls_bayes}, 去噪率: {denoising_rate_bayes}%")
print(f"密度估计去噪RMSE: {rmse_mls_density}, 去噪率: {denoising_rate_density}%")
