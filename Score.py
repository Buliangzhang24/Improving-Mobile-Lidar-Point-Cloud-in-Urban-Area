import numpy as np
import open3d as o3d
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree

def load_point_cloud(pcd_path):
    """
    加载点云数据
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd


def downsample_points(points, voxel_size):
    """
    对点云数据进行下采样
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)


def compute_rmse(tls_points, mls_points):
    """
    计算RMSE
    """
    # 使用KDTree寻找最近的点
    tree_tls = cKDTree(tls_points)
    dist, _ = tree_tls.query(mls_points)

    # 计算RMSE
    rmse = np.sqrt(np.mean(dist ** 2))
    return rmse


def compute_normal_consistency(tls_normals, mls_normals, threshold=0.9):
    """
    计算法向一致性，基于法向量之间的夹角余弦相似度。

    参数:
    tls_normals: TLS点云的法向量
    mls_normals: MLS点云的法向量
    threshold: 用于判断法向一致性的余弦相似度阈值，默认为0.9（即大约 25° 角度）

    返回:
    normal_consistency: 法向一致性评分（范围 [0, 1]）
    """

    # 使用cKDTree查找每个MLS点的最近TLS点
    tree_tls = cKDTree(tls_normals)
    _, indices = tree_tls.query(mls_normals, k=1)  # 查询每个MLS法向量的最近TLS法向量的索引

    # 根据索引获取相应的TLS法向量
    nearest_tls_normals = tls_normals[indices]

    # 计算余弦相似度
    cos_sim = np.sum(nearest_tls_normals * mls_normals, axis=1)  # 点对之间的余弦相似度
    cos_sim = cos_sim / (np.linalg.norm(nearest_tls_normals, axis=1) * np.linalg.norm(mls_normals, axis=1))  # 归一化

    # 计算法向一致性
    consistent_normals = np.sum(cos_sim >= threshold)  # 满足阈值的点对数量
    normal_consistency = consistent_normals / len(cos_sim)  # 法向一致性（范围 0-1）

    return normal_consistency


def compute_completeness(tls_pcd, mls_pcd):
    """
    计算点云的完整性
    """
    # 这里简单计算点云的点数差异来衡量完整性
    tls_points = np.asarray(tls_pcd.points)
    mls_points = np.asarray(mls_pcd.points)

    completeness = len(mls_points) / len(tls_points)
    return completeness


def compute_score(rmse, normal_consistency, completeness, rmse_weight=0.4, normal_consistency_weight=0.4,
                  completeness_weight=0.2):
    """
    综合评估得分
    """
    score = (rmse_weight * rmse) + (normal_consistency_weight * normal_consistency) + (
            completeness_weight * completeness)
    return score


def downsample_points_and_normals(points, voxel_size):
    """
    对点云数据进行下采样并估算法向量
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 下采样
    down_pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(down_pcd.points)

    # 估算法向量
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(down_pcd.normals)

    return down_points, normals


def evaluate_point_cloud(tls_pcd_path, mls_pcd_path, voxel_size=0.01):
    """
    评估点云数据，计算RMSE、法向一致性、完整性和综合得分
    """
    tls_pcd = load_point_cloud(tls_pcd_path)
    mls_pcd = load_point_cloud(mls_pcd_path)

    if len(tls_pcd.points) == 0 or len(mls_pcd.points) == 0:
        print("Error: One of the point clouds is empty.")
        return None, None, None, None

    # 提取点云坐标
    tls_points = np.asarray(tls_pcd.points)
    mls_points = np.asarray(mls_pcd.points)

    # 对点云进行下采样及法向量估算
    tls_points, tls_normals = downsample_points_and_normals(tls_points, voxel_size)
    mls_points, mls_normals = downsample_points_and_normals(mls_points, voxel_size)

    print(f"Downsampled TLS Points: {tls_points.shape}")
    print(f"Downsampled MLS Points: {mls_points.shape}")

    # 计算 RMSE
    rmse = compute_rmse(tls_points, mls_points)  # 正确传递下采样后的点云

    # 计算法向一致性（只传递法向量）
    normal_consistency = compute_normal_consistency(tls_normals, mls_normals)

    # 计算完整性
    completeness = compute_completeness(tls_pcd, mls_pcd)

    # 计算综合得分
    score = compute_score(rmse, normal_consistency, completeness)

    return rmse, normal_consistency, completeness, score


# 路径
tls_pcd_path = "D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.ply"  # TLS点云路径
mls_pcd_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/roof_denoised_PointFilter.ply"  # 已去噪的MLS点云路径

# 调用评估函数
rmse, normal_consistency, completeness, score = evaluate_point_cloud(tls_pcd_path, mls_pcd_path)

# 输出结果
print(f"RMSE: {rmse}")
print(f"Normal Consistency: {normal_consistency}")
print(f"Completeness: {completeness}")
print(f"综合得分: {score}")
