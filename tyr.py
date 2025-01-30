import open3d as o3d
import torch
import numpy as np

# 加载点云文件
def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

# 将 Open3D 点云转换为 PyTorch Tensor
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)

# 计算 RMSE
def calculate_rmse(pcd1, pcd2):
    # 确保两点云的大小一致，取最小的点云大小
    num_points = min(len(pcd1), len(pcd2))

    # 随机选择点云中的点
    idx1 = torch.randperm(len(pcd1))[:num_points]
    idx2 = torch.randperm(len(pcd2))[:num_points]

    pcd1 = pcd1[idx1]
    pcd2 = pcd2[idx2]

    # 计算欧几里得距离
    dist = torch.norm(pcd1 - pcd2, dim=1)
    rmse = torch.sqrt(torch.mean(dist ** 2))  # 均方根误差
    return rmse

# 加载去噪后的点云和 TLS 点云
denoised_pcd = load_point_cloud("D:/E_2024_Thesis/Data/Output/Roof/PointCloud/mls_pointnet.ply")  # 使用您保存的去噪点云
tls_pcd = load_point_cloud("D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.ply")  # TLS 点云

# 转换为 PyTorch Tensor
denoised_tensor = point_cloud_to_tensor(denoised_pcd)
tls_tensor = point_cloud_to_tensor(tls_pcd)

# 计算 RMSE
rmse = calculate_rmse(denoised_tensor, tls_tensor)
print(f"RMSE: {rmse.item():.6f}")

# 可视化结果
o3d.visualization.draw_geometries([denoised_pcd, tls_pcd])

