import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import laspy

# 读取 las 文件并转换为 open3d 点云对象
def read_las_to_o3d(file_path):
    # 读取 LAS 文件
    las = laspy.read(file_path)
    # 获取点云坐标
    points = np.vstack((las.x, las.y, las.z)).transpose()
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# 定义 IterativePFN 模型
class IterativePFN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3, num_iterations=3):
        super(IterativePFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_iterations = num_iterations

        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 迭代优化模块
        self.refinement_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, noisy_points):
        refined_points = noisy_points
        for _ in range(self.num_iterations):
            features = self.feature_extractor(refined_points)
            delta = self.refinement_network(features)
            refined_points = refined_points + delta
        return refined_points


# Chamfer 距离作为损失函数
def chamfer_loss(pred_points, gt_points):
    diff_1 = torch.cdist(pred_points, gt_points, p=2).min(dim=1)[0]
    diff_2 = torch.cdist(gt_points, pred_points, p=2).min(dim=1)[0]
    return diff_1.mean() + diff_2.mean()


# 训练模型的函数
def train_iterative_pfn(model, optimizer, noisy_points, gt_points, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        refined_points = model(noisy_points)
        loss = chamfer_loss(refined_points, gt_points)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model


# 准备点云数据的函数
def prepare_data(noisy_file_path, gt_file_path):
    noisy_pcd = read_las_to_o3d(noisy_file_path)
    gt_pcd = read_las_to_o3d(gt_file_path)
    noisy_points = torch.tensor(np.asarray(noisy_pcd.points), dtype=torch.float32)
    gt_points = torch.tensor(np.asarray(gt_pcd.points), dtype=torch.float32)
    return noisy_points, gt_points


# 计算 RMSE 的函数
def calculate_rmse(pred_points, gt_points):
    return np.sqrt(np.mean(np.linalg.norm(pred_points - gt_points, axis=1) ** 2))


# 主程序
if __name__ == "__main__":
    noisy_file_path = "D:/E_2024_Thesis/Data/roof/roof_MLS.las"  # 你的噪声点云文件路径
    gt_file_path = "D:/E_2024_Thesis/Data/roof/roof_TLS.las"  # 参考点云文件路径

    # 准备数据
    noisy_points, gt_points = prepare_data(noisy_file_path, gt_file_path)

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 将数据移到 GPU
    noisy_points = noisy_points.to(device)
    gt_points = gt_points.to(device)

    # 初始化模型和优化器，并将模型移到 GPU
    model = IterativePFN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    trained_model = train_iterative_pfn(model, optimizer, noisy_points, gt_points)

    # 预测（去噪）
    refined_points = trained_model(noisy_points).detach().cpu().numpy()
    print("Denoising complete.")

    # 计算 RMSE
    rmse = calculate_rmse(refined_points, gt_points.cpu().numpy())
    print(f"RMSE: {rmse}")

    # 可视化去噪结果
    refined_pcd = o3d.geometry.PointCloud()
    refined_pcd.points = o3d.utility.Vector3dVector(refined_points)
    o3d.visualization.draw_geometries([refined_pcd])

    # 可视化参考点云（ground truth）
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points.cpu().numpy())
    o3d.visualization.draw_geometries([gt_pcd])
