import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import laspy


# 加载 LAS 点云为 Open3D 点云对象
def load_las_as_o3d_point_cloud(las_file_path):
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 标准化和归一化
def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 计算RMSE
def calculate_rmse(predicted_pcd, ground_truth_pcd):
    predicted_points = np.asarray(predicted_pcd.points)
    ground_truth_points = np.asarray(ground_truth_pcd.points)

    # 确保点云大小一致
    assert predicted_points.shape == ground_truth_points.shape, "点云大小不匹配"

    # 计算每个点的误差并返回RMSE
    error = np.linalg.norm(predicted_points - ground_truth_points, axis=1)
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse


# 计算去噪率
def calculate_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)
    denoising_rate = len(denoised_points) / len(original_points) * 100
    return denoising_rate


# 定义 PointProNets（PointNet + CNN）
class PointProNets(nn.Module):
    def __init__(self):
        super(PointProNets, self).__init__()
        # PointNet 部分
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 3)  # 输出去噪后的点云

        # CNN 部分
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # PointNet 部分
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        # CNN 部分
        x = x.transpose(1, 2)  # 转置为适应1D卷积
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        return x.transpose(1, 2)  # 转回原来的形状


# 使用几何与语义融合方法进行去噪
class GeometricSemanticFusion(nn.Module):
    def __init__(self):
        super(GeometricSemanticFusion, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.transpose(1, 2)


# 基于CNN的鲁棒性方法
class RobustCNNMethod(nn.Module):
    def __init__(self):
        super(RobustCNNMethod, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.transpose(1, 2)


# 加载点云文件
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

# 归一化点云
tls_pcd = normalize_point_cloud(tls_pcd)
mls_pcd = normalize_point_cloud(mls_pcd)


# 转换为 Tensor
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)


tls_tensor = point_cloud_to_tensor(tls_pcd)
mls_tensor = point_cloud_to_tensor(mls_pcd)


# 定义函数来训练和评估每个模型
def train_and_evaluate_model(model, mls_tensor, tls_tensor):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 输入 MLS 点云，输出去噪后的点云
        output = model(mls_tensor)  # 获取去噪后的点云

        # 损失函数：这里使用简单的均方误差（MSE）
        loss = F.mse_loss(output, tls_tensor)  # 计算损失

        loss.backward()
        optimizer.step()

        # 每100次打印一次损失值
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    # 使用训练好的模型进行去噪
    model.eval()
    with torch.no_grad():
        denoised_pcd = model(mls_tensor)

    # 可视化去噪后的点云
    denoised_o3d_pcd = o3d.geometry.PointCloud()
    denoised_o3d_pcd.points = o3d.utility.Vector3dVector(denoised_pcd.numpy())
    o3d.visualization.draw_geometries([denoised_o3d_pcd])

    # 计算 RMSE 和 去噪率
    rmse = calculate_rmse(denoised_o3d_pcd, tls_pcd)
    denoising_rate = calculate_denoising_rate(mls_pcd, denoised_o3d_pcd)

    return rmse, denoising_rate


# 实例化三个不同的去噪模型
point_pro_nets_model = PointProNets()
geometric_semantic_model = GeometricSemanticFusion()
robust_cnn_model = RobustCNNMethod()

# 训练并评估三个模型
print("Training and Evaluating PointProNets Model...")
rmse1, denoising_rate1 = train_and_evaluate_model(point_pro_nets_model, mls_tensor, tls_tensor)
print(f"PointProNets Model RMSE: {rmse1:.4f}, Denoising Rate: {denoising_rate1:.2f}%")

print("\nTraining and Evaluating Geometric Semantic Fusion Model...")
rmse2, denoising_rate2 = train_and_evaluate_model(geometric_semantic_model, mls_tensor, tls_tensor)
print(f"Geometric Semantic Fusion Model RMSE: {rmse2:.4f}, Denoising Rate: {denoising_rate2:.2f}%")

print("\nTraining and Evaluating Robust CNN Model...")
rmse3, denoising_rate3 = train_and_evaluate_model(robust_cnn_model, mls_tensor, tls_tensor)
print(f"Robust CNN Model RMSE: {rmse3:.4f}, Denoising Rate: {denoising_rate3:.2f}%")
