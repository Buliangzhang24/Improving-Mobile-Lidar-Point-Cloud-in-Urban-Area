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


# 转换为 Tensor，并确保点云数据维度正确
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    # Add a batch dimension to the tensor, making it (1, num_points, 3)
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


# 定义计算RMSE函数
def calculate_rmse(denoised_tensor, tls_tensor):
    """
    计算 RMSE，输入为 Tensor，形状为 (batch_size, num_points, 3)。
    """
    if denoised_tensor.dim() == 3:
        denoised_points = denoised_tensor.squeeze(0)  # 移除 batch 维度
    else:
        denoised_points = denoised_tensor

    if tls_tensor.dim() == 3:
        tls_points = tls_tensor.squeeze(0)
    else:
        tls_points = tls_tensor

    rmse = torch.sqrt(F.mse_loss(denoised_points, tls_points))
    return rmse.item()


def calculate_denoising_rate(mls_tensor, denoised_tensor):
    """
    计算去噪率，输入为 Tensor，形状为 (batch_size, num_points, 3)。
    """
    if mls_tensor.dim() == 3:
        mls_points = mls_tensor.squeeze(0)  # 移除 batch 维度
    else:
        mls_points = mls_tensor

    if denoised_tensor.dim() == 3:
        denoised_points = denoised_tensor.squeeze(0)
    else:
        denoised_points = denoised_tensor

    original_num_points = mls_points.shape[0]
    denoised_num_points = denoised_points.shape[0]

    denoising_rate = (original_num_points - denoised_num_points) / original_num_points
    return denoising_rate


def preprocess_tensor(tls_tensor, target_num_points, output_dim):
    """
    上采样 TLS 点云张量并扩展其维度。

    参数:
    tls_tensor: torch.Tensor，形状为 [1, num_points, 3]。
    target_num_points: int，目标点数。
    output_dim: int，目标维度，默认值为 512。

    返回:
    torch.Tensor，形状为 [1, target_num_points, output_dim]。
    """
    # 上采样到目标点数
    tls_tensor_upsampled = F.interpolate(
        tls_tensor.permute(0, 2, 1), size=target_num_points, mode='linear', align_corners=False
    ).permute(0, 2, 1)

    # 使用线性变换扩展维度
    linear_layer = nn.Linear(3, output_dim)  # 定义线性层
    tls_upsampled_expanded = linear_layer(
        tls_tensor_upsampled.view(-1, 3))  # [batch_size * num_points, 3] -> [batch_size * num_points, output_dim]
    tls_upsampled_expanded = tls_upsampled_expanded.view(1, target_num_points,
                                                         output_dim)  # 调整为 [1, target_num_points, output_dim]

    return tls_upsampled_expanded


def train_and_evaluate_model(model, mls_tensor, tls_tensor, output_dim):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 150
    prev_loss = float('inf')  # 上一次的损失
    tolerance = 1e-6  # 损失变化的容忍度，用于提前停止

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 输入 MLS 点云，输出去噪后的点云
        output = model(mls_tensor)
        loss = F.mse_loss(output, tls_tensor)  # 计算损失
        print(f"Epoch {epoch} - Loss: {loss.item()}")

        # 如果损失值变化小于容忍度，则停止训练
        if abs(prev_loss - loss.item()) < tolerance:
            print(f"Early stopping at epoch {epoch}, Loss change is too small.")
            break

        prev_loss = loss.item()

        # 反向传播和优化步骤
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    # 使用训练好的模型进行去噪
    model.eval()
    with torch.no_grad():
        denoised_pcd = model(mls_tensor)

    # 计算 RMSE 和 去噪率
    rmse = calculate_rmse(denoised_pcd, tls_tensor)
    denoising_rate = calculate_denoising_rate(mls_tensor, denoised_pcd)
    return rmse, denoising_rate


# 可视化去噪结果的函数
def visualize_denoising(original_pcd, denoised_pcd):
    """
    将去噪去掉的点标记为红色，未去掉的点标记为蓝色。

    :param original_pcd: 原始点云数据 (o3d.geometry.PointCloud)
    :param denoised_pcd: 去噪后的点云数据 (o3d.geometry.PointCloud)
    """
    # 转换点云为 numpy 数组
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)

    # 创建一个布尔掩码，用于检查哪些点被保留
    mask = np.isin(original_points, denoised_points).all(axis=1)

    # 设置颜色：蓝色表示保留的点，红色表示去掉的点
    colors = np.zeros_like(original_points)  # 初始化颜色数组
    colors[mask] = [0, 0, 1]  # 蓝色
    colors[~mask] = [1, 0, 0]  # 红色

    # 应用颜色到原始点云
    original_pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([original_pcd])


# 加载点云文件
tls_pcd = load_las_as_o3d_point_cloud("autodl-tmp/Roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("autodl-tmp/Roof_MLS.las")

# 归一化点云
tls_pcd = normalize_point_cloud(tls_pcd)
mls_pcd = normalize_point_cloud(mls_pcd)

tls_tensor = point_cloud_to_tensor(tls_pcd)
mls_tensor = point_cloud_to_tensor(mls_pcd)

# 模型定义（请根据你的实际模型定义调整）
point_pro_nets_model = PointProNets()
geometric_semantic_model = GeometricSemanticFusion()
robust_cnn_model = RobustCNNMethod()

# 针对 PointProNets 模型

output_dim_1 = 512  # 设定输出维度
processed_tls_tensor_1 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_1)
print("Training and Evaluating PointProNets Model...")
output = point_pro_nets_model(mls_tensor)  # 获取模型输出
print(f"PointProNets output shape: {output.shape}")

# rmse1, denoising_rate1 = train_and_evaluate_model(point_pro_nets_model, mls_tensor, processed_tls_tensor_1, output_dim_1)
# print(f"PointProNets Model RMSE: {rmse1:.4f}, Denoising Rate: {denoising_rate1:.2f}%")


# 针对 GeometricSemanticFusion 模型

output_dim_2 = 3
print("Training and Evaluating GeometricSemanticFusion Model...")
output1 = geometric_semantic_model(mls_tensor)
print(f"GeometricSemanticFusion output shape: {output1.shape}")
processed_tls_tensor_2 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_2)
# 对于 GeometricSemanticFusion 模型，假设它不需要额外的上采样，直接使用 TLS 的目标
# rmse2, denoising_rate2 = train_and_evaluate_model(geometric_semantic_model, mls_tensor, processed_tls_tensor_2, output_dim_2)
# print(f"GeometricSemanticFusion Model RMSE: {rmse2:.4f}, Denoising Rate: {denoising_rate2:.2f}%")

# 针对 RobustCNNMethod 模型

output_dim_3 = 3
print("Training and Evaluating RobustCNNMethod Model...")
output2 = robust_cnn_model(mls_tensor)
print(f"RobustCNNMethod output shape: {output2.shape}")
processed_tls_tensor_3 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_3)
# 对于 RobustCNNMethod 模型，假设它需要不同的预处理（例如，不使用 `preprocess_tensor`）
# rmse3, denoising_rate3 = train_and_evaluate_model(robust_cnn_model, mls_tensor, processed_tls_tensor_3, output_dim_1)
# print(f"RobustCNNMethod Model RMSE: {rmse3:.4f}, Denoising Rate: {denoising_rate3:.2f}%")


