import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# 加载 LAS 点云为 Open3D PointCloud 对象
def load_las_as_o3d_point_cloud(las_file_path):
    pcd = o3d.io.read_point_cloud(las_file_path)
    return pcd


# 标准化和归一化函数
def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 定义 PointNet 模型
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 3)  # 输出去噪后的点云

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # 最后通过线性层将每个点的坐标输出
        return x


# Chamfer 距离损失函数
def chamfer_distance(pcd1, pcd2):
    dist1, _ = torch.min(torch.cdist(pcd1, pcd2), dim=1)
    dist2, _ = torch.min(torch.cdist(pcd2, pcd1), dim=1)
    return torch.mean(dist1) + torch.mean(dist2)


# 将 Open3D 点云转换为 PyTorch Tensor
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)


# 将 PyTorch Tensor 转换为 Open3D 点云
def tensor_to_o3d_point_cloud(tensor):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tensor.numpy()))


# 加载点云文件
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

# 对点云进行标准化和归一化
tls_pcd = normalize_point_cloud(tls_pcd)
mls_pcd = normalize_point_cloud(mls_pcd)

# 转换点云为 PyTorch Tensor
tls_tensor = point_cloud_to_tensor(tls_pcd)  # TLS 点云
mls_tensor = point_cloud_to_tensor(mls_pcd)  # MLS 点云（带噪声）

# 初始化 PointNet 模型和优化器
model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 输入 MLS 点云，输出去噪后的点云
    output = model(mls_tensor)  # 获取去噪后的点云

    # 计算损失：Chamfer 距离
    loss = chamfer_distance(output, tls_tensor)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 每100次打印一次损失值
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

# 使用训练好的模型进行去噪
model.eval()
with torch.no_grad():
    denoised_pcd = model(mls_tensor)

# 转换为 Open3D 点云对象
denoised_o3d_pcd = tensor_to_o3d_point_cloud(denoised_pcd)

# 可视化去噪后的点云
o3d.visualization.draw_geometries([denoised_o3d_pcd])
