import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import laspy


# Load LAS file as Open3D point cloud
def load_las_as_o3d_point_cloud(las_file_path):
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# Normalize point cloud (center and scale)
def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# Calculate RMSE between point clouds
def calculate_rmse(predicted_pcd, ground_truth_pcd):
    predicted_points = np.asarray(predicted_pcd.points)
    ground_truth_points = np.asarray(ground_truth_pcd.points)

    # Ensure point clouds have same size
    assert predicted_points.shape == ground_truth_points.shape, "Point cloud size mismatch"

    # Calculate RMSE
    error = np.linalg.norm(predicted_points - ground_truth_points, axis=1)
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse


# Calculate denoising rate
def calculate_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)
    denoising_rate = len(denoised_points) / len(original_points) * 100
    return denoising_rate


# PointProNets (PointNet + CNN)
class PointProNets(nn.Module):
    def __init__(self):
        super(PointProNets, self).__init__()
        # PointNet part
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 3)  # Output denoised points

        # CNN part
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # PointNet forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        # CNN forward pass
        x = x.transpose(1, 2)  # Reshape for 1D conv
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        return x.transpose(1, 2)  # Reshape back


# Geometric and semantic fusion model
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


# Robust CNN model
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


# Convert point cloud to tensor with batch dimension
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dim


# Calculate RMSE between tensors
def calculate_rmse(denoised_tensor, tls_tensor):
    """
    Calculate RMSE between tensors of shape (batch_size, num_points, 3)
    """
    if denoised_tensor.dim() == 3:
        denoised_points = denoised_tensor.squeeze(0)  # Remove batch dim
    else:
        denoised_points = denoised_tensor

    if tls_tensor.dim() == 3:
        tls_points = tls_tensor.squeeze(0)
    else:
        tls_points = tls_tensor

    rmse = torch.sqrt(F.mse_loss(denoised_points, tls_points))
    return rmse.item()


# Calculate denoising rate between tensors
def calculate_denoising_rate(mls_tensor, denoised_tensor):
    """
    Calculate denoising rate between tensors of shape (batch_size, num_points, 3)
    """
    if mls_tensor.dim() == 3:
        mls_points = mls_tensor.squeeze(0)
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


# Upsample and expand tensor dimensions
def preprocess_tensor(tls_tensor, target_num_points, output_dim):
    """
    Upsample TLS tensor and expand dimensions

    Args:
    tls_tensor: torch.Tensor of shape [1, num_points, 3]
    target_num_points: int, target number of points
    output_dim: int, target dimension (default 512)

    Returns:
    torch.Tensor of shape [1, target_num_points, output_dim]
    """
    # Upsample to target points
    tls_tensor_upsampled = F.interpolate(
        tls_tensor.permute(0, 2, 1), size=target_num_points, mode='linear', align_corners=False
    ).permute(0, 2, 1)

    # Expand dimensions
    linear_layer = nn.Linear(3, output_dim)
    tls_upsampled_expanded = linear_layer(
        tls_tensor_upsampled.view(-1, 3))
    tls_upsampled_expanded = tls_upsampled_expanded.view(1, target_num_points, output_dim)

    return tls_upsampled_expanded


# Train and evaluate model
def train_and_evaluate_model(model, mls_tensor, tls_tensor, output_dim):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 150
    prev_loss = float('inf')
    tolerance = 1e-6  # Early stopping threshold

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(mls_tensor)
        loss = F.mse_loss(output, tls_tensor)
        print(f"Epoch {epoch} - Loss: {loss.item()}")

        if abs(prev_loss - loss.item()) < tolerance:
            print(f"Early stopping at epoch {epoch}, Loss change too small")
            break

        prev_loss = loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        denoised_pcd = model(mls_tensor)

    rmse = calculate_rmse(denoised_pcd, tls_tensor)
    denoising_rate = calculate_denoising_rate(mls_tensor, denoised_pcd)
    return rmse, denoising_rate


# Visualize denoising results
def visualize_denoising(original_pcd, denoised_pcd):
    """
    Visualize denoising - removed points in red, kept points in blue

    :param original_pcd: Original point cloud (o3d.geometry.PointCloud)
    :param denoised_pcd: Denoised point cloud (o3d.geometry.PointCloud)
    """
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)

    # Create mask for kept points
    mask = np.isin(original_points, denoised_points).all(axis=1)

    # Set colors: blue for kept, red for removed
    colors = np.zeros_like(original_points)
    colors[mask] = [0, 0, 1]  # Blue
    colors[~mask] = [1, 0, 0]  # Red

    original_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([original_pcd])
# Load point cloud files
tls_pcd = load_las_as_o3d_point_cloud("autodl-tmp/Roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("autodl-tmp/Roof_MLS.las")

# Normalize point clouds
tls_pcd = normalize_point_cloud(tls_pcd)
mls_pcd = normalize_point_cloud(mls_pcd)

# Convert point clouds to tensors
tls_tensor = point_cloud_to_tensor(tls_pcd)
mls_tensor = point_cloud_to_tensor(mls_pcd)

# Model definitions (please adjust based on your actual models)
point_pro_nets_model = PointProNets()
geometric_semantic_model = GeometricSemanticFusion()
robust_cnn_model = RobustCNNMethod()

# For the PointProNets model

output_dim_1 = 512  # Set output dimension
processed_tls_tensor_1 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_1)
print("Training and Evaluating PointProNets Model...")
output = point_pro_nets_model(mls_tensor)  # Get model output
print(f"PointProNets output shape: {output.shape}")

rmse1, denoising_rate1 = train_and_evaluate_model(point_pro_nets_model, mls_tensor, processed_tls_tensor_1, output_dim_1)
print(f"PointProNets Model RMSE: {rmse1:.4f}, Denoising Rate: {denoising_rate1:.2f}%")

# For the GeometricSemanticFusion model

output_dim_2 = 3
print("Training and Evaluating GeometricSemanticFusion Model...")
output1 = geometric_semantic_model(mls_tensor)
print(f"GeometricSemanticFusion output shape: {output1.shape}")
processed_tls_tensor_2 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_2)
# For this model, assume it does not need upsampling, use TLS target directly
# rmse2, denoising_rate2 = train_and_evaluate_model(geometric_semantic_model, mls_tensor, processed_tls_tensor_2, output_dim_2)
# print(f"GeometricSemanticFusion Model RMSE: {rmse2:.4f}, Denoising Rate: {denoising_rate2:.2f}%")

# For the RobustCNNMethod model

output_dim_3 = 3
print("Training and Evaluating RobustCNNMethod Model...")
output2 = robust_cnn_model(mls_tensor)
print(f"RobustCNNMethod output shape: {output2.shape}")
processed_tls_tensor_3 = preprocess_tensor(tls_tensor, mls_tensor.size(1), output_dim_3)
# For this model, assume it needs different preprocessing (e.g., not using `preprocess_tensor`)
# rmse3, denoising_rate3 = train_and_evaluate_model(robust_cnn_model, mls_tensor, processed_tls_tensor_3, output_dim_1)
# print(f"RobustCNNMethod Model RMSE: {rmse3:.4f}, Denoising Rate: {denoising_rate3:.2f}%")

