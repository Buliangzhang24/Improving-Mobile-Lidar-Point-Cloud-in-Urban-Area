import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import laspy

# Load LAS file as Open3D PointCloud
def load_las_as_o3d_point_cloud(las_file_path):
    las = laspy.read(las_file_path)  # Read LAS file
    points = np.vstack((las.x, las.y, las.z)).transpose()  # Combine x,y,z coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Normalize point cloud (center and scale)
def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)  # Calculate center point
    points = points - centroid  # Center points
    scale = np.max(np.linalg.norm(points, axis=1))  # Calculate scale factor
    points = points / scale  # Scale points
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, centroid, scale  # Return normalized pcd, center, and scale

# Reverse normalization
def denormalize_point_cloud(pcd, centroid, scale):
    points = np.asarray(pcd.points)
    points = points * scale  # Reverse scaling
    points = points + centroid  # Reverse centering
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Voxel downsampling
def downsample_point_cloud(pcd, voxel_size=0.05):
    return pcd.voxel_down_sample(voxel_size)

# Downsample two point clouds to same size
def downsample_to_same_size(pcd1, pcd2, voxel_size=0.05):
    pcd1_downsampled = downsample_point_cloud(pcd1, voxel_size)
    pcd2_downsampled = downsample_point_cloud(pcd2, voxel_size)

    if len(pcd1_downsampled.points) != len(pcd2_downsampled.points):
        pcd2_downsampled = downsample_point_cloud(pcd2, voxel_size)

    return pcd1_downsampled, pcd2_downsampled

# PointNet model definition
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 3)  # Output denoised points

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # Final linear layer outputs point coordinates
        return x

# Chamfer distance (batch version)
def chamfer_distance_batch(pcd1, pcd2, batch_size=1024):
    total_dist = 0
    num_points = pcd1.shape[0]
    for i in range(0, num_points, batch_size):
        batch_pcd1 = pcd1[i:i + batch_size]
        batch_pcd2 = pcd2[i:i + batch_size]
        dist1, _ = torch.min(torch.cdist(batch_pcd1, batch_pcd2), dim=1)
        dist2, _ = torch.min(torch.cdist(batch_pcd2, batch_pcd1), dim=1)
        total_dist += torch.sum(dist1) + torch.sum(dist2)
    return total_dist / num_points

# Convert Open3D point cloud to PyTorch tensor
def point_cloud_to_tensor(pcd):
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)

# Calculate RMSE between point clouds
def calculate_rmse(pcd1, pcd2):
    num_points = min(len(pcd1), len(pcd2))

    idx1 = torch.randperm(len(pcd1))[:num_points]
    idx2 = torch.randperm(len(pcd2))[:num_points]

    pcd1 = pcd1[idx1]
    pcd2 = pcd2[idx2]

    dist = torch.norm(pcd1 - pcd2, dim=1)
    rmse = torch.sqrt(torch.mean(dist ** 2))
    return rmse

# Adjust point cloud sizes to match
def adjust_point_cloud_size(pcd1, pcd2):
    num_points = min(len(pcd1), len(pcd2))  # Make both same size
    pcd1_downsampled = pcd1[torch.randperm(len(pcd1))[:num_points]]
    pcd2_downsampled = pcd2[torch.randperm(len(pcd2))[:num_points]]
    return pcd1_downsampled, pcd2_downsampled

# Convert PyTorch tensor to Open3D point cloud
def tensor_to_o3d_point_cloud(tensor):
    points = tensor.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Visualize denoising results
def visualize_denoising_fast(pcd_original, pcd_denoised):
    original_points = np.asarray(pcd_original.points)
    denoised_points = np.asarray(pcd_denoised.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_denoised)

    retained_mask = np.zeros(len(original_points), dtype=bool)

    for i, point in enumerate(original_points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and np.linalg.norm(denoised_points[idx[0]] - point) <= 1e-6:
            retained_mask[i] = True

    colors = np.zeros_like(original_points)
    colors[~retained_mask] = [0, 0, 1]  # Removed points (blue)
    colors[retained_mask] = [1, 0, 0]  # Kept points (red)

    pcd_original.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")

# Tensor to point cloud conversion
def tensor_to_pointcloud(tensor):
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.numpy()
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(tensor)
    return pointcloud

# Calculate denoising rate
def compute_denoising_rate(original_pcd, denoised_pcd):
    if isinstance(original_pcd, o3d.geometry.PointCloud) and isinstance(denoised_pcd, o3d.geometry.PointCloud):
        original_points = np.asarray(original_pcd.points)
        denoised_points = np.asarray(denoised_pcd.points)
        removal_rate = (len(original_points) - len(denoised_points)) / len(original_points) * 100
        return removal_rate
    else:
        raise TypeError("Both inputs must be Open3D PointCloud objects.")

tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/TLS_Street.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/MLS_Street.las")

# Downsample point clouds to same size
tls_pcd_downsampled, mls_pcd_downsampled = downsample_to_same_size(tls_pcd, mls_pcd, voxel_size=0.05)

# Normalize point clouds
tls_pcd_downsampled, centroid_tls, scale_tls = normalize_point_cloud(tls_pcd_downsampled)
mls_pcd_downsampled, centroid_mls, scale_mls = normalize_point_cloud(mls_pcd_downsampled)

# Convert to PyTorch tensors
tls_tensor_downsampled = point_cloud_to_tensor(tls_pcd_downsampled)
mls_tensor_downsampled = point_cloud_to_tensor(mls_pcd_downsampled)

# Initialize PointNet model and optimizer
model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass: MLS point cloud in, denoised point cloud out
    output = model(mls_tensor_downsampled)

    # Calculate Chamfer distance loss
    loss = chamfer_distance_batch(output, tls_tensor_downsampled)

    # Backpropagation and optimize
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

# Denoise point cloud using trained model
model.eval()
denoised_mls_tensor = model(mls_tensor_downsampled)

# Convert tensors back to Open3D point clouds
denoised_mls_pcd = tensor_to_o3d_point_cloud(denoised_mls_tensor)
denoised_tls_pcd = tensor_to_o3d_point_cloud(tls_tensor_downsampled)

# Denormalize point cloud
denoised_mls_pcd1 = denormalize_point_cloud(denoised_mls_pcd, centroid_mls, scale_mls)

# Save denoised point clouds
output_dir = "D:/E_2024_Thesis/Data/Output/Road/"
o3d.io.write_point_cloud(output_dir + "mls_PointNet_Block.ply", denoised_mls_pcd)
# o3d.io.write_point_cloud(output_dir + "mls_pointnet_afterdenormal.ply", denoised_mls_pcd1)
# o3d.io.write_point_cloud(output_dir + "tls_pcd_pointnet.ply", denoised_tls_pcd)

# Visualize denoising results
visualize_denoising_fast(mls_pcd_downsampled, denoised_mls_pcd)