import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import laspy

def read_las_to_o3d(file_path):
    """
    Read a LAS file and convert it to an Open3D point cloud
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def estimate_normals(points, k=20):
    """
    Estimate normals for each point using k-nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = []
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)  # Covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals.append(eigvecs[:, 0])  # Normal vector is the eigenvector with the smallest eigenvalue

    normals = np.array(normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def point_filter(noisy_points, ground_truth_normals, k=20, iterations=5):
    """
    Implement the PointFilter algorithm
    """
    denoised_points = noisy_points.copy()

    for _ in range(iterations):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(denoised_points)
        _, indices = nbrs.kneighbors(denoised_points)

        new_points = []
        for i in range(denoised_points.shape[0]):
            neighbors = denoised_points[indices[i]]
            normal = ground_truth_normals[i]
            # Project neighbors to the normal plane
            projected_points = neighbors - np.dot(neighbors - denoised_points[i], normal)[:, None] * normal
            new_points.append(np.mean(projected_points, axis=0))

        denoised_points = np.array(new_points)

    return denoised_points

def calculate_rmse(denoised_points, ground_truth_points):
    """
    Compute RMSE (Root Mean Square Error) by matching nearest neighbor points
    """
    # Use nearest neighbor search to match point clouds
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(ground_truth_points)
    # Find the nearest neighbor for each point in denoised_points
    distances, _ = neigh.kneighbors(denoised_points)

    # Compute RMSE
    rmse = np.sqrt(np.mean(distances**2))
    return rmse

def downsample_points(points, voxel_size):
    """
    Downsample the point cloud data using voxel grid
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

def calculate_noise_removal_rate(original_points, denoised_points):
    """
    Calculate noise removal rate as the percentage of points removed during denoising.

    Parameters:
    original_points (np.ndarray): Original noisy point cloud points.
    denoised_points (np.ndarray): Denoised point cloud points.

    Returns:
    float: Noise removal rate in percentage.
    """
    original_count = len(original_points)
    denoised_count = len(denoised_points)
    removed_points = original_count - denoised_count
    noise_removal_rate = (removed_points / original_count) * 100
    return noise_removal_rate

# Load point cloud data
print("Loading LAS point cloud data...")
file_path_mls = "D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las"
file_path_tls = "D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las"

noisy_pcd = read_las_to_o3d(file_path_mls)
ground_truth_pcd = read_las_to_o3d(file_path_tls)

noisy_points = np.asarray(noisy_pcd.points)
ground_truth_points = np.asarray(ground_truth_pcd.points)

# Step 1: Estimate normals
print("Estimating ground truth normals...")
ground_truth_normals = estimate_normals(ground_truth_points)

# Step 2: Run PointFilter algorithm
print("Running PointFilter 3_denoising...")
denoised_points = point_filter(noisy_points, ground_truth_normals)

# Step 3: Save denoised point cloud
print("Saving denoised point cloud...")
denoised_pcd = o3d.geometry.PointCloud()
denoised_pcd.points = o3d.utility.Vector3dVector(denoised_points)
o3d.io.write_point_cloud("D:/E_2024_Thesis/Data/Output/Road/PointFilter.ply", denoised_pcd)
# o3d.io.write_point_cloud("D:/E_2024_Thesis/Output/Roof_Denoised_HighRes.pcd", denoised_pcd)  # Save in .pcd format if needed

# Step 4: Calculate RMSE
voxel_size = 0.05
denoised_points_down = downsample_points(denoised_points, voxel_size)
ground_truth_points_down = downsample_points(np.asarray(ground_truth_points), voxel_size)

print("Calculating RMSE...")
rmse = calculate_rmse(denoised_points_down, ground_truth_points_down)
print(f"RMSE: {rmse}")
