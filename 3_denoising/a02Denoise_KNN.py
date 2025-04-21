import open3d as o3d
import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import random

# Set random seed
random.seed(42)  # Set seed for Python random
np.random.seed(42)  # Set seed for NumPy

# ICP alignment function
def align_point_clouds(source_pcd, target_pcd, threshold=0.02, trans_init=np.eye(4)):
    # Estimate normals
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # ICP registration using point-to-plane method
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Apply the transformation
    aligned_pcd = source_pcd.transform(reg_p2l.transformation)
    return aligned_pcd

# KNN patch-based denoising
def knn_denoise_patch(point_cloud, k=25, distance_threshold=0.2):
    points = np.asarray(point_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    denoised_points = []
    for i, point in enumerate(points):
        avg_distance = np.mean(distances[i])
        if avg_distance < distance_threshold:
            denoised_points.append(point)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised

# KNN manifold-based denoising using curvature
def knn_denoise_manifold(point_cloud, k=25, curvature_threshold=0.2):
    points = np.asarray(point_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    denoised_points = []
    for i, point in enumerate(points):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        curvature = min(eigenvalues) / sum(eigenvalues)

        if curvature < curvature_threshold:
            denoised_points.append(point)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised

# Voxel-based denoising with KNN
def knn_denoise_voxel(point_cloud, voxel_size=0.05, k=25, distance_threshold=0.2):
    # Downsample to voxel grid
    voxel_grid = point_cloud.voxel_down_sample(voxel_size)
    voxel_centers = np.asarray(voxel_grid.points)

    denoised_points = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(voxel_centers)
    distances, _ = nbrs.kneighbors(voxel_centers)

    for i, center in enumerate(voxel_centers):
        avg_distance = np.mean(distances[i])
        if avg_distance < distance_threshold:
            denoised_points.append(center)

    point_cloud_denoised = o3d.geometry.PointCloud()
    point_cloud_denoised.points = o3d.utility.Vector3dVector(denoised_points)
    return point_cloud_denoised

# Read .las file and convert to Open3D point cloud
def read_las_to_o3d(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# Compute RMSE between denoised and reference points
def compute_rmse(denoised_points, reference_points):
    random_rotation = R.random().as_matrix()
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(denoised_points, axis=0)

    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

# Compute denoising rate
def compute_denoising_rate(original_point_cloud, denoised_point_cloud):
    original_points = np.asarray(original_point_cloud.points)
    denoised_points = np.asarray(denoised_point_cloud.points)
    removal_rate = (len(original_points) - len(denoised_points)) / len(original_points) * 100
    return removal_rate

# Visualize denoising results (removed = red, kept = blue)
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
    colors[~retained_mask] = [1, 0, 0]  # Red for removed
    colors[retained_mask] = [0, 0, 1]   # Blue for retained

    pcd_original.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")

# Load point clouds and run denoising
file_path = "D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las"
pcd = read_las_to_o3d(file_path)
file_path_1 = "D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las"
reference_pcd = read_las_to_o3d(file_path_1)

# Run denoising
pcd_denoised_patch = knn_denoise_patch(pcd)
pcd_denoised_manifold = knn_denoise_manifold(pcd)
pcd_denoised_voxel = knn_denoise_voxel(pcd)

# Save results
output_dir = "D:/E_2024_Thesis/Data/Output/Road/"
# Create folder if not exists
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

o3d.io.write_point_cloud(output_dir + "pcd_denoised_patch.ply", pcd_denoised_patch)
o3d.io.write_point_cloud(output_dir + "pcd_denoised_manifold.ply", pcd_denoised_manifold)
o3d.io.write_point_cloud(output_dir + "pcd_denoised_voxel.ply", pcd_denoised_voxel)

print("Point clouds have been saved to the output directory.")

# Visualize
visualize_denoising_fast(pcd, pcd_denoised_voxel)
visualize_denoising_fast(pcd, pcd_denoised_manifold)
visualize_denoising_fast(pcd, pcd_denoised_patch)

# registration
aligned_pcd_patch = align_point_clouds(pcd_denoised_patch, reference_pcd)
aligned_pcd_manifold = align_point_clouds(pcd_denoised_manifold, reference_pcd)
aligned_pcd_voxel = align_point_clouds(pcd_denoised_voxel, reference_pcd)
aligned_pcd_origin = align_point_clouds(pcd, reference_pcd)

# Calculate RMSE
rmse_patch = compute_rmse(np.asarray(aligned_pcd_patch.points), np.asarray(reference_pcd.points))
rmse_manifold = compute_rmse(np.asarray(aligned_pcd_manifold.points), np.asarray(reference_pcd.points))
rmse_voxel = compute_rmse(np.asarray(aligned_pcd_voxel.points), np.asarray(reference_pcd.points))
original_rmse = compute_rmse(np.asarray(aligned_pcd_origin.points), np.asarray(reference_pcd.points))

# Calculate Denoising Rate
denoising_rate_patch = compute_denoising_rate(pcd, pcd_denoised_patch)
denoising_rate_manifold = compute_denoising_rate(pcd, pcd_denoised_manifold)
denoising_rate_voxel = compute_denoising_rate(pcd, pcd_denoised_voxel)

# Output Result
print(f"Original RMSE: {original_rmse:.4f}")
print("KNN Patch RMSE:", rmse_patch)
print("KNN Patch Denoising Rate:", denoising_rate_patch, "%")
print("KNN Manifold RMSE:", rmse_manifold)
print("KNN Manifold Denoising Rate:", denoising_rate_manifold, "%")
print("KNN Voxel RMSE:", rmse_voxel)
print("KNN Voxel Denoising Rate:", denoising_rate_voxel, "%")
