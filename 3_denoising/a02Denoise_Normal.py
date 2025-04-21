import open3d as o3d
import numpy as np
import laspy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R


# Load LAS file as Open3D format point cloud
def load_las_as_o3d_point_cloud(file_path):
    # Use laspy to read .las file
    las_data = laspy.read(file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd


# Guided filtering for denoising (Zhou et al., 2022)
def guided_filtering(pcd, iterations=5, filter_strength=0.1):
    # Create KDTree for neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)

    for _ in range(iterations):
        filtered_normals = normals.copy()
        for i in range(len(normals)):
            # Search for 10 nearest neighbors
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
            avg_normal = np.mean(normals[neighbors], axis=0)
            filtered_normals[i] = (1 - filter_strength) * normals[i] + filter_strength * avg_normal
        normals = filtered_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# Guided filtering with Gaussian pyramid
def guided_filtering_with_gaussian_pyramid(pcd, iterations=5, filter_strength=0.1, num_levels=3):
    # Create KDTree for neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)

    for level in range(num_levels):
        scale_factor = 2 ** level
        for _ in range(iterations):
            filtered_normals = normals.copy()
            for i in range(len(normals)):
                # Reduce neighborhood size based on scale
                neighbor_count = int(10 / scale_factor) + 1
                [_, neighbors, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], neighbor_count)

                avg_normal = np.mean(normals[neighbors], axis=0)
                filtered_normals[i] = (1 - filter_strength) * normals[i] + filter_strength * avg_normal
            normals = filtered_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# Iterative bilateral filtering (Hurtado et al., 2023)
def bilateral_filtering(pcd, iterations=5, spatial_sigma=0.5, normal_sigma=0.1):
    # Create KDTree for neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    for _ in range(iterations):
        filtered_normals = normals.copy()
        for i in range(len(normals)):
            # Search for 10 nearest neighbors
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(points[i], 10)
            neighbor_normals = normals[neighbors]
            neighbor_points = points[neighbors]

            weights_spatial = np.exp(-np.linalg.norm(neighbor_points - points[i], axis=1) ** 2 / (2 * spatial_sigma ** 2))
            weights_normal = np.exp(-np.linalg.norm(neighbor_normals - normals[i], axis=1) ** 2 / (2 * normal_sigma ** 2))
            weights = weights_spatial * weights_normal

            filtered_normals[i] = np.sum(weights[:, None] * neighbor_normals, axis=0) / np.sum(weights)
        normals = filtered_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# Anisotropic diffusion with curvature
def anisotropic_diffusion_with_curvature(pcd, iterations=5, diffusion_factor=0.1):
    # Create KDTree for neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    for _ in range(iterations):
        updated_normals = normals.copy()
        for i in range(len(normals)):
            # Search for 10 nearest neighbors
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(points[i], 10)

            neighbor_points = points[neighbors]
            diff_vectors = neighbor_points - points[i]
            curvatures = np.linalg.norm(diff_vectors, axis=1)  # Local curvature estimation
            curvature_factor = np.exp(-curvatures ** 2)  # Weight based on curvature

            for j, neighbor_idx in enumerate(neighbors):
                diff = normals[neighbor_idx] - normals[i]
                weight = curvature_factor[j] * np.exp(-np.linalg.norm(points[neighbor_idx] - points[i]) ** 2)
                updated_normals[i] += diffusion_factor * weight * diff

            # Normalize normal vector
            updated_normals[i] = updated_normals[i] / np.linalg.norm(updated_normals[i])
        normals = updated_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# Anisotropic diffusion (normal smoothing)
def anisotropic_diffusion(pcd, iterations=5, diffusion_factor=0.1):
    # Create KDTree for neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    for _ in range(iterations):
        updated_normals = normals.copy()
        for i in range(len(normals)):
            # Search for 10 nearest neighbors
            [_, neighbors, _] = pcd_tree.search_knn_vector_3d(points[i], 10)
            for neighbor_idx in neighbors:
                diff = normals[neighbor_idx] - normals[i]
                weight = np.exp(-np.linalg.norm(points[neighbor_idx] - points[i]) ** 2)
                updated_normals[i] += diffusion_factor * weight * diff
            updated_normals[i] = updated_normals[i] / np.linalg.norm(updated_normals[i])  # Normalize
        normals = updated_normals
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# Compute RMSE
def compute_rmse(denoised_pcd, reference_pcd, random_seed=42):
    # Fix random seed
    np.random.seed(random_seed)

    # Optional: apply small random rotation/translation to avoid perfect alignment
    random_rotation = R.random().as_matrix()
    denoised_points = np.asarray(denoised_pcd.points)
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(denoised_points, axis=0)

    reference_points = np.asarray(reference_pcd.points)

    # Find nearest point in reference for each denoised point
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # Compute RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse


# Compute denoising rate
def compute_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)
    removal_rate = (len(original_points) - len(denoised_points)) / len(original_points) * 100
    return removal_rate


# Visualize denoising effect (highlight removed points)
def visualize_denoising_fast(pcd_original, pcd_denoised):
    original_points = np.asarray(pcd_original.points)
    denoised_points = np.asarray(pcd_denoised.points)

    # Build k-d tree for fast lookup
    kdtree = o3d.geometry.KDTreeFlann(pcd_denoised)

    # Initialize mask for retained points
    retained_mask = np.zeros(len(original_points), dtype=bool)

    # Check if each original point exists in denoised version
    for i, point in enumerate(original_points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and np.linalg.norm(denoised_points[idx[0]] - point) <= 1e-6:
            retained_mask[i] = True

    # Create color array
    colors = np.zeros_like(original_points)
    colors[~retained_mask] = [1, 0, 0]  # Red for removed points
    colors[retained_mask] = [0, 0, 1]   # Blue for retained points

    # Assign colors to point cloud
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    # Show result
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")


# Load point clouds
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las")
output_dir = "D:/E_2024_Thesis/Data/Output/Roof/"

# Estimate normals
tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))



# Calculate RMSE
original_rmse = compute_rmse(mls_pcd, tls_pcd)
print(f"Original RMSE: {original_rmse:.4f}")

denoised_pcd_guided = guided_filtering_with_gaussian_pyramid(mls_pcd)
o3d.io.write_point_cloud(output_dir + "mls_guided.ply", denoised_pcd_guided)

denoised_pcd_bilateral = bilateral_filtering(mls_pcd)
o3d.io.write_point_cloud(output_dir + "mls_bilateral.ply", denoised_pcd_bilateral)

denoised_pcd_anisotropic = anisotropic_diffusion_with_curvature(mls_pcd)
o3d.io.write_point_cloud(output_dir + "mls_anisotropic.ply", denoised_pcd_anisotropic)

