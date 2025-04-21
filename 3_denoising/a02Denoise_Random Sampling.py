import open3d as o3d
import laspy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

# Load LAS file and convert to Open3D point cloud
def load_las_as_o3d_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# Compute RMSE (Root Mean Square Error)
def compute_rmse(denoised_points, reference_points):
    # Use nearest neighbor search to match each denoised point to the closest reference point
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # Calculate the squared distance between each denoised point and its nearest reference point
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse


# Compute denoising rate
def compute_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)

    # Calculate how many points are kept after denoising
    denoising_rate = len(denoised_points) / len(original_points) * 100
    return denoising_rate


# RANSAC denoising
def ransac_denoise(pcd, distance_threshold=0.05):
    # Fit a plane model and remove outliers using RANSAC
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3,
                                                    num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inlier_indices)
    return inlier_cloud


# Bayesian denoising
def bayesian_denoise(pcd, prior_prob_real=0.9, prior_prob_noise=0.1):
    points = np.asarray(pcd.points)
    # Use Bayesian estimation to identify noisy points
    # This is just a simple example; real Bayesian methods need more modeling
    # For simplicity, assume points with z < 0 are noise
    filtered_points = points[points[:, 2] > 0]
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return denoised_pcd


# Density-based denoising
def density_denoise(pcd, bandwidth=0.1):
    points = np.asarray(pcd.points)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    log_density = kde.score_samples(points)
    threshold = np.percentile(log_density, 20)  # Keep 80% with highest density
    denoised_points = points[log_density > threshold]
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(denoised_points)
    return denoised_pcd


def visualize_denoising_fast(pcd_original, pcd_denoised):
    # Get points from original and denoised point clouds
    original_points = np.asarray(pcd_original.points)
    denoised_points = np.asarray(pcd_denoised.points)

    # Build a KD-tree from denoised point cloud for faster search
    kdtree = o3d.geometry.KDTreeFlann(pcd_denoised)

    # Create a mask to mark retained points
    retained_mask = np.zeros(len(original_points), dtype=bool)

    # Check if each original point exists in the denoised cloud
    for i, point in enumerate(original_points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and np.linalg.norm(denoised_points[idx[0]] - point) <= 1e-6:
            retained_mask[i] = True

    # Set colors: removed points = blue, kept points = red
    colors = np.zeros_like(original_points)
    colors[~retained_mask] = [0, 0, 1]  # Removed points: blue
    colors[retained_mask] = [1, 0, 0]   # Kept points: red

    # Add color to the original point cloud
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    # Show the result
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")


# Loading MLS and TLS
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las")

# Denoising
mls_denoised_ransac = ransac_denoise(mls_pcd)
mls_denoised_bayes = bayesian_denoise(mls_pcd)
mls_denoised_density = density_denoise(mls_pcd)

visualize_denoising_fast(mls_pcd, mls_denoised_ransac)
visualize_denoising_fast(mls_pcd, mls_denoised_bayes)
visualize_denoising_fast(mls_pcd, mls_denoised_density)

output_dir = "D:/E_2024_Thesis/Data/Output/Road/"
# if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

o3d.io.write_point_cloud(output_dir + "mls_ransac.ply", mls_denoised_ransac)
o3d.io.write_point_cloud(output_dir + "mls_bayes.ply", mls_denoised_bayes)
o3d.io.write_point_cloud(output_dir + "mls_density.ply", mls_denoised_density)

print("Point clouds have been saved to the output directory.")
