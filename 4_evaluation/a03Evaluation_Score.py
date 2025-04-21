import numpy as np
import open3d as o3d
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff

def load_point_cloud(pcd_path):
    """
    Load point cloud data
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd


def downsample_points(points, voxel_size):
    """
    Downsample point cloud data
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)


def compute_rmse(mls_points, tls_points, random_seed=42):
    # Set random seed
    np.random.seed(random_seed)

    # Generate a fixed random rotation matrix
    random_rotation = R.random(random_seed).as_matrix()

    # Use the input MLS point cloud data
    denoised_points = np.asarray(mls_points)
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(
        denoised_points, axis=0)

    reference_points = np.asarray(tls_points)

    # Ensure point cloud is a 2D array (n_samples, n_features)
    if denoised_points.ndim == 3:
        denoised_points = denoised_points.reshape(-1, denoised_points.shape[-1])

    if reference_points.ndim == 3:
        reference_points = reference_points.reshape(-1, reference_points.shape[-1])

    # Use Nearest Neighbors to find closest reference point for each denoised point
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # Compute RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse


def compute_normal_consistency(tls_normals, mls_normals, threshold=1):
    """
    Compute normal consistency using cosine similarity between normals.

    Parameters:
    tls_normals: Normals from TLS point cloud
    mls_normals: Normals from MLS point cloud
    threshold: Cosine similarity threshold, default is 1 (exact match)

    Returns:
    normal_consistency: Score between 0 and 1
    """

    # Use cKDTree to find nearest TLS normal for each MLS normal
    tree_tls = cKDTree(tls_normals)
    _, indices = tree_tls.query(mls_normals, k=1)

    # Get corresponding TLS normals
    nearest_tls_normals = tls_normals[indices]

    # Compute cosine similarity
    cos_sim = np.sum(nearest_tls_normals * mls_normals, axis=1)
    cos_sim = cos_sim / (np.linalg.norm(nearest_tls_normals, axis=1) * np.linalg.norm(mls_normals, axis=1))

    # Calculate consistency
    consistent_normals = np.sum(cos_sim >= threshold)
    normal_consistency = consistent_normals / len(cos_sim)

    return normal_consistency


def compute_completeness(tls_pcd, mls_pcd):
    """
    Compute point cloud completeness
    """
    # Use the ratio of point counts as a simple measure
    tls_points = np.asarray(tls_pcd.points)
    mls_points = np.asarray(mls_pcd.points)

    completeness = len(mls_points) / len(tls_points)
    return completeness


def compute_score(rmse, normal_consistency, completeness, rmse_weight=0.4, normal_consistency_weight=0.4,
                  completeness_weight=0.2):
    """
    Compute overall evaluation score
    """
    score = (rmse_weight * (1 / rmse)) + (normal_consistency_weight * normal_consistency) + (
                completeness_weight * completeness)
    return score


def downsample_points_and_normals(points, voxel_size):
    """
    Downsample point cloud and estimate normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample
    down_pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(down_pcd.points)

    # Estimate normals
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(down_pcd.normals)

    return down_points, normals


def evaluate_point_cloud(tls_pcd_path, mls_pcd_path, voxel_size=0.01):
    """
    Evaluate point clouds by computing RMSE, normal consistency, completeness, and overall score
    """
    tls_pcd = load_point_cloud(tls_pcd_path)
    mls_pcd = load_point_cloud(mls_pcd_path)

    if len(tls_pcd.points) == 0 or len(mls_pcd.points) == 0:
        print("Error: One of the point clouds is empty.")
        return None, None, None, None

    # Extract point coordinates
    tls_points = np.asarray(tls_pcd.points)
    mls_points = np.asarray(mls_pcd.points)

    # Downsample and estimate normals
    tls_points1, tls_normals = downsample_points_and_normals(tls_points, voxel_size)
    mls_points1, mls_normals = downsample_points_and_normals(mls_points, voxel_size)

    # Compute RMSE
    rmse = compute_rmse(mls_points, tls_points)

    # Compute normal consistency
    normal_consistency = compute_normal_consistency(tls_normals, mls_normals)

    # Compute completeness
    completeness = compute_completeness(tls_pcd, mls_pcd)

    # Compute overall score
    score = compute_score(rmse, normal_consistency, completeness)

    return rmse, normal_consistency, completeness, score


# File paths
tls_pcd_path = "D:/E_2024_Thesis/Data/Input/Street/TLS_Block.ply"  # Path to TLS point cloud
mls_pcd_path = "D:/E_2024_Thesis/Data/Output/Road/CNN/Block_PointProNets.ply"  # Path to denoised MLS point cloud

# Run evaluation
rmse, normal_consistency, completeness, score = evaluate_point_cloud(tls_pcd_path, mls_pcd_path)

# Output results
print(f"RMSE: {rmse}")
print(f"Normal Consistency: {normal_consistency}")
print(f"Completeness: {completeness}")
print(f"Overall Score: {score}")
