import open3d as o3d
import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans
import laspy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# Load LAS file as Open3D point cloud object
def load_las_as_o3d_point_cloud(file_path):
    # Use laspy to load .las file
    las_data = laspy.read(file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd

# Manifold reconstruction denoising (based on KNN and average)
def manifold_reconstruction_denoising(pcd, num_neighbors=20):
    # Compute manifold features using KNN
    pcd_points = np.asarray(pcd.points)
    tree = scipy.spatial.KDTree(pcd_points)

    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=num_neighbors)
        neighbor_points = pcd_points[neighbors[1], :]
        denoised_point = np.mean(neighbor_points, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd

# K-means clustering + manifold denoising
def kmeans_statistical_manifold_denoising(pcd, num_clusters=5):
    pcd_points = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pcd_points)

    # Denoise based on cluster assignment
    denoised_points = []
    for i in range(num_clusters):
        cluster_points = pcd_points[kmeans.labels_ == i]
        cluster_center = np.mean(cluster_points, axis=0)
        denoised_points.extend([cluster_center] * len(cluster_points))

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd

# Manifold-based distance truncation denoising
def manifold_distance_truncation_denoising(pcd, distance_threshold=0.05):
    pcd_points = np.asarray(pcd.points)
    tree = scipy.spatial.KDTree(pcd_points)

    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=20)
        neighbor_points = pcd_points[neighbors[1], :]

        distances = np.linalg.norm(neighbor_points - point, axis=1)
        close_neighbors = neighbor_points[distances < distance_threshold]

        # Use mean of nearby points
        denoised_point = np.mean(close_neighbors, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd

# Fluid-inspired denoising (smooth using neighborhood mean)
def fluid_inspired_denoising(pcd):
    pcd_points = np.asarray(pcd.points)

    # Smooth points based on neighbor mean (fluid-like)
    tree = scipy.spatial.KDTree(pcd_points)
    denoised_points = []
    for point in pcd_points:
        neighbors = tree.query(point, k=20)
        neighbor_points = pcd_points[neighbors[1], :]

        # Use average of neighbors
        denoised_point = np.mean(neighbor_points, axis=0)
        denoised_points.append(denoised_point)

    pcd.points = o3d.utility.Vector3dVector(np.array(denoised_points))
    return pcd

# Compute RMSE between denoised point cloud and reference
def compute_rmse(denoised_pcd, reference_pcd):
    # Optional: apply slight random rotation/translation
    random_rotation = R.random().as_matrix()
    denoised_points = np.asarray(denoised_pcd.points)
    denoised_points = np.dot(denoised_points - np.mean(denoised_points, axis=0), random_rotation) + np.mean(
        denoised_points, axis=0)

    reference_points = np.asarray(reference_pcd.points)

    # Find nearest point in reference for each denoised point
    nbrs = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, indices = nbrs.kneighbors(denoised_points)

    # Compute RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

# Compute denoising rate (how many points removed)
def compute_denoising_rate(original_pcd, denoised_pcd):
    original_points = np.asarray(original_pcd.points)
    denoised_points = np.asarray(denoised_pcd.points)
    removal_rate = (len(original_points) - len(denoised_points)) / len(original_points) * 100
    return removal_rate

# Fast visualization of denoising results (red = removed, blue = kept)
def visualize_denoising_fast(pcd_original, pcd_denoised):
    # Get points from original and denoised point clouds
    original_points = np.asarray(pcd_original.points)
    denoised_points = np.asarray(pcd_denoised.points)

    # Build k-d tree for denoised point cloud
    kdtree = o3d.geometry.KDTreeFlann(pcd_denoised)

    # Mask to check which points are retained
    retained_mask = np.zeros(len(original_points), dtype=bool)

    # Check each point if it's in the denoised version
    for i, point in enumerate(original_points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and np.linalg.norm(denoised_points[idx[0]] - point) <= 1e-6:
            retained_mask[i] = True

    # Assign colors: red = removed, blue = retained
    colors = np.zeros_like(original_points)
    colors[~retained_mask] = [1, 0, 0]  # removed
    colors[retained_mask] = [0, 0, 1]   # retained

    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    # Show result
    o3d.visualization.draw_geometries([pcd_original], window_name="Denoising Visualization")

# Load data
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/TLS_Block.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/Input/Street/MLS_Block.las")

# Select denoising method and apply
denoised_mls_reconstruction = manifold_reconstruction_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_reconstruction)
denoised_mls_statistical = kmeans_statistical_manifold_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_statistical)
denoised_mls_truncation = manifold_distance_truncation_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_truncation)
denoised_mls_fluid = fluid_inspired_denoising(mls_pcd)
#visualize_denoising_fast(mls_pcd, denoised_mls_fluid)

# Save denoised point clouds
output_dir = "D:/E_2024_Thesis/Data/Output/Road/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

o3d.io.write_point_cloud(output_dir + "mls_reconstruction.ply", denoised_mls_reconstruction)
o3d.io.write_point_cloud(output_dir + "mls_statistical.ply", denoised_mls_statistical)
o3d.io.write_point_cloud(output_dir + "mls_truncation.ply", denoised_mls_truncation)
o3d.io.write_point_cloud(output_dir + "mls_fluid.ply", denoised_mls_fluid)


