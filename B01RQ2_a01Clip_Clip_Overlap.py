import open3d as o3d
import numpy as np
import os

def load_mls_point_cloud(ply_file):
    """
    Load MLS point cloud from a PLY file.
    """
    return o3d.io.read_point_cloud(ply_file)

def load_tls_point_cloud(ply_file):
    """
    Load TLS point cloud from a PLY file.
    """
    tls_cloud = o3d.io.read_point_cloud(ply_file)
    tls_points = np.asarray(tls_cloud.points)  # Extract points from the cloud
    return tls_points

def extract_overlap_region(mls_cloud, tls_points, overlap_min, overlap_max):
    """
    Extract overlapping region of MLS and TLS point clouds based on bounding box.
    """
    # Extract points within the overlap bounding box for MLS
    mls_points = np.asarray(mls_cloud.points)
    mls_in_overlap = (mls_points >= overlap_min) & (mls_points <= overlap_max)
    mls_overlap_points = mls_points[np.all(mls_in_overlap, axis=1)]

    # Extract points within the overlap bounding box for TLS
    tls_in_overlap = (tls_points >= overlap_min) & (tls_points <= overlap_max)
    tls_overlap_points = tls_points[np.all(tls_in_overlap, axis=1)]

    return mls_overlap_points, tls_overlap_points

def save_point_clouds(mls_overlap_points, tls_overlap_points, mls_output_file, tls_output_file):
    """
    Save the overlapping MLS and TLS point clouds to output files.
    """
    # Delete existing files if they exist
    if os.path.exists(mls_output_file):
        os.remove(mls_output_file)
    if os.path.exists(tls_output_file):
        os.remove(tls_output_file)

    # Save MLS overlap points to PLY file
    mls_overlap_cloud = o3d.geometry.PointCloud()
    mls_overlap_cloud.points = o3d.utility.Vector3dVector(mls_overlap_points)
    o3d.io.write_point_cloud(mls_output_file, mls_overlap_cloud)

    # Save TLS overlap points to PLY file
    tls_overlap_cloud = o3d.geometry.PointCloud()
    tls_overlap_cloud.points = o3d.utility.Vector3dVector(tls_overlap_points)
    o3d.io.write_point_cloud(tls_output_file, tls_overlap_cloud)

def calculate_rmse(mls_points, tls_points):
    """
    Calculate RMSE (Root Mean Square Error) between MLS and TLS overlap points.
    """
    # Ensure that both point clouds have the same number of points
    min_points = min(len(mls_points), len(tls_points))
    mls_points = mls_points[:min_points]
    tls_points = tls_points[:min_points]

    # Calculate the squared differences and then the RMSE
    squared_diff = np.sum((mls_points - tls_points) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse

def extract_and_save_overlap(mls_ply, tls_ply, mls_output_file, tls_output_file):
    """
    Extract the overlapping region from MLS and TLS point clouds and save as new files.
    """
    # Load MLS and TLS point clouds
    mls_cloud = load_mls_point_cloud(mls_ply)
    tls_points = load_tls_point_cloud(tls_ply)

    # Get bounding box for MLS point cloud
    mls_bbox = mls_cloud.get_axis_aligned_bounding_box()
    tls_bbox_min = np.min(tls_points, axis=0)
    tls_bbox_max = np.max(tls_points, axis=0)

    # Define the overlap region
    overlap_min = np.maximum(mls_bbox.min_bound, tls_bbox_min)
    overlap_max = np.minimum(mls_bbox.max_bound, tls_bbox_max)

    # Extract the overlapping regions
    mls_overlap_points, tls_overlap_points = extract_overlap_region(mls_cloud, tls_points, overlap_min, overlap_max)

    # Save the overlapping point clouds
    save_point_clouds(mls_overlap_points, tls_overlap_points, mls_output_file, tls_output_file)

    # Calculate and print RMSE between MLS and TLS overlap points
    rmse = calculate_rmse(mls_overlap_points, tls_overlap_points)
    print(f"RMSE between MLS and TLS overlap points: {rmse:.4f}")

# Example usage
mls_ply = "D:/E_2024_Thesis/Data/roof/roof_MLS_1.ply"
tls_ply = "D:/E_2024_Thesis/Data/roof/roof_TLS_1.ply"
mls_output_file = "D:/E_2024_Thesis/Data/roof/MLS_overlap.ply"
tls_output_file = "D:/E_2024_Thesis/Data/roof/TLS_overlap.ply"

extract_and_save_overlap(mls_ply, tls_ply, mls_output_file, tls_output_file)

