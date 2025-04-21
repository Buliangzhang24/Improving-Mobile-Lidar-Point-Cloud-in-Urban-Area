import open3d as o3d

# Compute Hausdorff distance
def hausdorff_distance(mesh1, mesh2):
    # Convert mesh to point cloud
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=5000)
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=5000)

    # Compute point-to-point distance using Open3D
    dist1 = pcd1.compute_point_cloud_distance(pcd2)  # Distance from pcd1 to pcd2
    dist2 = pcd2.compute_point_cloud_distance(pcd1)  # Distance from pcd2 to pcd1

    # Get max distance
    hd1 = max(dist1)  # Max distance from mesh1 to mesh2
    hd2 = max(dist2)  # Max distance from mesh2 to mesh1

    return max(hd1, hd2)  # Final Hausdorff distance (max of both directions)

# Load mesh files
mesh_tls = o3d.io.read_triangle_mesh("D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D_CloudCompare/Roof_TLS - Cloud.ply")
mesh_mls = o3d.io.read_triangle_mesh("D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D/KNN_patch.ply")

# Compute Hausdorff distance
hd_value = hausdorff_distance(mesh_tls, mesh_mls)
print(f"The Hausdorff distance between TLS and MLS mesh is: {hd_value}")
