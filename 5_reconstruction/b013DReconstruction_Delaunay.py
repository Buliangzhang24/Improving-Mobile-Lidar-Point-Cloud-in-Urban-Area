import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

def fit_best_plane(points):
    """
    Fit a plane using least squares
    """
    # Create matrix A and B
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]

    # Solve linear system Ax = B
    plane_params = np.linalg.lstsq(A, B, rcond=None)[0]

    return plane_params


def project_to_plane(points, plane_params):
    """
    Project point cloud onto the fitted plane
    """
    a, b, c = plane_params
    # Calculate normal vector
    normal = np.array([a, b, -1])
    normal = normal / np.linalg.norm(normal)  # Normalize

    # Project points to the plane
    projected_points = points - np.dot(points - points.mean(axis=0), normal)[:, np.newaxis] * normal
    return projected_points


def delaunay_triangulation_2d(points_2d):
    """
    Perform 2D Delaunay triangulation
    """
    delaunay = Delaunay(points_2d)
    return delaunay.simplices


def apply_triangulation_to_3d(triangles, points_3d, plane_params):
    """
    Map 2D triangulation back to 3D space
    """
    # Restore 3D coordinates for each triangle
    triangles_3d = []
    for tri in triangles:
        pts_3d = points_3d[tri]
        triangles_3d.append(pts_3d)
    return triangles_3d


def filter_large_edges(points_2d, delaunay_result, max_edge_length):
    """
    Filter out triangles with long edges
    """
    filtered_triangles = []
    for tri in delaunay_result:
        # Compute all three edge lengths
        edges = [np.linalg.norm(points_2d[tri[i]] - points_2d[tri[(i + 1) % 3]]) for i in range(3)]
        if all(edge <= max_edge_length for edge in edges):
            filtered_triangles.append(tri)
    return filtered_triangles


# Read point cloud
pcd = o3d.io.read_point_cloud("D:/E_2024_Thesis/Data/Output/Roof/PointCloud/Normal/mls_guided.ply")
points = np.asarray(pcd.points)

# Step 1: Fit best plane
plane_params = fit_best_plane(points)

# Step 2: Project to the plane
projected_points = project_to_plane(points, plane_params)

# Set maximum edge length
max_edge_length = 2

# Step 3: Perform 2D Delaunay triangulation
delaunay_result = delaunay_triangulation_2d(projected_points[:, :2])

# Step 3.1: Filter out long-edge triangles
filtered_delaunay_result = filter_large_edges(projected_points[:, :2], delaunay_result, max_edge_length)

# Step 4: Apply triangulation result back to 3D
triangles_3d = apply_triangulation_to_3d(filtered_delaunay_result, points, plane_params)

# Check if triangles are available
if len(filtered_delaunay_result) == 0:
    print("Warning: No triangles after filtering. Try adjusting the max_edge_length or increasing the sampling.")
else:
    # Create mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(filtered_delaunay_result)

# Save mesh
output_path = "D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D/TOP/Normal_guided2.ply"
o3d.io.write_triangle_mesh(output_path, mesh)

# Visualize
o3d.visualization.draw_geometries([mesh])
