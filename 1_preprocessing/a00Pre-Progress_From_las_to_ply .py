import laspy
import open3d as o3d
import numpy as np

def convert_las_to_ply(input_las, output_ply):
    """
    Convert a .las file to .ply format and visualize the point cloud.

    Parameters:
    - input_las (str): Path to the input .las file.
    - output_ply (str): Path to save the output .ply file.
    """
    # Read the LAS file using laspy
    las = laspy.read(input_las)

    # Extract x, y, z coordinates
    x = las.x
    y = las.y
    z = las.z
    points = np.vstack((x, y, z)).T

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Step 1: Check if the point cloud is loaded successfully
    print(f"Point cloud loaded with {len(point_cloud.points)} points")

    # Step 2: Save the point cloud as a .ply file
    o3d.io.write_point_cloud(output_ply, point_cloud)
    print(f"Point cloud has been saved to: {output_ply}")

    # Step 3: Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

# Example usage
input_las = "D:/E_2024_Thesis/Data/Input/Street/TLS_Street.las"
output_ply = "D:/E_2024_Thesis/Data/Input/Street/TLS_Street.ply"
convert_las_to_ply(input_las, output_ply)

