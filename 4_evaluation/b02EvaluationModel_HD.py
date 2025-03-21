import open3d as o3d
import numpy as np

# 计算 Hausdorff 距离
def hausdorff_distance(mesh1, mesh2):
    # 转换网格为点云
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=5000)
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=5000)

    # 使用 Open3D 提供的 Hausdorff 距离计算方法
    dist1 = pcd1.compute_point_cloud_distance(pcd2)  # 计算点云1到点云2的距离
    dist2 = pcd2.compute_point_cloud_distance(pcd1)  # 计算点云2到点云1的距离

    # 计算最大距离
    hd1 = max(dist1)  # Mesh1到Mesh2的最大距离
    hd2 = max(dist2)  # Mesh2到Mesh1的最大距离

    return max(hd1, hd2)  # 取两个方向的最大值

# 加载点云或网格文件
mesh_tls = o3d.io.read_triangle_mesh("D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D_CloudCompare/Roof_TLS - Cloud.ply")
mesh_mls = o3d.io.read_triangle_mesh("D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D/KNN_patch.ply")

# 计算 Hausdorff 距离
hd_value = hausdorff_distance(mesh_tls, mesh_mls)
print(f"The Hausdorff distance between TLS and MLS mesh is: {hd_value}")
