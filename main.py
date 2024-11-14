import laspy
import open3d as o3d
import numpy as np

# 读取 .las 文件
las = laspy.read("D:/E_2024_Thesis/Data/roof_TLS/roof_TLS.las")
points = np.vstack((las.x, las.y, las.z)).transpose()

# 创建 Open3D 点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可视化原始点云
o3d.visualization.draw_geometries([point_cloud], window_name="原始点云")

# 统计离群点去除
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 将去除的点云与剩余的点云分开
inlier_cloud = point_cloud.select_by_index(ind)
outlier_cloud = point_cloud.select_by_index(ind, invert=True)

# 可视化去噪后的点云
inlier_cloud.paint_uniform_color([0.1, 0.9, 0.1])  # 将去噪后的点云涂成绿色
outlier_cloud.paint_uniform_color([1, 0, 0])  # 将噪声点涂成红色
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="去噪后的点云")

# 计算法线
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 生成原始点云的 3D 网格模型（使用泊松重建）
mesh_original, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
mesh_original.paint_uniform_color([0.5, 0.5, 0.5])  # 原始网格涂成灰色

# 生成去噪后点云的 3D 网格模型
mesh_denoised, densities_denoised = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=9)
mesh_denoised.paint_uniform_color([0.1, 0.8, 0.1])  # 去噪后的网格涂成绿色

# 可视化原始和去噪后的 3D 网格模型
o3d.visualization.draw_geometries([mesh_original], window_name="原始点云 3D 模型")
o3d.visualization.draw_geometries([mesh_denoised], window_name="去噪后点云 3D 模型")

# 保存 3D 网格模型
o3d.io.write_triangle_mesh("D:/E_2024_Thesis/Data/roof_TLS/roof_original_mesh.ply", mesh_original)
o3d.io.write_triangle_mesh("D:/E_2024_Thesis/Data/roof_TLS/roof_denoised_mesh.ply", mesh_denoised)
