import open3d as o3d

# 输入和输出路径
input_ply = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/roof_denoised_density.ply"
final_mesh_output = "D:/E_2024_Thesis/Data/Output/3D_Roof_Density.ply"

# 加载点云数据
print("加载点云...")
pcd = o3d.io.read_point_cloud(input_ply)
print(f"点云加载成功，包含 {len(pcd.points)} 个点")

# 法向量估算
print("估算点云法向量...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# BPA 重建
print("进行 BPA 重建...")
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = sum(distances) / len(distances)
radius = 10 * avg_dist  # 半径设置为平均距离的两倍
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector([radius, radius * 2, radius * 3])
)

# 保存重建网格
print(f"保存重建网格至: {final_mesh_output}")
o3d.io.write_triangle_mesh(final_mesh_output, mesh)

# 可视化网格（可选）
o3d.visualization.draw_geometries([mesh], window_name="BPA 重建网格")
print("BPA 重建完成！")


