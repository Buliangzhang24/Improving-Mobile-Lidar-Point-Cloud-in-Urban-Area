import open3d as o3d

# 加载点云
pcd = o3d.io.read_point_cloud("D:/E_2024_Thesis/Data/Output/Roof/PointCloud/TOP5/mls_guided.ply")

# 检查点云是否有法线
if not pcd.has_normals():
    print("点云未包含法线信息，正在计算法线...")
    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 调整法线方向，使其一致
    pcd.orient_normals_consistent_tangent_plane(k=5)
    print("法线计算完成并已调整方向。")

# 体素下采样（如果点云过于稠密）
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)

# 再次检查下采样后的点云是否仍有法线
if not downsampled_pcd.has_normals():
    print("下采样后的点云丢失法线信息，重新计算法线...")
    downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    downsampled_pcd.orient_normals_consistent_tangent_plane(k=5)
    print("法线重新计算并调整完成。")

# 定义一系列球体半径，从小到大
radii_list = [
    [0.05, 0.1, 0.3],  # 例4：较大的半径
    [0.1, 0.2, 0.5],  # 例5：很大的半径
    [0.2, 0.4, 1.0],  # 例6：更大的半径
    [0.5, 1.0, 2.0],  # 例7：更大的半径
    [1.0, 2.0, 4.0],  # 例8：更大的半径
]

# 自动遍历所有半径设置，并保存对应的网格
for i, radii in enumerate(radii_list):
    print(f"正在处理半径设置: {radii}")
    try:
        # Ball Pivoting 网格重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            downsampled_pcd, o3d.utility.DoubleVector(radii)
        )

        # 检查网格是否为空
        if len(mesh.triangles) == 0:
            print(f"警告: 半径设置 {radii} 生成的网格为空！")
            continue

        # 保存网格到文件
        output_path = f"D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D_BP/mesh_guided/big_mesh_{i + 1}.ply"
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"网格已保存到: {output_path}")
    except Exception as e:
        print(f"处理半径设置 {radii} 时发生错误: {e}")

print("所有网格处理完成！")
