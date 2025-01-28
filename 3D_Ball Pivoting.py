import open3d as o3d

# 加载点云
pcd = o3d.io.read_point_cloud("D:/E_2024_Thesis/Data/Output/Roof/PointCloud/TOP3/mls_bilateral.ply")

# 体素下采样（如果点云过于稠密）
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)

# 定义一系列球体半径，从小到大
radii_list = [
    [0.01, 0.02, 0.04],  # 例1：小的半径
    [0.01, 0.05, 0.1],  # 例2：稍大的半径
    [0.02, 0.1, 0.2],  # 例3：较大的半径
    [0.05, 0.1, 0.3],  # 例4：较大的半径
    [0.1, 0.2, 0.5]  # 例5：很大的半径
]

# 自动遍历所有半径设置，并保存对应的网格
for i, radii in enumerate(radii_list):
    print(f"正在处理半径设置: {radii}")
    try:
        # Ball Pivoting 网格重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downsampled_pcd,
                                                                               o3d.utility.DoubleVector(radii))

        # 检查网格是否为空
        if len(mesh.triangles) == 0:
            print(f"警告: 半径设置 {radii} 生成的网格为空！")
            continue

        # 保存网格到文件
        output_path = f"D:/E_2024_Thesis/Data/Output/Roof/PointCloud/TOP3/mesh_{i + 1}.ply"
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"网格已保存到: {output_path}")
    except Exception as e:
        print(f"处理半径设置 {radii} 时发生错误: {e}")

print("所有网格处理完成！")
