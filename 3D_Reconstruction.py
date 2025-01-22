import open3d as o3d
import numpy as np


# 文件路径
input_ply = "D:/E_2024_Thesis/Data/Input/roof/Roof_MLS.ply"
final_mesh_output = "D:/E_2024_Thesis/Data/Output/Roof_3D_0.5_1.ply"

# 加载点云
point_cloud = o3d.io.read_point_cloud(input_ply)
print(f"原始点云包含 {len(np.asarray(point_cloud.points))} 个点")

# 检查点云是否为空或者点数不足
if len(np.asarray(point_cloud.points)) < 3:
    print("点云数据不足，无法进行建模！")
    exit()

# 输出点云的前几个点，查看点云数据的分布情况
print("点云的前几个点：")
print(np.asarray(point_cloud.points)[:10])

# 可视化原始点云
o3d.visualization.draw_geometries([point_cloud], window_name="原始点云")


