import laspy
import open3d as o3d
import numpy as np

# 读取 .las 文件
input_las = "D:/E_2024_Thesis/Data/roof/roof_MLS.las"
output_ply = "D:/E_2024_Thesis/Data/roof/roof_MLS.ply"

# 使用 laspy 读取点云数据
las = laspy.read(input_las)

# 提取点云的 x, y, z 坐标
x = las.x
y = las.y
z = las.z
points = np.vstack((x, y, z)).T

# 创建 Open3D 点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Step 1: 检查点云是否成功加载
print(f"Point cloud loaded with {len(point_cloud.points)} points")

# Step 2: 保存为 .ply 格式
o3d.io.write_point_cloud(output_ply, point_cloud)
print(f"Point cloud has been saved to: {output_ply}")

# Step 3: 可视化点云
o3d.visualization.draw_geometries([point_cloud])
