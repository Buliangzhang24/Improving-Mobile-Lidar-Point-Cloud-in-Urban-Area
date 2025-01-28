import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d

# 1. 从文件加载点云数据
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

# Voxel downsampling function
def voxel_downsampling(pcd, voxel_size):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled_pcd.points)

# 计算 Hausdorff 距离
def compute_hausdorff_distance(points1, points2):
    forward_hd = directed_hausdorff(points1, points2)[0]
    backward_hd = directed_hausdorff(points2, points1)[0]
    return max(forward_hd, backward_hd)

# 加载点云数据
file_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/TOP3/mls_bilateral.ply"  # 替换为你的路径
point_cloud = load_point_cloud(file_path)

# 将点云数据转为 Open3D 对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 2. 进行体素下采样 (设置合适的体素大小)
voxel_size = 0.01  # Example voxel size, you can adjust it
downsampled_point_cloud = voxel_downsampling(pcd, voxel_size)

# 继续处理下采样后的点云数据
point_cloud_downsampled = np.asarray(downsampled_point_cloud)

# 3. Delaunay 三角化
tri = Delaunay(point_cloud_downsampled[:, :2])  # 只对 X, Y 坐标进行三角化

# 打印 tri.simplices 类型和内容，检查是否为符合要求的整型数组
print("Triangulation simplices (before type conversion):")
print(tri.simplices)
print("Data type of tri.simplices:", tri.simplices.dtype)

# 确保索引类型为 int32
triangles = tri.simplices.astype(np.int32)
print("Triangulation simplices (after type conversion):")
print(triangles)
print("Data type of triangles:", triangles.dtype)

# 4. 创建网格对象
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(point_cloud_downsampled)

# 赋值给 mesh.triangles
mesh.triangles = o3d.utility.Vector3iVector(triangles)  # 使用 Delaunay 的三角形索引

# 5. 保存网格为PLY文件
output_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/TOP3/mesh_bilateral_001.ply"  # 替换为你的路径
o3d.io.write_triangle_mesh(output_path, mesh)

print("Mesh saved to", output_path)

# 6. 计算 Hausdorff 距离
# 获取三角网格的顶点
mesh_vertices = np.asarray(mesh.vertices)

# 计算 Hausdorff 距离
hd = compute_hausdorff_distance(point_cloud_downsampled, mesh_vertices)

# 输出 Hausdorff 距离
print(f"Hausdorff Distance: {hd:.4f}")

