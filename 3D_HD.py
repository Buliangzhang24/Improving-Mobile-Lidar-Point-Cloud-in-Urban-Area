import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def downsample_point_cloud(pcd, voxel_size):
    """
    对点云进行下采样
    """
    return pcd.voxel_down_sample(voxel_size)

def downsample_mesh(mesh, target_ratio):
    """
    对网格进行简化，减少三角形数目
    """
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles) * target_ratio))
    return mesh

def hausdorff_distance_kdtree(pcd_points, mesh_points):
    """
    计算点云与网格之间的Hausdorff距离，使用KDTree优化计算
    """
    # 创建KDTree
    pcd_tree = cKDTree(pcd_points)
    mesh_tree = cKDTree(mesh_points)

    # 对于点云中的每个点，找到离它最近的网格点
    dist_pcd_to_mesh, _ = pcd_tree.query(mesh_points)
    dist_mesh_to_pcd, _ = mesh_tree.query(pcd_points)

    # 返回最大最小距离
    return max(np.max(dist_pcd_to_mesh), np.max(dist_mesh_to_pcd))

def load_ply(file_path):
    """
    加载PLY文件，返回网格的点
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh_points = np.asarray(mesh.vertices)
    return mesh

def load_point_cloud(pcd_file_path):
    """
    加载点云文件，返回点云的点
    """
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    pcd_points = np.asarray(pcd.points)
    return pcd_points

# 文件路径
reference_pcd_file = "D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.ply"  # 替换为你的参考点云文件路径
mesh_file = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/Random Sample/mls_denoised_density.ply"   # 替换为网格文件路径

# 加载参考点云和网格数据
pcd = o3d.io.read_point_cloud(reference_pcd_file)
mesh = o3d.io.read_triangle_mesh(mesh_file)

# 对点云和网格进行下采样
voxel_size = 0.05  # 设置点云下采样的体素大小
target_ratio = 0.1  # 设置网格简化的目标比例

downsampled_pcd = downsample_point_cloud(pcd, voxel_size)
downsampled_mesh = downsample_mesh(mesh, target_ratio)

# 获取下采样后的点
pcd_points = np.asarray(downsampled_pcd.points)
mesh_points = np.asarray(downsampled_mesh.vertices)

# 计算Hausdorff距离
hd_value = hausdorff_distance_kdtree(pcd_points, mesh_points)

print(f"Hausdorff Distance between reference point cloud and surface mesh: {hd_value}")










