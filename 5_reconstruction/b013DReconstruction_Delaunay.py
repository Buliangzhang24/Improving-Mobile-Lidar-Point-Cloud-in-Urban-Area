import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay


def fit_best_plane(points):
    """
    使用最小二乘法拟合平面
    """
    # 创建矩阵A和B
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]

    # 解线性方程 Ax = B
    plane_params = np.linalg.lstsq(A, B, rcond=None)[0]

    return plane_params


def project_to_plane(points, plane_params):
    """
    将点云投影到拟合的平面上
    """
    a, b, c = plane_params
    # 计算平面法向量
    normal = np.array([a, b, -1])
    normal = normal / np.linalg.norm(normal)  # 单位化法向量

    # 计算点到平面的投影
    projected_points = points - np.dot(points - points.mean(axis=0), normal)[:, np.newaxis] * normal
    return projected_points


def delaunay_triangulation_2d(points_2d):
    """
    对2D点进行Delaunay三角化
    """
    delaunay = Delaunay(points_2d)
    return delaunay.simplices


def apply_triangulation_to_3d(triangles, points_3d, plane_params):
    """
    将2D三角化结果映射回3D空间
    """
    # 恢复原始3D点的坐标
    triangles_3d = []
    for tri in triangles:
        # 每个三角形的3D坐标
        pts_3d = points_3d[tri]
        triangles_3d.append(pts_3d)
    return triangles_3d

def filter_large_edges(points_2d, delaunay_result, max_edge_length):
    """
    过滤掉包含大边的三角形
    """
    filtered_triangles = []
    for tri in delaunay_result:
        # 计算三角形的三条边
        edges = [np.linalg.norm(points_2d[tri[i]] - points_2d[tri[(i + 1) % 3]]) for i in range(3)]
        if all(edge <= max_edge_length for edge in edges):
            filtered_triangles.append(tri)
    return filtered_triangles

# 读取点云数据
pcd = o3d.io.read_point_cloud("D:/E_2024_Thesis/Data/Output/Roof/PointCloud/Normal/mls_guided.ply")
points = np.asarray(pcd.points)

# 步骤1: 拟合最佳平面
plane_params = fit_best_plane(points)

# 步骤2: 投影到平面
projected_points = project_to_plane(points, plane_params)

# 设置最大边长
max_edge_length = 2

# 步骤3: 进行Delaunay三角化（2D）
delaunay_result = delaunay_triangulation_2d(projected_points[:, :2])

# 步骤3.1: 过滤掉包含大边的三角形
filtered_delaunay_result = filter_large_edges(projected_points[:, :2], delaunay_result, max_edge_length)

# 步骤4: 将Delaunay三角化结果应用到3D点云
triangles_3d = apply_triangulation_to_3d(filtered_delaunay_result, points, plane_params)

# 确保过滤后的三角形数据不为空
if len(filtered_delaunay_result) == 0:
    print("Warning: No triangles after filtering. Try adjusting the max_edge_length or increasing the sampling.")
else:
    # 创建Mesh对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(filtered_delaunay_result)

# 保存Mesh
output_path = "D:/E_2024_Thesis/Data/Output/Roof/Mesh/3D/TOP/Normal_guided2.ply"
o3d.io.write_triangle_mesh(output_path, mesh)

# 可视化
o3d.visualization.draw_geometries([mesh])
