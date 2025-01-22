import open3d as o3d
import numpy as np

def reconstruct_surface_with_alpha_shape(point_cloud, alpha=0.05):
    """
    使用 Alpha Shape 进行表面重建
    """
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
        if len(mesh.vertices) == 0:
            raise ValueError("生成的网格为空")
        return mesh
    except Exception as e:
        raise RuntimeError(f"Alpha Shape 重建失败，错误: {e}")

# 创建简化的点云
test_pcd = o3d.geometry.PointCloud()
test_pcd.points = o3d.utility.Vector3dVector(
    np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])
)

# 可视化简化的点云
o3d.visualization.draw_geometries([test_pcd], window_name="Test Point Cloud")

# 尝试 Alpha Shape 重建
alpha = 0.1
try:
    mesh = reconstruct_surface_with_alpha_shape(test_pcd, alpha)
    print("Alpha Shape 重建成功！网格顶点数:", len(mesh.vertices))
    o3d.visualization.draw_geometries([mesh], window_name="Alpha Shape Reconstruction")
except RuntimeError as e:
    print(e)
