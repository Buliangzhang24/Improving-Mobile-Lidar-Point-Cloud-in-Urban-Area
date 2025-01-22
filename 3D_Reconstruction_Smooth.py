import open3d as o3d
import numpy as np


def interpolate_point_cloud(point_cloud, voxel_size=0.1):
    # 对点云进行体素降采样
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled_pcd.points)

    # 创建 KD 树用于最近邻查找
    kd_tree = o3d.geometry.KDTreeFlann(downsampled_pcd)

    # 插值生成新点
    interpolated_points = []
    for point in points:
        [_, idx, _] = kd_tree.search_knn_vector_3d(point, 10)  # 最近邻数量
        neighbors = points[idx]
        interpolated_point = np.mean(neighbors, axis=0)
        interpolated_points.append(interpolated_point)

    # 将插值点与原始点合并
    all_points = np.vstack((points, np.array(interpolated_points)))
    interpolated_pcd = o3d.geometry.PointCloud()
    interpolated_pcd.points = o3d.utility.Vector3dVector(all_points)
    return interpolated_pcd


def reconstruct_surface_with_alpha_shape(point_cloud, alpha=0.05):
    # 使用 Alpha Shape 进行表面重建
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    return mesh


def calculate_normal_consistency(pcd, radius=0.1):
    """
    计算点云法线的一致性，评估表面平滑度。

    参数：
    - pcd: 输入的点云对象
    - radius: 用于估算法线的邻域半径

    返回：
    - 平滑度度量值（平均法线角度差异）
    """
    # 估算点云法线
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # 获取法线向量
    normals = np.asarray(pcd.normals)

    # 计算法线角度差异
    angle_differences = []
    for i in range(1, len(normals)):
        normal1 = normals[i - 1]
        normal2 = normals[i]

        # 计算法线之间的夹角（弧度）
        dot_product = np.dot(normal1, normal2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_differences.append(angle)

    # 计算平均法线角度差异
    avg_angle_diff = np.mean(angle_differences)

    return avg_angle_diff


# 加载点云
input_ply = "D:/E_2024_Thesis/Data/Input/roof/Roof_MLS.ply"
interpolated_output_ply = "D:/E_2024_Thesis/Data/Output/Roof_Interpolated_PointCloud2.ply"
final_mesh_output = "D:/E_2024_Thesis/Data/Output/1Origianl_Roof_PointFiler3D.ply"

point_cloud = o3d.io.read_point_cloud(input_ply)

# 检查点云是否成功加载
if not point_cloud.has_points():
    raise ValueError(f"点云加载失败或为空: {input_ply}")
print(f"点云加载成功，包含 {len(point_cloud.points)} 个点")

# Step 1: 插值补全点云
print("正在进行点云插值...")
interpolated_pcd = interpolate_point_cloud(point_cloud)
print(f"插值点云中包含 {len(interpolated_pcd.points)} 个点")
if not interpolated_pcd.has_points():
    raise ValueError("插值点云为空，无法进行表面重建。")

interpolated_pcd.remove_non_finite_points()
interpolated_pcd.remove_duplicated_points()
o3d.visualization.draw_geometries([interpolated_pcd])

# Step 2: 使用 Alpha Shape 进行 3D 重建
print("正在进行 3D 表面重建...")
alpha_values = [0.05, 0.1, 0.5, 2]  # 尝试不同的 alpha 参数
mesh = None

for alpha in alpha_values:
    try:
        print(f"尝试使用 alpha = {alpha} 进行表面重建...")
        mesh = reconstruct_surface_with_alpha_shape(interpolated_pcd, alpha)
        if len(mesh.vertices) == 0:
            print(f"Alpha = {alpha} 时生成的网格为空。")
            continue
        print(f"成功生成网格，顶点数: {len(mesh.vertices)}")
        o3d.io.write_triangle_mesh(final_mesh_output, mesh)
        print(f"网格已保存至: {final_mesh_output}")
        break
    except Exception as e:
        print(f"Alpha = {alpha} 重建失败: {e}")

if mesh is None or len(mesh.vertices) == 0:
    raise RuntimeError("所有 alpha 参数均无法生成有效网格，请检查点云数据或调整参数。")

# Step 3: 计算表面平滑度
pcd_from_mesh = o3d.geometry.PointCloud()
pcd_from_mesh.points = mesh.vertices
smoothness = calculate_normal_consistency(pcd_from_mesh)
print(f"Average normal angle difference (surface smoothness): {smoothness:.4f} radians")

# 可视化最终结果
print("可视化插值点云和重建结果...")
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Alpha Shape Reconstruction")
