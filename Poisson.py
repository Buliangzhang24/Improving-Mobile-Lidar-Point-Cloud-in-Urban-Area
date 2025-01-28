import open3d as o3d
import numpy as np


def interpolate_point_cloud(point_cloud, voxel_size=0.3):
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


def reconstruct_surface_with_poisson(point_cloud, depth=8):
    # 使用泊松重建进行表面重建
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    return mesh, densities


def calculate_hausdorff_distance(ref_pcd, target_pcd):
    """
    计算参考点云和目标点云之间的 Hausdorff 距离。

    参数：
    - ref_pcd: 参考点云
    - target_pcd: 目标点云

    返回：
    - Hausdorff 距离
    """
    # 使用 open3d 的点到点距离计算工具
    distances = ref_pcd.compute_point_cloud_distance(target_pcd)
    hausdorff_distance = np.max(distances)
    return hausdorff_distance


# 加载点云
input_ply = "D:/E_2024_Thesis/Data/Input/roof/Roof_MLS.ply"
final_mesh_output = "D:/E_2024_Thesis/Data/Input/roof/Roof_Mesh.ply"

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

# Step 2: 使用泊松进行 3D 重建
print("正在进行泊松表面重建...")
mesh, densities = reconstruct_surface_with_poisson(interpolated_pcd, depth=9)

if not mesh.vertices:
    raise RuntimeError("泊松重建生成的网格为空，请检查点云数据或调整参数。")

# 保存重建结果
o3d.io.write_triangle_mesh(final_mesh_output, mesh)
print(f"泊松网格已保存至: {final_mesh_output}")

# Step 3: 计算 Hausdorff 距离
print("正在计算 Hausdorff 距离...")
mesh_sampled_pcd = mesh.sample_points_uniformly(number_of_points=10000)  # 从网格中采样点
hausdorff_distance = calculate_hausdorff_distance(interpolated_pcd, mesh_sampled_pcd)
print(f"Hausdorff 距离: {hausdorff_distance:.4f}")

# 可视化最终结果
print("可视化插值点云和泊松重建结果...")
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Poisson Reconstruction")
