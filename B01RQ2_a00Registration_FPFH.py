import open3d as o3d
import laspy
import numpy as np
import os
import cupy as cp

# 加载LAS文件为Open3D点云
def load_las_as_o3d_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# 计算法向量
def estimate_normals(pcd, voxel_size=1.0):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    return pcd

# 计算FPFH特征
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

# 计算RMSE
# 计算RMSE（使用GPU加速）
def compute_rmse_gpu(source_pcd, target_pcd):
    # 将目标点云转换为numpy数组
    target_points = np.asarray(target_pcd.points)

    # 将numpy数组转换为GPU数组（CuPy）
    target_points_gpu = cp.array(target_points)

    distances = []
    for point in source_pcd.points:
        # 将源点云中的点转换为GPU数组
        source_point_gpu = cp.array(point)

        # 计算源点与目标点云中所有点的距离，并取最小值
        diff = target_points_gpu - source_point_gpu
        dist = cp.linalg.norm(diff, axis=1)
        distances.append(cp.min(dist))  # 取最小距离

    # 将所有距离转换为GPU数组
    distances_gpu = cp.array(distances)

    # 计算RMSE
    rmse_gpu = cp.sqrt(cp.mean(distances_gpu ** 2))

    # 将结果从GPU传回CPU
    return rmse_gpu.get()  # 获取结果


# 加载点云数据
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/LiDAR_Engelseplein/Engelseplein_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/LiDAR_Engelseplein/Engelseplein_MLS.las")

# 下采样
voxel_size = 1.0  # 设置体素大小，可以根据需要调整
tls_down = tls_pcd.voxel_down_sample(voxel_size)
mls_down = mls_pcd.voxel_down_sample(voxel_size)

# 计算法向量
tls_down = estimate_normals(tls_down, voxel_size)
mls_down = estimate_normals(mls_down, voxel_size)

# 计算 FPFH 特征
tls_fpfh = compute_fpfh(tls_down, voxel_size)
mls_fpfh = compute_fpfh(mls_down, voxel_size)

# 基于 RANSAC 的粗配准
distance_threshold = voxel_size * 1.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    mls_down, tls_down, mls_fpfh, tls_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4,  # RANSAC 中的对应关系数
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 500)
)

# 使用粗配准结果作为初始变换，进行精细配准
threshold = 0.1  # ICP 的距离阈值
fine_reg = o3d.pipelines.registration.registration_icp(
    mls_pcd, tls_pcd, threshold, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# 应用精细配准的变换
mls_pcd.transform(fine_reg.transformation)

# 计算配准精度（RMSE）
rmse = compute_rmse_gpu(mls_pcd, tls_pcd)
print(f"配准后的RMSE: {rmse}")

# 将配准后的点云保存
output_path = "D:/E_2024_Thesis/Data/aligned_mls_FPFH.ply"

# 检查文件是否存在，存在则删除
if os.path.exists(output_path):
    print(f"文件 {output_path} 已存在，正在删除...")
    os.remove(output_path)

# 保存配准后的点云
o3d.io.write_point_cloud(output_path, mls_pcd)
print(f"配准后的点云已保存至 {output_path}")

# 可视化配准后的点云
o3d.visualization.draw_geometries([tls_pcd, mls_pcd])
