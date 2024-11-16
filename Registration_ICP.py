import open3d as o3d
import laspy
import numpy as np
import cupy as cp

# 使用 laspy 加载 LAS 文件并转换为 Open3D 点云
def load_las_as_o3d_point_cloud(las_file_path):
    try:
        # 读取 LAS 文件
        las = laspy.read(las_file_path)

        # 获取点云数据
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd
    except Exception as e:
        print(f"Error loading LAS file {las_file_path}: {e}")
        return None


# 对点云进行降采样（体素网格滤波）
def voxel_downsample(pcd, voxel_size=0.2):
    print(f"Downsampling point cloud with voxel size: {voxel_size}")
    return pcd.voxel_down_sample(voxel_size=voxel_size)


# 粗配准（使用 Fast Global Registration）
# 粗配准（使用 Fast Global Registration）
def coarse_registration(mls_pcd, tls_pcd):
    # 估计法线
    mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    # 确保法线估计成功
    if not mls_pcd.has_normals():
        print("Error: MLS point cloud has no normals!")
        return None
    if not tls_pcd.has_normals():
        print("Error: TLS point cloud has no normals!")
        return None

    # 提取FPFH特征
    radius = 0.1
    mls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        mls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    tls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

    # 粗配准：使用Fast Global Registration
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        mls_pcd, tls_pcd, mls_fpfh, tls_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption())
    return result

# ICP配准（使用PointToPlane精细配准）
def icp_registration(mls_pcd, tls_pcd, trans_init):
    try:
        threshold = 0.5  # 可以根据需要调整距离阈值
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000, relative_fitness=1e-6,
                                                                     relative_rmse=1e-6)

        # 使用PointToPlane ICP进行精细配准
        print("Performing ICP registration (PointToPlane)...")
        reg_icp = o3d.pipelines.registration.registration_icp(
            mls_pcd, tls_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
        return reg_icp
    except Exception as e:
        print(f"Error during ICP registration: {e}")
        return None


# 可视化配准结果
def visualize_registration(mls_pcd, tls_pcd):
    print("Visualizing registered point clouds...")
    o3d.visualization.draw_geometries([mls_pcd, tls_pcd])

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

# 加载点云
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

if tls_pcd is None or mls_pcd is None:
    print("Error: One or more point clouds failed to load!")
    exit()

# 1. 降采样（体素网格滤波）
tls_pcd = voxel_downsample(tls_pcd, voxel_size=0.2)  # 可以调整体素大小
mls_pcd = voxel_downsample(mls_pcd, voxel_size=0.2)

# 2. 粗配准（使用 Fast Global Registration）
coarse_result = coarse_registration(mls_pcd, tls_pcd)
if coarse_result is None:
    print("粗配准失败")
    exit()

trans_init = coarse_result.transformation  # 获得粗配准的变换矩阵

# 3. 精细配准（使用ICP）
icp_result = icp_registration(mls_pcd, tls_pcd, trans_init)
if icp_result is None:
    print("ICP 配准失败")
    exit()

# 4. 查看ICP配准结果
print("ICP 配准结果：")
print(icp_result.transformation)

# 5. 应用配准变换
mls_pcd.transform(icp_result.transformation)

# 6. 可视化配准后的点云
visualize_registration(mls_pcd, tls_pcd)

rmse = compute_rmse_gpu(mls_pcd, tls_pcd)
print(f"配准后的RMSE: {rmse}")

# 7. 保存配准后的点云
o3d.io.write_point_cloud("D:/E_2024_Thesis/Data/Output/aligned_mls_ICP.ply", mls_pcd)