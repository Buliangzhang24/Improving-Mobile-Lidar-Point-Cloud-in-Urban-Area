import open3d as o3d
import numpy as np

def align_and_evaluate(tls_pcd_path, mls_pcd_path, output_path):
    """
    对齐 TLS 和去噪后的 MLS 点云，并保存对齐后的 MLS 点云，同时计算 RMSE 以验证对齐效果。

    参数:
    tls_pcd_path: TLS 点云文件路径
    mls_pcd_path: 已去噪的 MLS 点云文件路径
    output_path: 对齐后 MLS 点云的保存路径

    返回:
    rmse: 对齐后的 RMSE 值
    """
    # 加载 TLS 和 MLS 点云
    tls_pcd = o3d.io.read_point_cloud(tls_pcd_path)
    mls_pcd = o3d.io.read_point_cloud(mls_pcd_path)

    print("原始点云加载完成。")
    print(f"TLS 点云有 {np.asarray(tls_pcd.points).shape[0]} 个点")
    print(f"MLS 点云有 {np.asarray(mls_pcd.points).shape[0]} 个点")

    # 配准前，使用特征匹配（如果可用）
    print("估计法线和特征...")
    tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    tls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    mls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(mls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    # 使用RANSAC方法进行初步配准
    print("正在进行 RANSAC 配准...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        mls_pcd, tls_pcd, mls_fpfh, tls_fpfh, mutual_filter=True,
        max_correspondence_distance=0.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    print("RANSAC 配准完成。")

    # 使用ICP进一步优化
    print("正在进行 ICP 配准...")
    threshold = 0.05  # 配准阈值
    reg_p2p = o3d.pipelines.registration.registration_icp(
        mls_pcd, tls_pcd, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print("ICP 配准完成。")

    # 应用变换到 MLS 点云
    mls_pcd.transform(reg_p2p.transformation)

    # 保存对齐后的 MLS 点云
    o3d.io.write_point_cloud(output_path, mls_pcd)
    print(f"对齐后的 MLS 点云已保存到: {output_path}")

    # 计算对齐后的 RMSE
    distances = np.asarray(tls_pcd.compute_point_cloud_distance(mls_pcd))
    rmse = np.sqrt(np.mean(distances**2))
    print(f"对齐后的 RMSE: {rmse}")

    return rmse

# 输入路径
tls_pcd_path = "D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.ply"
mls_pcd_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/CNN/PointRobust.ply"
output_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/Align/aligned_mls_Robust.ply"

# 调用对齐和验证函数
rmse = align_and_evaluate(tls_pcd_path, mls_pcd_path, output_path)
print(f"最终对齐 RMSE: {rmse}")
