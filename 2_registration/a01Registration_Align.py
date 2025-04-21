import open3d as o3d
import numpy as np

def align_and_evaluate(tls_pcd_path, mls_pcd_path, output_path):
    """
    align TLS with MLS and save the MLS after alignment. Calculate RMSE

    parameters:
    tls_pcd_path: TLS pathway
    mls_pcd_path: MlS pathway
    output_path: MLS after alignment

    Returns:
    rmse
    """
    # loading
    tls_pcd = o3d.io.read_point_cloud(tls_pcd_path)
    mls_pcd = o3d.io.read_point_cloud(mls_pcd_path)

    tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    tls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    mls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(mls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    # Coarse Alignment
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        mls_pcd, tls_pcd, mls_fpfh, tls_fpfh, mutual_filter=True,
        max_correspondence_distance=0.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Fine Alignment using ICP
    threshold = 0.05
    reg_p2p = o3d.pipelines.registration.registration_icp(
        mls_pcd, tls_pcd, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Applying to MLS
    mls_pcd.transform(reg_p2p.transformation)

    # Save to MLS
    o3d.io.write_point_cloud(output_path, mls_pcd)

    distances = np.asarray(tls_pcd.compute_point_cloud_distance(mls_pcd))
    rmse = np.sqrt(np.mean(distances**2))

    return rmse

# pathway
tls_pcd_path = "D:/E_2024_Thesis/Data/Input/roof/Roof_TLS.ply"
mls_pcd_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/KNN/pcd_denoised_patch.ply"
output_path = "D:/E_2024_Thesis/Data/Output/Roof/PointCloud/Align/aligned_mls_patch.ply"

rmse = align_and_evaluate(tls_pcd_path, mls_pcd_path, output_path)
