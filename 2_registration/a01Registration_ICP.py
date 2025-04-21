import open3d as o3d
import laspy
import numpy as np
import cupy as cp

# Using laspy load LAS and transform to  Open3D
def load_las_as_o3d_point_cloud(las_file_path):
    try:
        las = laspy.read(las_file_path)

        points = np.vstack((las.x, las.y, las.z)).transpose()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd
    except Exception as e:
        print(f"Error loading LAS file {las_file_path}: {e}")
        return None


# Downsampling
def voxel_downsample(pcd, voxel_size=0.2):
    print(f"Downsampling point cloud with voxel size: {voxel_size}")
    return pcd.voxel_down_sample(voxel_size=voxel_size)

# Fast Global Registration
def coarse_registration(mls_pcd, tls_pcd):

    mls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    tls_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    if not mls_pcd.has_normals():
        print("Error: MLS point cloud has no normals!")
        return None
    if not tls_pcd.has_normals():
        print("Error: TLS point cloud has no normals!")
        return None

    # extract FPFH feature
    radius = 0.1
    mls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        mls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    tls_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tls_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

    # Coarse Alignment: Fast Global Registration
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        mls_pcd, tls_pcd, mls_fpfh, tls_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption())
    return result

# Fine Alignment: ICP(using PointToPlane）
def icp_registration(mls_pcd, tls_pcd, trans_init):
    try:
        threshold = 0.5
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000, relative_fitness=1e-6,
                                                                     relative_rmse=1e-6)

        # Using PointToPlane ICP to Fine Alignment
        print("Performing ICP 2_registration (PointToPlane)...")
        reg_icp = o3d.pipelines.registration.registration_icp(
            mls_pcd, tls_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
        return reg_icp
    except Exception as e:
        print(f"Error during ICP 2_registration: {e}")
        return None


# Visualize the result
def visualize_registration(mls_pcd, tls_pcd):
    print("Visualizing registered point clouds...")
    o3d.visualization.draw_geometries([mls_pcd, tls_pcd])

def compute_rmse_gpu(source_pcd, target_pcd):

    target_points = np.asarray(target_pcd.points)

    target_points_gpu = cp.array(target_points)

    distances = []
    for point in source_pcd.points:
        source_point_gpu = cp.array(point)

        diff = target_points_gpu - source_point_gpu
        dist = cp.linalg.norm(diff, axis=1)
        distances.append(cp.min(dist))

    # using them in the GPU
    distances_gpu = cp.array(distances)

    # Calculate RMSE
    rmse_gpu = cp.sqrt(cp.mean(distances_gpu ** 2))

    return rmse_gpu.get()  #

# loading point clouds
tls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_TLS.las")
mls_pcd = load_las_as_o3d_point_cloud("D:/E_2024_Thesis/Data/roof/roof_MLS.las")

if tls_pcd is None or mls_pcd is None:
    print("Error: One or more point clouds failed to load!")
    exit()

# 1. Downsampling (Voxel Grid Filter)
tls_pcd = voxel_downsample(tls_pcd, voxel_size=0.2)  # Adjust voxel size as needed
mls_pcd = voxel_downsample(mls_pcd, voxel_size=0.2)

# 2. Coarse Registration (Fast Global Registration)
coarse_result = coarse_registration(mls_pcd, tls_pcd)
if coarse_result is None:
    print("Coarse registration failed")
    exit()

trans_init = coarse_result.transformation  # Get coarse registration transform matrix

# 3. Fine Registration (ICP)
icp_result = icp_registration(mls_pcd, tls_pcd, trans_init)
if icp_result is None:
    print("ICP registration failed")
    exit()

# 4. Show ICP registration result
print("ICP registration result:")
print(icp_result.transformation)

# 5. Apply registration transform
mls_pcd.transform(icp_result.transformation)

# 6. Visualize registered point clouds
visualize_registration(mls_pcd, tls_pcd)

rmse = compute_rmse_gpu(mls_pcd, tls_pcd)
print(f"Registration RMSE: {rmse}")

# 7. Save registered point cloud
o3d.io.write_point_cloud("D:/E_2024_Thesis/Data/Output/aligned_mls_ICP.ply", mls_pcd)