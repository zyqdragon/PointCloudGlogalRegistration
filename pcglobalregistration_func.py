import open3d as o3d
import numpy as np
import copy
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: 使用大小为为{}的体素下采样点云.".format(voxel_size))
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: 使用搜索半径为{}估计法线".format(radius_normal))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: 使用搜索半径为{}计算FPFH特征".format(radius_feature))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("./datasets/pcpair1/Depth_0000.ply")
    target = o3d.io.read_point_cloud("./datasets/pcpair1/Depth_0001.ply")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh
voxel_size = 0.05 # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: 对下采样的点云进行RANSAC配准.")
    print("   下采样体素的大小为： %.3f," % voxel_size)
    print("   使用宽松的距离阈值： %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
start = time.time()
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print("快速全局配准花费了： %.3f 秒.\n" % (time.time() - start))
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: 对原始点云进行点对面ICP配准精细对齐， 这次使用严格的距离阈值： %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold,
                                                         result_ransac.transformation,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)
#https: // blog.csdn.net / u014072827 / article / details / 113788879
