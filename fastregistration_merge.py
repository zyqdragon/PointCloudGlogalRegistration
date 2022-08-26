import open3d as o3d
import numpy as np
import copy
import time

 # merge source_temp and target_temp into one pcd file
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
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # merge source_temp and target_temp into one pcd file
    pcddata1=np.array(source_temp.points)
    pcddata2=np.array(target_temp.points)
    pcddata1=np.append(pcddata1,pcddata2, axis=0)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pcddata1)
    o3d.io.write_point_cloud("mynew_girl.pcd",pcd)#以二进制格式存储点数据集部分
    o3d.visualization.draw_geometries([source_temp, target_temp])

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("./datasets/pcpair2/Depth_0000.ply")
    target = o3d.io.read_point_cloud("./datasets/pcpair2/Depth_0001.ply")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

voxel_size = 0.01  # 相当于使用5cm的体素对点云进行均值操作
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: 基于距离阈值为 %.3f的快速全局配准" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source_down, target_down,
                                                                                    source_fpfh, target_fpfh,
                                                                                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                        maximum_correspondence_distance=distance_threshold))
    return result
start = time.time()
result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print("快速全局配准花费了： %.3f 秒.\n" % (time.time() - start))
print(result_fast)
draw_registration_result(source_down, target_down, result_fast.transformation)



