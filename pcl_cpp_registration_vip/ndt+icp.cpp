#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>//icp头文件
#include <pcl/registration/ndt.h> //ndt头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>//去除NAN点的头文件
#include <pcl/filters/approximate_voxel_grid.h> //官网上采样过滤，先不用

using namespace std;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;


//点云可视化

void visualize_pcd(PointCloud::Ptr pcd_src, PointCloud::Ptr pcd_tgt, PointCloud::Ptr pcd_final)
{
	pcl::visualization::PCLVisualizer viewer("registration Viewer");
	//原始点云绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(pcd_src, 0, 255, 0);
	//目标点云红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(pcd_tgt, 255, 0, 0);
	//匹配好的点云蓝色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(pcd_final, 0, 0, 255);

	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(pcd_src, src_h, "source cloud");
	viewer.addPointCloud(pcd_tgt, tgt_h, "target cloud");
	viewer.addPointCloud(pcd_final, final_h, "result cloud");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}
//由旋转平移矩阵计算旋转角度
void matrix2angle(Eigen::Matrix4f &result_trans, Eigen::Vector3f &result_angle)
{
	double ax, ay, az;
	if (result_trans(2, 0) == 1 || result_trans(2, 0) == -1)
	{
		az = 0;
		double dlta;
		dlta = atan2(result_trans(0, 1), result_trans(0, 2));
		if (result_trans(2, 0) == -1)
		{
			ay = M_PI / 2;
			ax = az + dlta;
		}
		else
		{
			ay = -M_PI / 2;
			ax = -az + dlta;
		}
	}
	else
	{
		ay = -asin(result_trans(2, 0));
		ax = atan2(result_trans(2, 1) / cos(ay), result_trans(2, 2) / cos(ay));
		az = atan2(result_trans(1, 0) / cos(ay), result_trans(0, 0) / cos(ay));
	}
	result_angle << ax, ay, az;

	cout << "x轴旋转角度：" << ax << endl;
	cout << "y轴旋转角度：" << ay << endl;
	cout << "z轴旋转角度：" << az << endl;
}

int main(int argc, char** argv)
{

	//创建点云指针
	PointCloud::Ptr cloud_source(new PointCloud);
	PointCloud::Ptr cloud_target(new PointCloud);


	// 加载点云文件
	pcl::io::loadPCDFile("E:/vs13/pcldata/bun/rabbit.pcd", *cloud_source);
	std::cout << "source loaded!" << std::endl;
	pcl::io::loadPCDFile("E:/vs13/pcldata/bun/rabbit_1.pcd", *cloud_target);
	std::cout << "target loaded!" << std::endl;

	clock_t start = clock();

	//去除NAN点

	std::vector<int> indices_src; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*cloud_source, *cloud_source, indices_src);
	std::cout << "remove *cloud_source nan" << endl;

	std::vector<int> indices_tgt; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*cloud_target, *cloud_target, indices_tgt);
	std::cout << "remove *cloud_target nan" << endl;

	//下采样滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setLeafSize(0.012, 0.012, 0.012);
	voxel_grid.setInputCloud(cloud_source);
	PointCloud::Ptr cloud_src(new PointCloud);
	voxel_grid.filter(*cloud_src);
	std::cout << "down size *cloud_src_o from " << cloud_source->size() << "to" << cloud_src->size() << endl;


	//NDT配准
	//初始化正太分布NDT对象
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	PointCloud::Ptr cloud_ndt(new PointCloud);
	ndt.setTransformationEpsilon(0.01);
	ndt.setStepSize(0.05);
	ndt.setResolution(3);//网格分辨率
	ndt.setMaximumIterations(100);

	//载入点云
	ndt.setInputSource(cloud_src);
	ndt.setInputTarget(cloud_target);

	//设置初始变换矩阵，可有可无
	Eigen::AngleAxisf init_rotation(M_PI / 4, Eigen::Vector3f::UnitZ());//以z轴为坐标轴，旋转45°
	Eigen::Translation3f init_transtion(0, 0, 0);
	Eigen::Matrix4f init_guess = (init_transtion*init_rotation).matrix();

	ndt.align(*cloud_ndt, init_guess);

	clock_t ndt_t = clock();
	cout << "ndt time" << (double)(ndt_t - start) / CLOCKS_PER_SEC << endl;

	Eigen::Matrix4f transformation = ndt.getFinalTransformation();
	std::cout << transformation << std::endl;
	pcl::transformPointCloud(*cloud_source, *cloud_ndt, transformation);

	std::vector<int> indices_ndt; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*cloud_ndt, *cloud_ndt, indices_ndt);
	std::cout << "remove *cloud_ndt nan" << endl;


	//下采样滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
	voxel_grid_2.setLeafSize(0.012, 0.012, 0.012);
	voxel_grid_2.setInputCloud(cloud_ndt);
	PointCloud::Ptr cloud_ndt_o(new PointCloud);
	voxel_grid.filter(*cloud_ndt_o);
	std::cout << "down size *cloud_ndt from " << cloud_ndt->size() << "to" << cloud_ndt_o->size() << endl;



	//icp配准算法


	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	PointCloud::Ptr cloud_icp_registration(new PointCloud);
	//kdTree 加速搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(cloud_ndt_o);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(cloud_target);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);

	//设置参数
	icp.setInputSource(cloud_ndt_o);
	icp.setInputTarget(cloud_target);
	icp.setMaxCorrespondenceDistance(0.5);//当两个点云相距较远时候，距离值要变大，所以一开始需要粗配准。
	icp.setTransformationEpsilon(1e-10);//svd奇异值分解，对icp时间影响不大
	icp.setEuclideanFitnessEpsilon(0.01);//前后两次误差大小，当误差值小于这个值停止迭代
	icp.setMaximumIterations(100);//最大迭代次数
	icp.align(*cloud_icp_registration, transformation);

	clock_t end = clock();

	cout << "icp time" << (double)(end - ndt_t) / CLOCKS_PER_SEC << endl;
	cout << "total time" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	Eigen::Matrix4f transformation_icp = icp.getFinalTransformation();
	std::cout << transformation_icp << std::endl;

	pcl::transformPointCloud(*cloud_source, *cloud_icp_registration, transformation_icp);

	//计算误差
	Eigen::Vector3f ANGLE_origin;
	Eigen::Vector3f TRANS_origin;
	ANGLE_origin << 0, 0, M_PI / 4;
	TRANS_origin << 0, 0.3, 0.2;
	double a_error_x, a_error_y, a_error_z;
	double t_error_x, t_error_y, t_error_z;
	Eigen::Vector3f ANGLE_result;
	matrix2angle(icp_trans, ANGLE_result);
	a_error_x = fabs(ANGLE_result(0)) - fabs(ANGLE_origin(0));
	a_error_y = fabs(ANGLE_result(1)) - fabs(ANGLE_origin(1));
	a_error_z = fabs(ANGLE_result(2)) - fabs(ANGLE_origin(2));
	cout << "点云实际旋转角度:\n" << ANGLE_origin << endl;
	cout << "x轴旋转误差 : " << a_error_x << "  y轴旋转误差 : " << a_error_y << "  z轴旋转误差 : " << a_error_z << endl;

	cout << "点云实际平移距离:\n" << TRANS_origin << endl;
	t_error_x = fabs(icp_trans(0, 3)) - fabs(TRANS_origin(0));
	t_error_y = fabs(icp_trans(1, 3)) - fabs(TRANS_origin(1));
	t_error_z = fabs(icp_trans(2, 3)) - fabs(TRANS_origin(2));
	cout << "计算得到的平移距离" << endl << "x轴平移" << icp_trans(0, 3) << endl << "y轴平移" << icp_trans(1, 3) << endl << "z轴平移" << icp_trans(2, 3) << endl;
	cout << "x轴平移误差 : " << t_error_x << "  y轴平移误差 : " << t_error_y << "  z轴平移误差 : " << t_error_z << endl;
	//可视化
	visualize_pcd(cloud_source, cloud_target, cloud_icp_registration);
	return 0;
}
