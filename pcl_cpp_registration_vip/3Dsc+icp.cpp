#include <pcl/registration/ia_ransac.h>//����һ����
#include <pcl/filters/random_sample.h>//��ȡ�̶������ĵ���
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/3dsc.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>//
#include <pcl/filters/filter.h>//
#include <pcl/registration/icp.h>//icp��׼
#include <pcl/visualization/pcl_visualizer.h>//���ӻ�
#include <time.h>//ʱ��

using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//���ƿ��ӻ�
void visualize_pcd(PointCloud::Ptr pcd_src, PointCloud::Ptr pcd_tgt, PointCloud::Ptr pcd_final)
{

	//������ʼ��Ŀ��
	pcl::visualization::PCLVisualizer viewer("registration Viewer");


	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(pcd_src, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(pcd_tgt, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(pcd_final, 0, 0, 255);
	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(pcd_src, src_h, "source cloud");
	viewer.addPointCloud(pcd_tgt, tgt_h, "tgt cloud");
	viewer.addPointCloud(pcd_final, final_h, "final cloud");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
//����תƽ�ƾ��������ת�Ƕ�
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

	cout << "x����ת�Ƕȣ�" << ax << endl;
	cout << "y����ת�Ƕȣ�" << ay << endl;
	cout << "z����ת�Ƕȣ�" << az << endl;
}


int main(int argc, char** argv)
{
	//���ص����ļ�
	PointCloud::Ptr cloud_src_o(new PointCloud);//ԭ���ƣ�����׼
	pcl::io::loadPCDFile("E:/vs13/pcldata/bun/rabbit.pcd", *cloud_src_o);
	PointCloud::Ptr cloud_tgt_o(new PointCloud);//Ŀ�����
	pcl::io::loadPCDFile("E:/vs13/pcldata/bun/rabbit_1.pcd", *cloud_tgt_o);

	clock_t start = clock();

	//ȥ��NAN��
	std::vector<int> indices_src; //����ȥ���ĵ������
	pcl::removeNaNFromPointCloud(*cloud_src_o, *cloud_src_o, indices_src);
	std::cout << "remove *cloud_src_o nan" << endl;

	std::vector<int> indices_tgt;
	pcl::removeNaNFromPointCloud(*cloud_tgt_o, *cloud_tgt_o, indices_tgt);
	std::cout << "remove *cloud_tgt_o nan" << endl;


	//�����̶��ĵ�������

	pcl::RandomSample<PointT> rs_src;
	rs_src.setInputCloud(cloud_src_o);
	rs_src.setSample(500);
	PointCloud::Ptr cloud_src(new PointCloud);
	rs_src.filter(*cloud_src);
	std::cout << "down size *cloud_src_o from " << cloud_src_o->size() << "to" << cloud_src->size() << endl;

	pcl::RandomSample<PointT> rs_tgt;
	rs_tgt.setInputCloud(cloud_tgt_o);
	rs_tgt.setSample(500);
	PointCloud::Ptr cloud_tgt(new PointCloud);
	rs_tgt.filter(*cloud_tgt);
	std::cout << "down size *cloud_tgt_o from " << cloud_tgt_o->size() << "to" << cloud_tgt->size() << endl;

	//������淨��
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_src;
	ne_src.setInputCloud(cloud_src);
	pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
	ne_src.setSearchMethod(tree_src);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
	ne_src.setRadiusSearch(0.02);
	ne_src.compute(*cloud_src_normals);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_tgt;
	ne_tgt.setInputCloud(cloud_tgt);
	pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
	ne_tgt.setSearchMethod(tree_tgt);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
	//ne_tgt.setKSearch(20);
	ne_tgt.setRadiusSearch(0.02);
	ne_tgt.compute(*cloud_tgt_normals);


	//����3dsc


	pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal, pcl::ShapeContext1980> sp_tgt;
	sp_tgt.setInputCloud(cloud_tgt);
	sp_tgt.setInputNormals(cloud_tgt_normals);
	//kdTree����
	pcl::search::KdTree<PointT>::Ptr tree_tgt_sp(new pcl::search::KdTree<PointT>);
	sp_tgt.setSearchMethod(tree_tgt_sp);
	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_tgt(new pcl::PointCloud<pcl::ShapeContext1980>());
	sp_tgt.setRadiusSearch(0.5);
	sp_tgt.compute(*sps_tgt);

	cout << "compute *cloud_tgt_sps" << endl;

	pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal, pcl::ShapeContext1980> sp_src;
	sp_src.setInputCloud(cloud_src);
	sp_src.setInputNormals(cloud_src_normals);
	//kdTree����
	pcl::search::KdTree<PointT>::Ptr tree_src_sp(new pcl::search::KdTree<PointT>);
	sp_src.setSearchMethod(tree_src_sp);
	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_src(new pcl::PointCloud<pcl::ShapeContext1980>());
	sp_src.setRadiusSearch(0.5);
	sp_src.compute(*sps_src);

	cout << "compute *cloud_tgt_sps" << endl;



	//SAC��׼
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::ShapeContext1980> scia;
	scia.setInputSource(cloud_src);
	scia.setInputTarget(cloud_tgt);
	scia.setSourceFeatures(sps_src);
	scia.setTargetFeatures(sps_tgt);
	//scia.setMinSampleDistance(1);
	//scia.setNumberOfSamples(2);
	//scia.setCorrespondenceRandomness(20);
	PointCloud::Ptr sac_result(new PointCloud);
	scia.align(*sac_result);
	std::cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	Eigen::Matrix4f sac_trans;
	sac_trans = scia.getFinalTransformation();
	std::cout << sac_trans << endl;
	//pcl::io::savePCDFileASCII("bunny_transformed_sac.pcd", *sac_result);
	clock_t sac_time = clock();

	//icp��׼
	PointCloud::Ptr icp_result(new PointCloud);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_src);
	icp.setInputTarget(cloud_tgt_o);
	//Set the max correspondence distance to 4cm (e.g., correspondences with higher distances will be ignored)
	icp.setMaxCorrespondenceDistance(0.04);
	// ����������
	icp.setMaximumIterations(100);
	// ���α仯����֮��Ĳ�ֵ
	icp.setTransformationEpsilon(1e-10);
	// �������
	icp.setEuclideanFitnessEpsilon(0.01);
	icp.align(*icp_result, sac_trans);

	clock_t end = clock();
	cout << "total time: " << (double)(end - start) / (double)CLOCKS_PER_SEC << " s" << endl;

	cout << "sac time: " << (double)(sac_time - start) / (double)CLOCKS_PER_SEC << " s" << endl;
	cout << "icp time: " << (double)(end - sac_time) / (double)CLOCKS_PER_SEC << " s" << endl;

	std::cout << "ICP has converged:" << icp.hasConverged()
		<< " score: " << icp.getFitnessScore() << std::endl;
	Eigen::Matrix4f icp_trans;
	icp_trans = icp.getFinalTransformation();
	//cout<<"ransformationProbability"<<icp.getTransformationProbability()<<endl;
	std::cout << icp_trans << endl;
	//ʹ�ô����ı任��δ���˵�������ƽ��б任
	pcl::transformPointCloud(*cloud_src_o, *icp_result, icp_trans);
	//����ת�����������
	//pcl::io::savePCDFileASCII("_transformed_sac_ndt.pcd", *icp_result);

	//�������
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
	cout << "����ʵ����ת�Ƕ�:\n" << ANGLE_origin << endl;
	cout << "x����ת��� : " << a_error_x << "  y����ת��� : " << a_error_y << "  z����ת��� : " << a_error_z << endl;

	cout << "����ʵ��ƽ�ƾ���:\n" << TRANS_origin << endl;
	t_error_x = fabs(icp_trans(0, 3)) - fabs(TRANS_origin(0));
	t_error_y = fabs(icp_trans(1, 3)) - fabs(TRANS_origin(1));
	t_error_z = fabs(icp_trans(2, 3)) - fabs(TRANS_origin(2));
	cout << "����õ���ƽ�ƾ���" << endl << "x��ƽ��" << icp_trans(0, 3) << endl << "y��ƽ��" << icp_trans(1, 3) << endl << "z��ƽ��" << icp_trans(2, 3) << endl;
	cout << "x��ƽ����� : " << t_error_x << "  y��ƽ����� : " << t_error_y << "  z��ƽ����� : " << t_error_z << endl;
	
	//���ӻ�
	visualize_pcd(cloud_src_o, cloud_tgt_o, icp_result);
	return (0);
}
