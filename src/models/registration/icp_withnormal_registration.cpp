/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified: 2022-01-18 17:17
#
# Filename: icp_withnormal_registration.cpp
#
# Description:
#
************************************************/

#include "models/registration/icp_withnormal_registration.hpp"
#include "global_defination/message_print.hpp"
#include <fstream>
#include <pcl/io/pcd_io.h>

namespace vision_localization
{

ICPNormalRegistration::ICPNormalRegistration(const YAML::Node &node)
    : icp_ptr_(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZINormal, pcl::PointXYZINormal>())
{
  float fit_eps = node["fit_eps"].as<float>();
  float trans_eps = node["trans_eps"].as<float>();
  float max_dist = node["max_dist"].as<float>();
  int max_iter = node["max_iter"].as<int>();

  SetRegistrationParam(fit_eps, trans_eps, max_dist, max_iter);
}

ICPNormalRegistration::ICPNormalRegistration(float fit_eps, float trans_eps, float max_dist, int max_iter)
    : icp_ptr_(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZINormal, pcl::PointXYZINormal>())
{
  SetRegistrationParam(fit_eps, trans_eps, max_dist, max_iter);
}

#include <pcl/registration/correspondence_rejection.h>
bool ICPNormalRegistration::SetRegistrationParam(float fit_eps, float trans_eps, float max_dist, int max_iter)
{
  //算法已经收敛之前，设置 ICP 循环中两个连续步骤之间允许的最大欧几里得误差。
  //误差估计为欧几里得意义上的对应之间差异的总和除以对应数量。
  icp_ptr_->setEuclideanFitnessEpsilon(fit_eps);
  //转换 epsilon（两个连续转换之间的最大允许差异），以便将优化视为已收敛到最终解决方案。
  icp_ptr_->setTransformationEpsilon(trans_eps);
  icp_ptr_->setMaxCorrespondenceDistance(max_dist);
  icp_ptr_->setMaximumIterations(max_iter);

  std::cout << "ICP_withnormal 的匹配参数为： fit_eps: " << fit_eps << ", "
            << "trans_eps: " << trans_eps << ", "
            << "max_dist: " << max_dist << ", "
            << "max_iter: " << max_iter << std::endl
            << std::endl;

  return true;
}
bool ICPNormalRegistration::SetInputTarget(const CloudData::CLOUD_PTR &input_target)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  GetNormals(input_target, normals, 0.3);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::concatenateFields(*input_target, *normals, *cloud);
  for (std::size_t i = 0; i < cloud->size(); i++) {
    if (!pcl::isFinite(cloud->at(i))) {
      std::cout << "[SetTarget] infinite point =" << cloud->at(i).x << " " << cloud->at(i).y << " " << cloud->at(i).z
                << " " << cloud->at(i).intensity << " " << cloud->at(i).normal_x << " " << cloud->at(i).normal_y << " "
                << cloud->at(i).normal_z << " " << cloud->at(i).curvature << std::endl;
    }
  }
  icp_ptr_->setInputTarget(cloud);

  return true;
}

bool ICPNormalRegistration::ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                                      CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose)
{
  input_source_ = input_source;

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  GetNormals(input_source, normals, 0.3);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::concatenateFields(*input_source, *normals, *cloud);
  for (std::size_t i = 0; i < cloud->size(); i++) {
    if (!pcl::isFinite(cloud->at(i))) {
      std::cout << "[SetSource] infinite point =" << cloud->at(i).x << " " << cloud->at(i).y << " " << cloud->at(i).z
                << " " << cloud->at(i).intensity << " " << cloud->at(i).normal_x << " " << cloud->at(i).normal_y << " "
                << cloud->at(i).normal_z << " " << cloud->at(i).curvature << std::endl;
    }
  }
  icp_ptr_->setInputSource(cloud);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr tmp_ptr(new pcl::PointCloud<pcl::PointXYZINormal>);
  icp_ptr_->align(*tmp_ptr, predict_pose.matrix());
  pcl::copyPointCloud(*tmp_ptr, *result_cloud_ptr);

  result_pose = icp_ptr_->getFinalTransformation();

  return icp_ptr_->hasConverged();
}

float ICPNormalRegistration::GetFitnessScore()
{
  return icp_ptr_->getFitnessScore();
}

}  // namespace vision_localization
