/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified: 2022-01-18 17:17
#
# Filename: gicp_registration.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_REGISTRATION_GICP_REGISTRATION_HPP_
#define VSION_LOCALIZATION_MODELS_REGISTRATION_GICP_REGISTRATION_HPP_

#include "models/registration/registration_interface.hpp"
#include "tools/convert_matrix.hpp"

#include "sensor_data/cloud_data.hpp"
#include <pcl/features/normal_3d.h>
#include <pcl/registration/gicp.h>
//#include <pcl/common/impl/io.hpp>

namespace vision_localization
{
class GICPRegistration : public RegistrationInterface
{
public:
  GICPRegistration(const YAML::Node &node);
  GICPRegistration(float fit_eps, float trans_eps, float max_dist, int max_iter);

  bool SetInputTarget(const CloudData::CLOUD_PTR &input_target) override;
  bool ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                 CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose) override;
  float GetFitnessScore() override;

private:
  bool SetRegistrationParam(float fit_eps, float trans_eps, float max_dist, int max_iter);

private:
  // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal>::Ptr icp_ptr_;
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>::Ptr icp_ptr_;
};
}  // namespace vision_localization

#endif
