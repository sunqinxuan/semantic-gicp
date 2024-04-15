/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2022-03-02 15:05
#
# Filename:		sgicp_registration.cpp
#
# Description:
#
************************************************/

#include "models/registration/sgicp_registration.hpp"
#include "global_defination/message_print.hpp"
#include "tools/convert_matrix.hpp"
#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace vision_localization
{
SGICPRegistration::SGICPRegistration(const YAML::Node &node)
{
  min_dist_ = node["corr_dist"][0].as<float>();
  max_dist_ = node["corr_dist"][1].as<float>();
  max_iterations_ = node["max_iter"].as<int>();

  translation_threshold_ = node["trans_eps"].as<float>();
  rotation_threshold_ = 0.99999;

  mse_threshold_relative_ = node["fit_eps"].as<float>();
  mse_threshold_absolute_ = 1e-10;

  sgicp_epsilon_ = node["epsilon"].as<float>();
  num_neighbors_cov_ = node["num_neighbors_cov"].as<int>();

  std::cout << "SGICP 的匹配参数为： fit_eps: " << mse_threshold_relative_ << ", "
            << "trans_eps: " << translation_threshold_ << ", "
            << "max_dist: " << max_dist_ << ", "
            << "min_dist: " << min_dist_ << ", "
            << "max_iter: " << max_iterations_ << std::endl
            << std::endl;

  rigid_transformation_estimation_ = [this](const VecCloud &cloud_src, const std::vector<std::vector<int>> &indices_src,
                                            const VecCloud &cloud_tgt, const std::vector<std::vector<int>> &indices_tgt,
                                            Eigen::Isometry3f &transformation) {
    estimateRigidTransformationBFGS(cloud_src, indices_src, cloud_tgt, indices_tgt, transformation);
  };
}

bool SGICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR &input_target)
{
  input_target_ = input_target;
  CloudClassify(input_target, input_target_group_, input_target_group_kdtree_, input_target_group_empty,
                input_target_group_covariance_);

  return true;
}

bool SGICPRegistration::CloudClassify(const CloudData::CLOUD_PTR &input, VecCloud &group, VecKdTreePtr &group_kdtree,
                                      std::vector<bool> &empty, std::vector<VecMatPtr> &group_cov)
{
  size_t point_num = input->points.size();
  if (point_num < 10) {
    ERROR("[SGICP][CloudClassify] not sufficient points in input cloud!");
    return false;
  }
  group.resize(SEMANTIC_NUMS);
  group_kdtree.resize(SEMANTIC_NUMS);
  empty.resize(SEMANTIC_NUMS);
  group_cov.resize(SEMANTIC_NUMS);

  for (int i = 0; i < SEMANTIC_NUMS; i++) {
    group[i].reset(new CloudData::CLOUD());
  }
  for (size_t i = 0; i < point_num; i++) {
    int class_id = floorf(input->points[i].intensity);
    group[class_id]->points.push_back(input->points[i]);
  }
  for (int i = 0; i < SEMANTIC_NUMS; i++) {
    if (group[i]->points.size() < static_cast<std::size_t>(num_neighbors_cov_)) {
      empty[i] = true;
    } else {
      empty[i] = false;
      group_kdtree[i].reset(new KdTree());
      group_kdtree[i]->setInputCloud(group[i]);
      group_cov[i].reset(new VecMat);
      group_cov[i]->resize(group[i]->size());
      if (!ComputeCovariance(group[i], group_kdtree[i], group_cov[i])) empty[i] = true;
    }
  }
  return true;
}

bool SGICPRegistration::ComputeCovariance(const CloudData::CLOUD_PTR &cloud, const KdTreePtr &tree, VecMatPtr &covs)
{
  if (static_cast<std::size_t>(num_neighbors_cov_) > cloud->size()) {
    ERROR("[SGICP][ComputeCovariance] input cloud is too small!");
    return false;
  }

  Eigen::Vector3f mean;
  std::vector<int> nn_indices;
  nn_indices.reserve(num_neighbors_cov_);
  std::vector<float> nn_dist_sq;
  nn_dist_sq.reserve(num_neighbors_cov_);

  if (covs->size() != cloud->size()) {
    covs->resize(cloud->size());
  }

  VecMat::iterator it_cov = covs->begin();
  for (auto it_pt = cloud->begin(); it_pt != cloud->end(); ++it_pt, ++it_cov) {
    const CloudData::POINTXYZI &query = *it_pt;
    Eigen::Matrix3f &cov = *it_cov;
    cov.setZero();
    mean.setZero();

    // search for the num_neighbors_cov_ nearest neighbours;
    tree->nearestKSearch(query, num_neighbors_cov_, nn_indices, nn_dist_sq);

    for (int j = 0; j < num_neighbors_cov_; j++) {
      const CloudData::POINTXYZI &pt = (*cloud)[nn_indices[j]];
      mean(0) += pt.x;
      mean(1) += pt.y;
      mean(2) += pt.z;
      // left-bottom triangle of cov matrix;
      cov(0, 0) += pt.x * pt.x;
      cov(1, 0) += pt.y * pt.x;
      cov(1, 1) += pt.y * pt.y;
      cov(2, 0) += pt.z * pt.x;
      cov(2, 1) += pt.z * pt.y;
      cov(2, 2) += pt.z * pt.z;
    }
    mean /= static_cast<float>(num_neighbors_cov_);
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l <= k; l++) {
        cov(k, l) /= static_cast<float>(num_neighbors_cov_);
        cov(k, l) -= mean[k] * mean[l];
        cov(l, k) = cov(k, l);
      }
    }
    // compute the SVD (symmetric -> EVD);
    // singular values sorted in decreasing order;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
    cov.setZero();
    Eigen::Matrix3f U = svd.matrixU();
    // reconstitute the covariance matrx with modified singular values;
    for (int k = 0; k < 3; k++) {
      Eigen::Vector3f col = U.col(k);
      float v = sgicp_epsilon_;  // smallest two singular value replaced by sgicp_epsilon_;
      if (k == 0) v = 1.;        // biggest singular value replaced by 1;
      cov += v * col * col.transpose();
    }
  }
  return true;
}

bool SGICPRegistration::ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                                  CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose)
{
  // input_source_ = input_source;
  max_corr_dist_ = max_dist_;

  int nr_iterations_ = 0;
  converged_ = false;
  transformation_.setIdentity();
  group_informations_.resize(SEMANTIC_NUMS);

  // pre-process input source:
  CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
  pcl::transformPointCloud(*input_source, *transformed_input_source, predict_pose);

  CloudClassify(transformed_input_source, input_source_group_, input_source_group_kdtree_, input_source_group_empty,
                input_source_group_covariance_);

  std::vector<std::vector<int>> source_indices_group(SEMANTIC_NUMS);
  std::vector<std::vector<int>> target_indices_group(SEMANTIC_NUMS);

  while (!converged_) {
    correspondences_cur_mse_ = 0.0;
    correspondences_cur_max_ = 0.0;
    int cnt = 0, pc_cnt = 0;
    for (int i = 0; i < SEMANTIC_NUMS; i++) {
      if (input_target_group_empty[i] || input_source_group_empty[i]) {
        source_indices_group[i].resize(0);
        target_indices_group[i].resize(0);
        continue;
      }
      GetCorrespondence(input_source_group_[i], input_target_group_kdtree_[i], input_source_group_covariance_[i],
                        input_target_group_covariance_[i], source_indices_group[i], target_indices_group[i],
                        group_informations_[i]);
      pc_cnt += input_source_group_[i]->size();
      cnt += source_indices_group[i].size();
    }
    correspondences_cur_mse_ /= float(cnt);
    UpdateCorrMaxDist(correspondences_cur_mse_, correspondences_cur_max_, pc_cnt, cnt);

    previous_transformation_ = transformation_;
    //		WARNING("[ScanMatch] transformation_=\n",transformation_.matrix());
    rigid_transformation_estimation_(input_source_group_, source_indices_group, input_target_group_,
                                     target_indices_group, transformation_);
    //		WARNING("[ScanMatch] transformation_=\n",transformation_.matrix());
    Converter::NormTransformMatrix(transformation_);

    nr_iterations_++;

    if (hasConverged(transformation_) || nr_iterations_ >= max_iterations_) {
      converged_ = true;
      previous_transformation_ = transformation_;
    }
  }
  result_pose = previous_transformation_ * predict_pose;
  Converter::NormTransformMatrix(result_pose);

  align_cloud_.reset(new CloudData::CLOUD());
  pcl::transformPointCloud(*input_source, *align_cloud_, result_pose);
  result_cloud_ptr = align_cloud_;

  // base_transformation_ = predict_pose;
  cloud_transformation_ = previous_transformation_;
  input_source_.reset(new CloudData::CLOUD());
  informations_.resize(0);
  // cost_min_ = 0.0;
  int cnt = 0;
  //	DEBUG("[ScanMatch] previous_transformation_=\n",previous_transformation_.matrix());
  //	DEBUG("[ScanMatch] transformation_=\n",transformation_.matrix());
  //	DEBUG("[ScanMatch] cost_min_=",cost_min_);
  for (std::size_t i = 0; i < SEMANTIC_NUMS; i++) {
    CloudData::CLOUD_PTR tmp(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_group_[i], *tmp, predict_pose.inverse());
    for (std::size_t j = 0; j < source_indices_group[i].size(); j++) {
      std::size_t idx = source_indices_group[i][j];
      // std::size_t idx_tgt = target_indices_group[i][j];
      input_source_->push_back(tmp->at(idx));
      informations_.push_back(group_informations_[i][idx]);

      // Eigen::Vector3f pt_src(input_source_group_[i]->at(idx).x, input_source_group_[i]->at(idx).y,
      //                       input_source_group_[i]->at(idx).z);
      // Eigen::Vector3f pt_tgt(input_target_group_[i]->at(idx_tgt).x, input_target_group_[i]->at(idx_tgt).y,
      //                       input_target_group_[i]->at(idx_tgt).z);
      // Eigen::Vector3f residual = pt_tgt - transformation_ * pt_src;
      // cost_min_ += residual.dot(group_informations_[i][idx] * residual);
    }
    cnt += source_indices_group[i].size();
  }
  // cost_min_ /= float(cnt - 6);
  cost_min_ = GetFitnessScore();
  // DEBUG("[ScanMatch] cost_min_=", cost_min_);
  has_hessian_computed_ = false;

  return true;
}

float SGICPRegistration::GetFitnessScore()
{
  float max_range = std::numeric_limits<float>::max();
  float fitness_score = 0.0;

  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);
  // For each point in the source dataset
  int nr = 0;
  size_t pc_size = align_cloud_->size();
  for (size_t i = 0; i < pc_size; ++i) {
    // Find its nearest neighbor in the target
    int class_id = floorf(align_cloud_->at(i).intensity);
    if (input_target_group_empty[class_id]) {
      continue;
    }
    input_target_group_kdtree_[class_id]->nearestKSearch(align_cloud_->at(i), 1, nn_indices, nn_dists);
    // Deal with occlusions (incomplete targets)
    if (nn_dists[0] <= max_range) {
      // Add to the fitness score
      fitness_score += nn_dists[0];
      nr++;
    }
  }
  if (nr > 0)
    return (fitness_score / nr);
  else
    return (std::numeric_limits<float>::max());
}
int SGICPRegistration::GetCorrespondence(const CloudData::CLOUD_PTR &source_cloud, const KdTreePtr &target_kdtree,
                                         const VecMatPtr &source_covs, const VecMatPtr &target_covs,
                                         std::vector<int> &source_indices, std::vector<int> &target_indices,
                                         VecMat &infos)
{
  const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;
  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);

  std::size_t cnt = 0;
  source_indices.resize(source_cloud->size());
  target_indices.resize(source_cloud->size());
  infos.resize(source_cloud->size());
  Eigen::Matrix3f trans_R = transformation_.linear();

  for (std::size_t i = 0; i < source_cloud->size(); i++) {
    CloudData::POINTXYZI query = source_cloud->at(i);
    query.getVector4fMap() = transformation_ * query.getVector4fMap();

    if (target_kdtree->nearestKSearch(query, 1, nn_indices, nn_dists) == 0) {
      continue;
    }

    float dist = sqrt(nn_dists[0]);
    correspondences_cur_mse_ += dist;
    if (dist > correspondences_cur_max_) correspondences_cur_max_ = dist;

    if (nn_dists[0] < MAX_CORR_DIST_SQR) {
      Eigen::Matrix3f &C1 = (*source_covs)[i];
      Eigen::Matrix3f &C2 = (*target_covs)[nn_indices[0]];
      Eigen::Matrix3f &M = infos[i];
      M = trans_R * C1;
      Eigen::Matrix3f tmp = M * trans_R.transpose();
      tmp += C2;
      M = tmp.inverse();
      source_indices[cnt] = static_cast<int>(i);
      target_indices[cnt] = nn_indices[0];
      cnt++;
    }
  }
  source_indices.resize(cnt);
  target_indices.resize(cnt);

  return cnt;
}

bool SGICPRegistration::hasConverged(const Eigen::Isometry3f &transformation)
{
  // 1. The epsilon (difference) between the previous transformation and the current estimated transformation
  // a. translation magnitude -- squaredNorm:
  float translation_sqr = transformation.translation().squaredNorm();
  // b. rotation magnitude -- angle:
  float cos_angle = (transformation.linear().trace() - 1.0f) / 2.0f;
  if (cos_angle >= rotation_threshold_ && translation_sqr <= translation_threshold_) {
    return true;
  }
  // 3. The relative sum of Euclidean squared errors is smaller than a user defined threshold
  // Absolute
  if (fabs(correspondences_cur_mse_ - correspondences_prev_mse_) < mse_threshold_absolute_) {
    return true;
  }
  // Relative
  if (fabs(correspondences_cur_mse_ - correspondences_prev_mse_) / correspondences_prev_mse_ <
      mse_threshold_relative_) {
    return true;
  }

  correspondences_prev_mse_ = correspondences_cur_mse_;
  return false;
}
bool SGICPRegistration::UpdateCorrMaxDist(const float &mean_dist, const float &max_dist, const size_t &pc_size,
                                          const size_t &inline_size)
{
  if (pc_size < 10) return false;
  float inline_rate = static_cast<float>(float(inline_size) / pc_size);

  if (inline_rate < 0.7) return false;

  float corr_dist = mean_dist * (0.5 + inline_rate) + max_dist * (1 - inline_rate);
  corr_dist = corr_dist > max_dist_ ? max_dist_ : corr_dist;
  max_corr_dist_ = corr_dist < min_dist_ ? min_dist_ : corr_dist;

  return true;
}

void SGICPRegistration::ApplyState(Eigen::Isometry3f &t, const Vector6f &x) const
{
  // Z Y X euler angles convention
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(x[5]), Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(static_cast<float>(x[4]), Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(static_cast<float>(x[3]), Eigen::Vector3f::UnitX());
  t.linear() = R * t.linear();
  t.translation()(0) += x[0];
  t.translation()(1) += x[1];
  t.translation()(2) += x[2];
}

void SGICPRegistration::ComputeRDerivative(const Vector6f &x, const Eigen::Matrix3f &R, Vector6f &g) const
{
  Eigen::Matrix3f dR_dPhi;
  Eigen::Matrix3f dR_dTheta;
  Eigen::Matrix3f dR_dPsi;

  float phi = x[3], theta = x[4], psi = x[5];

  float cphi = std::cos(phi), sphi = sin(phi);
  float ctheta = std::cos(theta), stheta = sin(theta);
  float cpsi = std::cos(psi), spsi = sin(psi);

  dR_dPhi(0, 0) = 0.;
  dR_dPhi(1, 0) = 0.;
  dR_dPhi(2, 0) = 0.;

  dR_dPhi(0, 1) = sphi * spsi + cphi * cpsi * stheta;
  dR_dPhi(1, 1) = -cpsi * sphi + cphi * spsi * stheta;
  dR_dPhi(2, 1) = cphi * ctheta;

  dR_dPhi(0, 2) = cphi * spsi - cpsi * sphi * stheta;
  dR_dPhi(1, 2) = -cphi * cpsi - sphi * spsi * stheta;
  dR_dPhi(2, 2) = -ctheta * sphi;

  dR_dTheta(0, 0) = -cpsi * stheta;
  dR_dTheta(1, 0) = -spsi * stheta;
  dR_dTheta(2, 0) = -ctheta;

  dR_dTheta(0, 1) = cpsi * ctheta * sphi;
  dR_dTheta(1, 1) = ctheta * sphi * spsi;
  dR_dTheta(2, 1) = -sphi * stheta;

  dR_dTheta(0, 2) = cphi * cpsi * ctheta;
  dR_dTheta(1, 2) = cphi * ctheta * spsi;
  dR_dTheta(2, 2) = -cphi * stheta;

  dR_dPsi(0, 0) = -ctheta * spsi;
  dR_dPsi(1, 0) = cpsi * ctheta;
  dR_dPsi(2, 0) = 0.;

  dR_dPsi(0, 1) = -cphi * cpsi - sphi * spsi * stheta;
  dR_dPsi(1, 1) = -cphi * spsi + cpsi * sphi * stheta;
  dR_dPsi(2, 1) = 0.;

  dR_dPsi(0, 2) = cpsi * sphi - cphi * spsi * stheta;
  dR_dPsi(1, 2) = sphi * spsi + cphi * cpsi * stheta;
  dR_dPsi(2, 2) = 0.;

  g[3] = MatricesInnerProd(dR_dPhi, R);
  g[4] = MatricesInnerProd(dR_dTheta, R);
  g[5] = MatricesInnerProd(dR_dPsi, R);
}

inline float SGICPRegistration::MatricesInnerProd(const Eigen::MatrixXf &mat1, const Eigen::MatrixXf &mat2) const
{
  float r = 0.;
  std::size_t n = mat1.rows();
  // tr(mat1*mat2)
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) r += mat1(j, i) * mat2(i, j);
  return r;
}

// rigid_transformation_estimation_
void SGICPRegistration::estimateRigidTransformationBFGS(const VecCloud &cloud_src,
                                                        const std::vector<std::vector<int>> &indices_src,
                                                        const VecCloud &cloud_tgt,
                                                        const std::vector<std::vector<int>> &indices_tgt,
                                                        Eigen::Isometry3f &transformation)
{
  // Set the initial solution
  Vector6f x = Vector6f::Zero();
  x[0] = transformation.translation()(0);
  x[1] = transformation.translation()(1);
  x[2] = transformation.translation()(2);
  x[3] = std::atan2(transformation.linear()(2, 1), transformation.linear()(2, 2));
  x[4] = asin(-transformation.linear()(2, 0));
  x[5] = std::atan2(transformation.linear()(1, 0), transformation.linear()(0, 0));

  // Set temporary pointers
  tmp_src_ = &cloud_src;
  tmp_tgt_ = &cloud_tgt;
  tmp_idx_src_ = &indices_src;
  tmp_idx_tgt_ = &indices_tgt;

  // Optimize using forward-difference approximation LM
  const float gradient_tol = 1e-2;
  OptimizationFunctor functor(this);
  BFGS<OptimizationFunctor> bfgs(functor);
  bfgs.parameters.sigma = 0.01;
  bfgs.parameters.rho = 0.01;
  bfgs.parameters.tau1 = 9;
  bfgs.parameters.tau2 = 0.05;
  bfgs.parameters.tau3 = 0.5;
  bfgs.parameters.order = 3;

  int inner_iterations_ = 0;
  int result = bfgs.minimizeInit(x);
  result = BFGSSpace::Running;
  do {
    inner_iterations_++;
    result = bfgs.minimizeOneStep(x);
    if (result) {
      break;
    }
    result = bfgs.testGradient(gradient_tol);
  } while (result == BFGSSpace::Running && inner_iterations_ < max_inner_iterations_);
  if (result == BFGSSpace::NoProgress || result == BFGSSpace::Success || inner_iterations_ == max_inner_iterations_) {
    transformation.setIdentity();
    ApplyState(transformation, x);
  } else
    ERROR("[SGICP][estimateRigidTransformationBFGS] BFGS solver does not converge!");
}

inline double SGICPRegistration::OptimizationFunctor::operator()(const Vector6f &x)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  sgicp_->ApplyState(transformation_matrix, x);
  float f = 0;
  int N = 0;
  for (int k = 0; k < sgicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*sgicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*sgicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*sgicp_->tmp_idx_tgt_)[k][i];
      CloudData::POINTXYZI pt_src = (*sgicp_->tmp_src_)[k]->points[idx_src];
      CloudData::POINTXYZI pt_tgt = (*sgicp_->tmp_tgt_)[k]->points[idx_tgt];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);
      // Estimate the distance (cost function)
      Eigen::Vector3f res = pp - p_tgt;
      Eigen::Vector3f temp(sgicp_->group_informations_[k][idx_src] * res);
      // increment= res'*temp/num_matches = temp'*M*temp/num_matches (we postpone
      // 1/num_matches after the loop closes)
      f += float(res.transpose() * temp);
    }
    N += m;
  }
  return f / float(N);
}

inline void SGICPRegistration::OptimizationFunctor::df(const Vector6f &x, Vector6f &g)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  sgicp_->ApplyState(transformation_matrix, x);
  g.setZero();
  Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
  int N = 0;
  for (int k = 0; k < sgicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*sgicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*sgicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*sgicp_->tmp_idx_tgt_)[k][i];
      CloudData::POINTXYZI pt_src = (*sgicp_->tmp_src_)[k]->points[idx_src];
      CloudData::POINTXYZI pt_tgt = (*sgicp_->tmp_tgt_)[k]->points[idx_tgt];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);
      Eigen::Vector3f res = pp - p_tgt;
      Eigen::Vector3f temp(sgicp_->group_informations_[k][idx_src] * res);
      g.head<3>() += temp;
      R += p_src * temp.transpose();
    }
    N += m;
  }
  g.head<3>() *= 2.0 / float(N);
  R *= 2.0 / float(N);
  sgicp_->ComputeRDerivative(x, R, g);
}

inline void SGICPRegistration::OptimizationFunctor::fdf(const Vector6f &x, float &f, Vector6f &g)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  sgicp_->ApplyState(transformation_matrix, x);
  f = 0;
  g.setZero();
  Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
  int N = 0;
  for (int k = 0; k < sgicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*sgicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*sgicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*sgicp_->tmp_idx_tgt_)[k][i];
      CloudData::POINTXYZI pt_src = (*sgicp_->tmp_src_)[k]->points[idx_src];
      CloudData::POINTXYZI pt_tgt = (*sgicp_->tmp_tgt_)[k]->points[idx_tgt];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);
      Eigen::Vector3f res = pp - p_tgt;
      Eigen::Vector3f temp(sgicp_->group_informations_[k][idx_src] * res);
      f += float(res.transpose() * temp);
      g.head<3>() += temp;
      R += p_src * temp.transpose();
    }
    N += m;
  }
  f /= float(N);
  g.head<3>() *= float(2.0 / N);
  R *= 2.0 / float(N);
  sgicp_->ComputeRDerivative(x, R, g);
}
}  // namespace vision_localization
