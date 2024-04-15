/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2022-06-08 09:44
#
# Filename:		plicp_registration.cpp
#
# Description:
#
************************************************/

#include "models/registration/plicp_registration.hpp"
#include "global_defination/message_print.hpp"
#include "tools/convert_matrix.hpp"
#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace vision_localization
{
PLICPRegistration::PLICPRegistration(const YAML::Node &node)
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

  std::string method = node["line_feature_extraction_method"].as<std::string>();
  if (method == "region_grow") {
    line_extract_ptr_ = std::make_shared<LineFeatureExtractionRG>(node["line_region_grow"]);
  } else {
    std::cerr << "[PLICPRegistration] cannot find line extration method: " << method << std::endl;
  }
}

bool PLICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR &input_target)
{
  input_target_ = input_target;
  CloudClassify(input_target, input_target_group_, input_target_group_kdtree_, input_target_group_empty_,
                input_target_group_covariance_);

  input_target_group_lines_.resize(input_target_group_.size());
  input_target_group_lines_indices_.resize(input_target_group_.size());
  for (std::size_t i = 0; i < input_target_group_.size(); i++) {
    input_target_group_lines_[i].clear();
    input_target_group_lines_indices_[i].clear();
    input_target_group_lines_indices_[i].resize(input_target_group_[i]->size(), -1);
    if (input_target_group_empty_[i]) continue;
    // TODO i == 1 ||
    if (i == 2 || i == 4 || i == 5 || i == 6) {
      line_extract_ptr_->Extract(input_target_group_[i], i, input_target_group_lines_[i],
                                 input_target_group_lines_indices_[i]);
    }
  }

  // DEBUG
  // output line information: size and residual;
  // std::cout << std::endl;
  // DEBUG("input target size = ", input_target->size());
  // for (std::size_t i = 0; i < input_target_group_.size(); i++) {
  //  if (input_target_group_lines_[i].size() == 0) continue;
  //  std::cout << "==========================" << std::endl;
  //  std::cout << "class_id=" << i << std::endl;
  //  for (std::size_t j = 0; j < input_target_group_lines_[i].size(); j++) {
  //    LineFeature &line = input_target_group_lines_[i][j];
  //    std::cout << "\t" << j << "\t" << line.size << "\t" << line.residual << std::endl;
  //  }
  //}
  // std::cout << std::endl;
  //	getchar();

  // DEBUG
  // visualize extracted lines;
  cloud_dbg.reset(new CloudData::CLOUD());
  for (std::size_t i = 0; i < input_target_group_lines_indices_.size(); i++) {
    for (std::size_t j = 0; j < input_target_group_lines_indices_[i].size(); j++) {
      int idx = input_target_group_lines_indices_[i][j];
      if (idx < 0) continue;
      LineFeature &line = input_target_group_lines_[i][idx];
      Eigen::Vector3f pc = line.centroid;
      Eigen::Vector3f ep1 = line.endpoint_1;
      Eigen::Vector3f ep2 = line.endpoint_2;
      Eigen::Vector3f d = line.direction;
      CloudData::POINTXYZI pt = input_target_group_[i]->points[j];
      Eigen::Vector3f p(pt.x, pt.y, pt.z);
      pt.intensity = idx;
      // pt.intensity = float(d.cross(p - pc).norm());
      cloud_dbg->push_back(pt);
    }
  }

  return true;
}

bool PLICPRegistration::CloudClassify(const CloudData::CLOUD_PTR &input, VecCloud &group, VecKdTreePtr &group_kdtree,
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

bool PLICPRegistration::ComputeCovariance(const CloudData::CLOUD_PTR &cloud, const KdTreePtr &tree, VecMatPtr &covs)
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

bool PLICPRegistration::ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
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

  CloudClassify(transformed_input_source, input_source_group_, input_source_group_kdtree_, input_source_group_empty_,
                input_source_group_covariance_);

  // input_source_group_lines_.resize(input_source_group_.size());
  // input_source_group_lines_indices_.resize(input_source_group_.size());
  // for (std::size_t i = 0; i < input_source_group_.size(); i++) {
  //  input_source_group_lines_[i].clear();
  //  input_source_group_lines_indices_[i].clear();
  //  input_source_group_lines_indices_[i].resize(input_source_group_[i]->size(), -1);
  //  if (input_source_group_empty_[i]) continue;
  //  if (i == 2 || i == 4 || i == 5 || i == 1 || i == 6)
  //    line_extract_ptr_->Extract(input_source_group_[i], i, input_source_group_lines_[i],
  //                               input_source_group_lines_indices_[i]);
  //}

  std::vector<std::vector<int>> source_indices_group(SEMANTIC_NUMS);
  std::vector<std::vector<int>> target_indices_group(SEMANTIC_NUMS);

  while (!converged_) {
    correspondences_cur_mse_ = 0.0;
    correspondences_cur_max_ = 0.0;
    int cnt = 0, pc_cnt = 0;
    for (int i = 0; i < SEMANTIC_NUMS; i++) {
      if (input_target_group_empty_[i] || input_source_group_empty_[i]) {
        source_indices_group[i].resize(0);
        target_indices_group[i].resize(0);
        continue;
      }
      GetCorrespondence(input_source_group_[i], input_target_group_kdtree_[i], input_source_group_covariance_[i],
                        input_target_group_covariance_[i], input_target_group_lines_[i],
                        input_target_group_lines_indices_[i], source_indices_group[i], target_indices_group[i],
                        group_informations_[i]);
      pc_cnt += input_source_group_[i]->size();
      cnt += source_indices_group[i].size();
    }
    correspondences_cur_mse_ /= float(cnt);
    UpdateCorrMaxDist(correspondences_cur_mse_, correspondences_cur_max_, pc_cnt, cnt);

    previous_transformation_ = transformation_;
    rigid_transformation_estimation_(input_source_group_, source_indices_group, input_target_group_,
                                     target_indices_group, transformation_);
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
  CloudData::CLOUD_PTR tmp_cloud;
  for (int i = 0; i < SEMANTIC_NUMS; i++) {
    // transform input_source_group_;
    tmp_cloud.reset(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_group_[i], *tmp_cloud, predict_pose.inverse());
    for (std::size_t j = 0; j < source_indices_group[i].size(); j++) {
      std::size_t idx = source_indices_group[i][j];
      input_source_->push_back(tmp_cloud->at(idx));
      informations_.push_back(group_informations_[i][idx]);
    }
  }
  has_hessian_computed_ = false;

  // DEBUG
  // visualize the point-to-point and point-to-line residuals;
  // cloud_dbg.reset(new CloudData::CLOUD());
  // for (int i = 0; i < SEMANTIC_NUMS; i++) {
  // for (std::size_t j = 0; j < source_indices_group[i].size(); j++) {
  //   CloudData::CLOUD_PTR tmp_cloud(new CloudData::CLOUD());
  //   pcl::transformPointCloud(*input_source_group_[i], *tmp_cloud, previous_transformation_);
  //   std::size_t idx_src = source_indices_group[i][j];
  //   std::size_t idx_tgt = target_indices_group[i][j];
  //   CloudData::POINTXYZI pt_src = tmp_cloud->points[idx_src];
  //   CloudData::POINTXYZI pt_tgt = input_target_group_[i]->points[idx_tgt];
  //   int idx_line = input_target_group_lines_indices_[i][idx_tgt];
  //   if (idx_line < 0) {
  //     continue;
  //   } else {
  //     LineFeature &line = input_target_group_lines_[i][idx_line];
  //     Eigen::Vector3f pt(pt_src.x, pt_src.y, pt_src.z);
  //     pt_src.intensity = float(line.direction.cross(pt - line.centroid).norm());
  //   }
  //   cloud_dbg->push_back(pt_src);
  // }
  //}

  return true;
}

float PLICPRegistration::GetFitnessScore()
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
    if (input_target_group_empty_[class_id]) {
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
int PLICPRegistration::GetCorrespondence(const CloudData::CLOUD_PTR &source_cloud, const KdTreePtr &target_kdtree,
                                         const VecMatPtr &source_covs, const VecMatPtr &target_covs,
                                         const VecLine &target_lines, const std::vector<int> &target_line_indices,
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
    //		std::cout<<"query="<<query.x<<" "<<query.y<<std::endl;

    if (target_kdtree->nearestKSearch(query, 1, nn_indices, nn_dists) == 0) continue;
    // int num = target_kdtree->nearestKSearch(query, 1, nn_indices, nn_dists);
    //		std::cout<<"nearestKSearch="<<num<<std::endl;
    // if (num == 0) continue;
    //		std::cout<<"i="<<i<<", nn_indices[0]="<<nn_indices[0]<<std::endl;
    //		std::cout<<"nn_dists[0]="<<sqrt(nn_dists[0])<<std::endl;

    float dist = sqrt(nn_dists[0]);
    correspondences_cur_mse_ += dist;
    if (dist > correspondences_cur_max_) correspondences_cur_max_ = dist;

    // if (nn_dists[0] > MAX_CORR_DIST_SQR) continue;

    //		std::cout<<"tgt_line_idx="<<target_line_indices[nn_indices[0]]<<std::endl;
    //		std::cout<<"query="<<query.x<<" "<<query.y<<std::endl;
    //		std::cout<<"nn_dists[0]="<<sqrt(nn_dists[0])<<std::endl;

    // if (target_line_indices[nn_indices[0]] >= 0) {
    //  const LineFeature &line = target_lines[target_line_indices[nn_indices[0]]];
    //  if (Point2LineDistance(query, line) > max_corr_dist_) continue;
    //}

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

float PLICPRegistration::Point2LineDistance(const CloudData::POINTXYZI &pt, const LineFeature &line)
{
  Eigen::Vector3f p(pt.x, pt.y, pt.z);
  // Eigen::Vector3f tmp1 = p - line.endpoint_1;
  // Eigen::Vector3f tmp2 = p - line.endpoint_2;
  // Eigen::Vector3f den = line.endpoint_1 - line.endpoint_2;
  // return float(tmp1.cross(tmp2).norm() / den.norm());
  return float(line.direction.cross(p - line.centroid).norm());
}

bool PLICPRegistration::hasConverged(const Eigen::Isometry3f &transformation)
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
bool PLICPRegistration::UpdateCorrMaxDist(const float &mean_dist, const float &max_dist, const size_t &pc_size,
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

void PLICPRegistration::ApplyState(Eigen::Isometry3f &t, const Vector6f &x) const
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

void PLICPRegistration::ComputeRDerivative(const Vector6f &x, const Eigen::Vector3f &RX,
                                           Eigen::Matrix3f &dP_dtheta) const
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

  // g[3] = MatricesInnerProd(dR_dPhi, R);
  // g[4] = MatricesInnerProd(dR_dTheta, R);
  // g[5] = MatricesInnerProd(dR_dPsi, R);
  dP_dtheta.col(0) = dR_dPhi * RX;
  dP_dtheta.col(1) = dR_dTheta * RX;
  dP_dtheta.col(2) = dR_dPsi * RX;
}

// inline float PLICPRegistration::MatricesInnerProd(const Eigen::MatrixXf &mat1, const Eigen::MatrixXf &mat2) const
//{
//  float r = 0.;
//  std::size_t n = mat1.rows();
//  // tr(mat1^t.mat2)
//  for (std::size_t i = 0; i < n; i++)
//    for (std::size_t j = 0; j < n; j++) r += mat1(j, i) * mat2(i, j);
//  return r;
//}

// rigid_transformation_estimation_
void PLICPRegistration::estimateRigidTransformationBFGS(const VecCloud &cloud_src,
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
  tmp_lines_tgt_ = &input_target_group_lines_;
  tmp_idx_lines_tgt_ = &input_target_group_lines_indices_;
  tmp_lines_src_ = &input_source_group_lines_;
  tmp_idx_lines_src_ = &input_source_group_lines_indices_;

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

inline double PLICPRegistration::OptimizationFunctor::operator()(const Vector6f &x)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  plicp_->ApplyState(transformation_matrix, x);
  float f = 0;
  int N = 0;
  for (int k = 0; k < plicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*plicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*plicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*plicp_->tmp_idx_tgt_)[k][i];
      CloudData::POINTXYZI pt_src = (*plicp_->tmp_src_)[k]->points[idx_src];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);

      int idx_line_tgt = (*plicp_->tmp_idx_lines_tgt_)[k][idx_tgt];
      // int idx_line_src = (*plicp_->tmp_idx_lines_src_)[k][idx_src];

      // if (idx_line_tgt < 0 && idx_line_src < 0) {
      if (idx_line_tgt < 0) {
        CloudData::POINTXYZI pt_tgt = (*plicp_->tmp_tgt_)[k]->points[idx_tgt];
        Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
        Eigen::Vector3f res = pp - p_tgt;
        Eigen::Vector3f temp(plicp_->group_informations_[k][idx_src] * res);
        f += float(res.transpose() * temp);

        // point-to-point distance;
        // f += float(res.transpose() * res);
      } else {
        // Eigen::Vector3f pc = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].centroid;
        // Eigen::Vector3f v = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].direction;
        // Eigen::Vector3f dL = v.cross(pp - pc);
        // f += float(dL.transpose() * dL);

        Eigen::Vector3f ep_1 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_1;
        Eigen::Vector3f ep_2 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_2;
        Eigen::Vector3f tmp_1 = pp - ep_1;
        Eigen::Vector3f tmp_2 = pp - ep_2;
        Eigen::Vector3f den = ep_1 - ep_2;
        float dE = float(tmp_1.cross(tmp_2).norm() / den.norm());
        f += dE * dE;
      }
    }
    N += m;
  }
  return f / float(N);
}

inline void PLICPRegistration::OptimizationFunctor::df(const Vector6f &x, Vector6f &g)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  plicp_->ApplyState(transformation_matrix, x);
  g.setZero();
  int N = 0;
  for (int k = 0; k < plicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*plicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*plicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*plicp_->tmp_idx_tgt_)[k][i];
      CloudData::POINTXYZI pt_src = (*plicp_->tmp_src_)[k]->points[idx_src];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);

      int idx_line_tgt = (*plicp_->tmp_idx_lines_tgt_)[k][idx_tgt];
      // int idx_line_src = (*plicp_->tmp_idx_lines_src_)[k][idx_src];

      // if (idx_line_tgt < 0 && idx_line_src < 0) {
      if (idx_line_tgt < 0) {
        CloudData::POINTXYZI pt_tgt = (*plicp_->tmp_tgt_)[k]->points[idx_tgt];
        Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
        Eigen::Vector3f res = pp - p_tgt;
        Eigen::Vector3f temp(plicp_->group_informations_[k][idx_src] * res);

        g.head<3>() += temp;
        // g.head<3>() += 2.0 * res;

        Eigen::Matrix3f dP_dtheta;
        plicp_->ComputeRDerivative(x, p_src, dP_dtheta);
        g.tail<3>() += dP_dtheta.transpose() * temp;
        // g.tail<3>() += 2.0 * dP_dtheta.transpose() * res;
      } else {
        // Eigen::Vector3f pc = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].centroid;
        // Eigen::Vector3f v = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].direction;
        // Eigen::Vector3f dL = v.cross(pp - pc);
        //
        // Eigen::Vector3f ddL_dP;
        // ddL_dP[0] = (-v[2] * dL[1] + v[1] * dL[2]) / dL.norm();
        // ddL_dP[1] = (v[2] * dL[0] - v[0] * dL[2]) / dL.norm();
        // ddL_dP[2] = (-v[1] * dL[0] + v[0] * dL[1]) / dL.norm();
        // g.head<3>() += dL.norm() * ddL_dP;
        //
        // Eigen::Matrix3f dP_dtheta;
        // plicp_->ComputeRDerivative(x, pp, dP_dtheta);
        //// g.tail<3>() += ddL_dP.transpose() * dP_dtheta;
        // g.tail<3>() += dL.norm() * dP_dtheta.transpose() * ddL_dP;

        Eigen::Vector3f ep_1 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_1;
        Eigen::Vector3f ep_2 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_2;

        Eigen::Vector3f tmp_1 = pp - ep_1;
        Eigen::Vector3f tmp_2 = pp - ep_2;
        Eigen::Vector3f pc = tmp_1.cross(tmp_2);
        Eigen::Vector3f den = ep_1 - ep_2;
        float Lle = pc.norm() * den.norm();
        float dE = float(tmp_1.cross(tmp_2).norm() / den.norm());

        Eigen::Vector3f ddE_dt;
        ddE_dt[0] = (pc[1] * (ep_2[2] - ep_1[2]) + pc[2] * (ep_1[1] - ep_2[1])) / Lle;
        ddE_dt[1] = (pc[0] * (ep_1[2] - ep_2[2]) + pc[2] * (ep_2[0] - ep_1[0])) / Lle;
        ddE_dt[2] = (pc[0] * (ep_2[1] - ep_1[1]) + pc[1] * (ep_1[0] - ep_2[0])) / Lle;
        g.head<3>() += dE * ddE_dt;

        Eigen::Matrix3f dP_dtheta;
        plicp_->ComputeRDerivative(x, pp, dP_dtheta);
        g.tail<3>() += dE * ddE_dt.transpose() * dP_dtheta;
      }
    }
    N += m;
  }
  g *= 2.0 / float(N);
}

inline void PLICPRegistration::OptimizationFunctor::fdf(const Vector6f &x, float &f, Vector6f &g)
{
  Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();
  plicp_->ApplyState(transformation_matrix, x);
  f = 0;
  g.setZero();
  int N = 0;
  for (int k = 0; k < plicp_->SEMANTIC_NUMS; k++) {
    int m = static_cast<int>((*plicp_->tmp_idx_src_)[k].size());
    for (int i = 0; i < m; ++i) {
      int idx_src = (*plicp_->tmp_idx_src_)[k][i];
      int idx_tgt = (*plicp_->tmp_idx_tgt_)[k][i];
      // std::cout << "idx_src=" << idx_src << std::endl;
      // std::cout << "idx_tgt=" << idx_tgt << std::endl;
      CloudData::POINTXYZI pt_src = (*plicp_->tmp_src_)[k]->points[idx_src];
      Eigen::Vector3f p_src(pt_src.x, pt_src.y, pt_src.z);
      Eigen::Vector3f pp(transformation_matrix * p_src);
      // std::cout << "p_src=" << p_src.transpose() << std::endl;
      // std::cout << "pp=" << pp.transpose() << std::endl;

      int idx_line_tgt = (*plicp_->tmp_idx_lines_tgt_)[k][idx_tgt];
      // int idx_line_src = (*plicp_->tmp_idx_lines_src_)[k][idx_src];
      // std::cout << "idx_line_tgt=" << idx_line_tgt << std::endl;

      // if (idx_line_tgt < 0 && idx_line_src < 0) {
      if (idx_line_tgt < 0) {
        CloudData::POINTXYZI pt_tgt = (*plicp_->tmp_tgt_)[k]->points[idx_tgt];
        Eigen::Vector3f p_tgt(pt_tgt.x, pt_tgt.y, pt_tgt.z);
        Eigen::Vector3f res = pp - p_tgt;
        Eigen::Vector3f temp(plicp_->group_informations_[k][idx_src] * res);
        f += float(res.transpose() * temp);

        // std::cout << "f_p=" << float(res.transpose() * res) << std::endl;

        // point-to-point distance;
        // f += float(res.transpose() * res);

        g.head<3>() += temp;
        // g.head<3>() += 2.0 * res;

        Eigen::Matrix3f dP_dtheta;
        plicp_->ComputeRDerivative(x, p_src, dP_dtheta);
        g.tail<3>() += dP_dtheta.transpose() * temp;
        // g.tail<3>() += 2.0 * dP_dtheta.transpose() * res;

        // Eigen::Vector3f tmp = 2.0 * res;
        // std::cout << "g_p_t=" << tmp.transpose() << std::endl;
        // tmp = 2.0 * dP_dtheta.transpose() * res;
        // std::cout << "g_p_theta=" << tmp.transpose() << std::endl;

      } else {
        // Eigen::Vector3f pc = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].centroid;
        // Eigen::Vector3f v = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].direction;
        //// point-to-line distance;
        // Eigen::Vector3f dL = v.cross(pp - pc);
        // f += float(dL.transpose() * dL);
        //
        // Eigen::Vector3f ddL_dP;
        // ddL_dP[0] = (-v[2] * dL[1] + v[1] * dL[2]) / dL.norm();
        // ddL_dP[1] = (v[2] * dL[0] - v[0] * dL[2]) / dL.norm();
        // ddL_dP[2] = (-v[1] * dL[0] + v[0] * dL[1]) / dL.norm();
        // g.head<3>() += dL.norm() * ddL_dP;
        //
        // Eigen::Matrix3f dP_dtheta;
        // plicp_->ComputeRDerivative(x, pp, dP_dtheta);
        //// g.tail<3>() += ddL_dP.transpose() * dP_dtheta;
        // g.tail<3>() += dL.norm() * dP_dtheta.transpose() * ddL_dP;

        Eigen::Vector3f ep_1 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_1;
        Eigen::Vector3f ep_2 = (*plicp_->tmp_lines_tgt_)[k][idx_line_tgt].endpoint_2;

        Eigen::Vector3f tmp_1 = pp - ep_1;
        Eigen::Vector3f tmp_2 = pp - ep_2;
        Eigen::Vector3f den = ep_1 - ep_2;
        float dE = float(tmp_1.cross(tmp_2).norm() / den.norm());
        f += dE * dE;

        Eigen::Vector3f pc = tmp_1.cross(tmp_2);
        float Lle = pc.norm() * den.norm();

        Eigen::Vector3f ddE_dt;
        ddE_dt[0] = (pc[1] * (ep_2[2] - ep_1[2]) + pc[2] * (ep_1[1] - ep_2[1])) / Lle;
        ddE_dt[1] = (pc[0] * (ep_1[2] - ep_2[2]) + pc[2] * (ep_2[0] - ep_1[0])) / Lle;
        ddE_dt[2] = (pc[0] * (ep_2[1] - ep_1[1]) + pc[1] * (ep_1[0] - ep_2[0])) / Lle;
        g.head<3>() += dE * ddE_dt;

        Eigen::Matrix3f dP_dtheta;
        plicp_->ComputeRDerivative(x, pp, dP_dtheta);
        g.tail<3>() += dE * ddE_dt.transpose() * dP_dtheta;
      }
      // std::cout << std::endl;
    }
    N += m;
  }
  f /= float(N);
  g *= 2.0 / float(N);
}
}  // namespace vision_localization
