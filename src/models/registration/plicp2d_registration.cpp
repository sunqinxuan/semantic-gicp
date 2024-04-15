/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2022-07-05 09:12
#
# Filename:		plicp2d_registration.cpp
#
# Description:
#
************************************************/

#include "models/registration/plicp2d_registration.hpp"
#include "global_defination/message_print.hpp"
#include "tools/convert_matrix.hpp"
#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace vision_localization
{
PLICP2DRegistration::PLICP2DRegistration(const YAML::Node &node)
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

  // rigid_transformation_estimation_ = [this](const VecCloud &cloud_src, const std::vector<std::vector<int>>
  // &indices_src,
  //                                          const VecCloud &cloud_tgt, const std::vector<std::vector<int>>
  //                                          &indices_tgt, Eigen::Isometry3f &transformation) {
  //  estimateRigidTransformationBFGS(cloud_src, indices_src, cloud_tgt, indices_tgt, transformation);
  //};

  std::string method = node["line_feature_extraction_method"].as<std::string>();
  if (method == "region_grow") {
    line_extract_ptr_ = std::make_shared<LineFeatureExtractionRG>(node["line_region_grow"]);
  } else {
    std::cerr << "[PLICP2DRegistration] cannot find line extration method: " << method << std::endl;
  }

  max_num_iterations_ = node["ceres"]["max_num_iterations"].as<unsigned int>();
  minimizer_progress_to_stdout_ = node["ceres"]["minimizer_progress_to_stdout"].as<bool>();
  std::string solver_type = node["ceres"]["solver_type"].as<std::string>();
  if (solver_type == "sparse_normal_cholesky") {
    linear_solver_type_ = ceres::SPARSE_NORMAL_CHOLESKY;
  } else if (solver_type == "dense_normal_cholesky") {
    linear_solver_type_ = ceres::DENSE_NORMAL_CHOLESKY;
  } else if (solver_type == "sparse_schur") {
    linear_solver_type_ = ceres::SPARSE_SCHUR;
  } else if (solver_type == "dense_schur") {
    linear_solver_type_ = ceres::DENSE_SCHUR;
  } else {
    ERROR("[PLICP2D] wrong linear solver type in config file!");
  }
  std::string trust_region_type = node["ceres"]["trust_region_type"].as<std::string>();
  if (trust_region_type == "LM") {
    trust_region_strategy_type_ = ceres::LEVENBERG_MARQUARDT;
  } else if (trust_region_type == "DogLeg") {
    trust_region_strategy_type_ = ceres::DOGLEG;
  } else {
    ERROR("[PLICP2D] wrong trust region type in config file!");
  }
}

bool PLICP2DRegistration::SetInputTarget(const CloudData::CLOUD_PTR &input_target)
{
  input_target_ = input_target;
  CloudClassify(input_target, input_target_group_, input_target_group_empty_, input_target_group_lines_,
                input_target_group_lines_indices_);

  //DEBUG("1 input target [6] size = ", input_target_group_[6]->points.size());

  input_target_group_kdtree_.resize(input_target_group_.size());
  // DEBUG("input_target_group_kdtree_.size()=", input_target_group_kdtree_.size());
  for (std::size_t i = 0; i < input_target_group_kdtree_.size(); i++) {
    // DEBUG("i=", i);
    // DEBUG("empty=", input_target_group_empty_[i]);
    if (input_target_group_empty_[i]) continue;
    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
    // DEBUG("input_target_group_lines_[i].size()=", input_target_group_lines_[i].size());
    cloud->resize(input_target_group_lines_[i].size());
    for (std::size_t j = 0; j < input_target_group_lines_[i].size(); j++) {
      pcl::PointXY &pt = cloud->points[j];
      pt.x = input_target_group_lines_[i][j].centroid[0];
      pt.y = input_target_group_lines_[i][j].centroid[1];
    }
    input_target_group_kdtree_[i].reset(new KdTree());
    input_target_group_kdtree_[i]->setInputCloud(cloud);
    // DEBUG("cloud->size()=", cloud->size());
  }

  //DEBUG("2 input target [6] size = ", input_target_group_[6]->points.size());

  // input_target_group_lines_.resize(input_target_group_.size());
  // input_target_group_lines_indices_.resize(input_target_group_.size());
  ////	input_target_sample_group_.resize(input_target_group_.size());
  // for (std::size_t i = 0; i < input_target_group_.size(); i++) {
  //  input_target_group_lines_[i].clear();
  //  input_target_group_lines_indices_[i].clear();
  //  input_target_group_lines_indices_[i].resize(input_target_group_[i]->size(), -1);
  //  //		input_target_sample_group_[i].reset(new pcl::PointCloud<pcl::PointXY>());
  //  if (input_target_group_empty_[i]) continue;
  //  if (i == 2 || i == 4 || i == 5 || i == 1 || i == 6) {
  //    line_extract_ptr_->Extract(input_target_group_[i], i, input_target_group_lines_[i],
  //                               input_target_group_lines_indices_[i]);
  //    //			for(std::size_t j=0;j<input_target_group_lines_[i].size();j++)
  //    //			{
  //    //				LineFeature2D &line =input_target_group_lines_[i][j];
  //    //				pcl::PointXY pt;
  //    //				pt.x=line.centroid[0];
  //    //				pt.y=line.centroid[1];
  //    //				input_target_sample_group_[i]->push_back(pt);
  //    //			}
  //  }
  //}

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
  //	std::ofstream fp;
  //	fp.open("/home/sun/debug.txt",std::ios::out);
  //	using namespace std;
  cloud_dbg.reset(new CloudData::CLOUD());
  for (std::size_t i = 0; i < input_target_group_lines_indices_.size(); i++) {
    //		 if(input_target_group_empty_[i]) continue;
    //		 fp<<"i = "<<i<<"=================================================="<<endl;
    //		 std::size_t i=6;
    for (std::size_t j = 0; j < input_target_group_lines_indices_[i].size(); j++) {
      //			fp<<"j="<<j<<endl;
      int idx = input_target_group_lines_indices_[i][j];
      //			fp<<"idx="<<idx<<endl;
      if (idx < 0) continue;
      // LineFeature2D &line = input_target_group_lines_[i][idx];
      // Eigen::Vector3f pc = line.centroid;
      // Eigen::Vector3f ep1 = line.endpoint_1;
      // Eigen::Vector3f ep2 = line.endpoint_2;
      // Eigen::Vector3f d = line.direction;
      CloudData::POINTXYZI pt;
      //			fp<<"points.size="<<input_target_group_[i]->points.size()<<endl;
      pt.x = input_target_group_[i]->points[j].x;
      pt.y = input_target_group_[i]->points[j].y;
      pt.z = 0.0;
      // Eigen::Vector3f p(pt.x, pt.y, pt.z);
      pt.intensity = idx;
      //			fp<<"pt= "<<pt.x<<"\t"<<pt.y<<"\t"<<pt.intensity<<endl;
      // pt.intensity = float(d.cross(p - pc).norm());
      cloud_dbg->push_back(pt);
    }
  }
  //	 fp.close();

  return true;
}

bool PLICP2DRegistration::CloudClassify(const CloudData::CLOUD_PTR &input, VecCloud &group, std::vector<bool> &empty,
                                        std::vector<VecLine> &group_lines,
                                        std::vector<std::vector<int>> &group_line_indices)
{
  size_t point_num = input->points.size();
  if (point_num < 10) {
    ERROR("[SGICP][CloudClassify] not sufficient points in input cloud!");
    return false;
  }
  group.resize(SEMANTIC_NUMS);
  // group_kdtree.resize(SEMANTIC_NUMS);
  empty.resize(SEMANTIC_NUMS);
  // group_cov.resize(SEMANTIC_NUMS);
  group_lines.resize(SEMANTIC_NUMS);
  group_line_indices.resize(SEMANTIC_NUMS);

  for (int i = 0; i < SEMANTIC_NUMS; i++) {
    group[i].reset(new pcl::PointCloud<pcl::PointXY>());
  }
  for (size_t i = 0; i < point_num; i++) {
    int class_id = floorf(input->points[i].intensity);
    pcl::PointXY pt;
    pt.x = input->points[i].x;
    pt.y = input->points[i].y;
    group[class_id]->points.push_back(pt);
  }
  for (int i = 0; i < SEMANTIC_NUMS; i++) {
    group_lines[i].clear();
    group_line_indices[i].clear();
    if (group[i]->points.size() < 7) {
      empty[i] = true;
    } else {
      // empty[i] = false;
      // group_cov[i].reset(new VecMat);
      // group_cov[i]->resize(group[i]->size());
      // if (!ComputeCovariance(group[i], group_kdtree[i], group_cov[i])) empty[i] = true;
      group_line_indices[i].resize(group[i]->size(), -1.0);
      line_extract_ptr_->Extract(group[i], i, group_lines[i], group_line_indices[i]);
      //DEBUG(i, "\t", group_lines[i].size());

      if (group_lines[i].size() < 7)
        empty[i] = true;
      else
        empty[i] = false;

      // pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
      // cloud->resize(group_lines[i].size());
      // for (std::size_t j = 0; j < group_lines[i].size(); j++) {
      //  pcl::PointXY &pt = cloud->points[j];
      //  pt.x = group_lines[i][j].centroid[0];
      //  pt.y = group_lines[i][j].centroid[1];
      //}
      // group_kdtree[i].reset(new KdTree());
      // group_kdtree[i]->setInputCloud(cloud);
    }
  }
  return true;
}

// bool PLICP2DRegistration::ComputeCovariance(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud, const KdTreePtr &tree,
//                                            VecMatPtr &covs)
//{
//  if (static_cast<std::size_t>(num_neighbors_cov_) > cloud->size()) {
//    ERROR("[SGICP][ComputeCovariance] input cloud is too small!");
//    return false;
//  }
//
//  Eigen::Vector2f mean;
//  std::vector<int> nn_indices;
//  nn_indices.reserve(num_neighbors_cov_);
//  std::vector<float> nn_dist_sq;
//  nn_dist_sq.reserve(num_neighbors_cov_);
//
//  if (covs->size() != cloud->size()) {
//    covs->resize(cloud->size());
//  }
//
//  VecMat::iterator it_cov = covs->begin();
//  for (auto it_pt = cloud->begin(); it_pt != cloud->end(); ++it_pt, ++it_cov) {
//    const pcl::PointXY &query = *it_pt;
//    Eigen::Matrix2f &cov = *it_cov;
//    cov.setZero();
//    mean.setZero();
//
//    // search for the num_neighbors_cov_ nearest neighbours;
//    tree->nearestKSearch(query, num_neighbors_cov_, nn_indices, nn_dist_sq);
//
//    for (int j = 0; j < num_neighbors_cov_; j++) {
//      const pcl::PointXY &pt = (*cloud)[nn_indices[j]];
//      mean(0) += pt.x;
//      mean(1) += pt.y;
//      // mean(2) += pt.z;
//      // left-bottom triangle of cov matrix;
//      cov(0, 0) += pt.x * pt.x;
//      cov(1, 0) += pt.y * pt.x;
//      cov(1, 1) += pt.y * pt.y;
//      // cov(2, 0) += pt.z * pt.x;
//      // cov(2, 1) += pt.z * pt.y;
//      // cov(2, 2) += pt.z * pt.z;
//    }
//    mean /= static_cast<float>(num_neighbors_cov_);
//    for (int k = 0; k < 2; k++) {
//      for (int l = 0; l <= k; l++) {
//        cov(k, l) /= static_cast<float>(num_neighbors_cov_);
//        cov(k, l) -= mean[k] * mean[l];
//        cov(l, k) = cov(k, l);
//      }
//    }
//    // compute the SVD (symmetric -> EVD);
//    // singular values sorted in decreasing order;
//    Eigen::JacobiSVD<Eigen::Matrix2f> svd(cov, Eigen::ComputeFullU);
//    cov.setZero();
//    Eigen::Matrix2f U = svd.matrixU();
//    // reconstitute the covariance matrx with modified singular values;
//    for (int k = 0; k < 2; k++) {
//      Eigen::Vector2f col = U.col(k);
//      float v = sgicp_epsilon_;  // smallest two singular value replaced by sgicp_epsilon_;
//      if (k == 0) v = 1.;        // biggest singular value replaced by 1;
//      cov += v * col * col.transpose();
//    }
//  }
//  return true;
//}

bool PLICP2DRegistration::ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                                    CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose)
{
  // input_source_ = input_source;
  max_corr_dist_ = max_dist_;

  int nr_iterations_ = 0;
  converged_ = false;
  transformation_.setIdentity();
  // group_informations_.resize(SEMANTIC_NUMS);

  // pre-process input source:
  CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
  pcl::transformPointCloud(*input_source, *transformed_input_source, predict_pose);

  CloudClassify(transformed_input_source, input_source_group_, input_source_group_empty_, input_source_group_lines_,
                input_source_group_lines_indices_);

  // input_source_group_lines_.resize(input_source_group_.size());
  // input_source_group_lines_indices_.resize(input_source_group_.size());
  ////	 input_source_sample_group_.resize(input_source_group_.size());
  // for (std::size_t i = 0; i < input_source_group_.size(); i++) {
  //  input_source_group_lines_[i].clear();
  //  input_source_group_lines_indices_[i].clear();
  //  input_source_group_lines_indices_[i].resize(input_source_group_[i]->size(), -1);
  //  //		input_source_sample_group_[i].reset(new pcl::PointCloud<pcl::PointXY>());
  //  if (input_source_group_empty_[i]) continue;
  //  if (i == 2 || i == 4 || i == 5 || i == 1 || i == 6) {
  //    line_extract_ptr_->Extract(input_source_group_[i], i, input_source_group_lines_[i],
  //                               input_source_group_lines_indices_[i]);
  //    //			for(std::size_t j=0;j<input_source_group_lines_[i].size();j++)
  //    //			{
  //    //				LineFeature2D &line =input_source_group_lines_[i][j];
  //    //				pcl::PointXY pt;
  //    //				pt.x=line.centroid[0];
  //    //				pt.y=line.centroid[1];
  //    //				input_source_sample_group_[i]->push_back(pt);
  //    //			}
  //  }
  //}

  // std::vector<std::vector<int>> source_indices_group(SEMANTIC_NUMS);
  // std::vector<std::vector<int>> target_indices_group(SEMANTIC_NUMS);
  source_indices_group_.resize(SEMANTIC_NUMS);
  target_indices_group_.resize(SEMANTIC_NUMS);

  while (!converged_) {
    correspondences_cur_mse_ = 0.0;
    correspondences_cur_max_ = 0.0;
    int cnt = 0, pc_cnt = 0;
    for (int i = 0; i < SEMANTIC_NUMS; i++) {
      source_indices_group_[i].resize(0);
      target_indices_group_[i].resize(0);
      if (input_target_group_empty_[i] || input_source_group_empty_[i]) {
        continue;
      }
      GetCorrespondence(input_source_group_[i], input_target_group_kdtree_[i], input_source_group_lines_[i],
                        source_indices_group_[i], target_indices_group_[i]);
      // pc_cnt += input_source_group_[i]->size();
      pc_cnt += input_source_group_lines_[i].size();
      cnt += source_indices_group_[i].size();
    }
    correspondences_cur_mse_ /= float(cnt);
    UpdateCorrMaxDist(correspondences_cur_mse_, correspondences_cur_max_, pc_cnt, cnt);

    previous_transformation_ = transformation_;
    // rigid_transformation_estimation_(input_source_group_, source_indices_group, input_target_group_,
    //                                 target_indices_group, transformation_);
    // Converter::NormTransformMatrix(transformation_);

    double x = transformation_.translation()[0];
    double y = transformation_.translation()[1];
    double yaw = acos(0.5 * transformation_.linear().trace());
    // DEBUG("transformation_=\n", transformation_.matrix());
    // DEBUG("x=", x);
    // DEBUG("y=", y);
    // DEBUG("yaw=", yaw);
    ceres::Problem problem;
    if (BuildCeresProblem(&problem, &x, &y, &yaw, source_indices_group_, target_indices_group_)) {
      if (!SolveCeresProblem(&problem)) {
        ERROR("[PLICP2D][ScanMatch] Invalid ceres solution!");
        return false;
      }
    } else {
      ERROR("[PLICP2D][ScanMatch] Build ceres problem failure!");
      return false;
    }
    transformation_.matrix() << cos(yaw), -sin(yaw), x, sin(yaw), cos(yaw), y, 0.0, 0.0, 1.0;
    // DEBUG("-----------------");
    // DEBUG("transformation_=\n", transformation_.matrix());
    // DEBUG("x=", x);
    // DEBUG("y=", y);
    // DEBUG("yaw=", yaw);

    nr_iterations_++;

    if (hasConverged(transformation_) || nr_iterations_ >= max_iterations_) {
      converged_ = true;
      previous_transformation_ = transformation_;
    }
  }
  // result_pose = previous_transformation_ * predict_pose;
  Eigen::Isometry3f tmp_transformation = Eigen::Isometry3f::Identity();
  tmp_transformation.linear().block<2, 2>(0, 0) = previous_transformation_.linear();
  tmp_transformation.translation().block<2, 1>(0, 0) = previous_transformation_.translation();
  result_pose = tmp_transformation * predict_pose;
  Converter::NormTransformMatrix(result_pose);

  // align_cloud_.reset(new CloudData::CLOUD());
  // pcl::transformPointCloud(*input_source, *align_cloud_, result_pose);
  // result_cloud_ptr = align_cloud_;

  pcl::transformPointCloud(*input_source, *result_cloud_ptr, result_pose);

  // base_transformation_ = predict_pose;
  // cloud_transformation_ = tmp_transformation;
  // input_source_.reset(new CloudData::CLOUD());
  // informations_.resize(0);
  // CloudData::CLOUD_PTR tmp_cloud;
  // for (int i = 0; i < SEMANTIC_NUMS; i++) {
  //  // transform input_source_group_;
  //  tmp_cloud.reset(new CloudData::CLOUD());
  //  tmp_cloud->resize(input_source_group_lines_[i].size());
  //  for (std::size_t k = 0; k < input_source_group_lines_[i].size(); k++) {
  //    tmp_cloud->at(k).x = input_target_group_lines_[i][k].centroid[0];
  //    tmp_cloud->at(k).y = input_target_group_lines_[i][k].centroid[1];
  //    // tmp_cloud->at(k).x = input_source_group_[i]->at(k).x;
  //    // tmp_cloud->at(k).y = input_source_group_[i]->at(k).y;
  //    tmp_cloud->at(k).z = 0.0;
  //  }
  //  // pcl::transformPointCloud(*input_source_group_[i], *tmp_cloud, predict_pose.inverse());
  //  pcl::transformPointCloud(*tmp_cloud, *tmp_cloud, predict_pose.inverse());
  //  for (std::size_t j = 0; j < source_indices_group_[i].size(); j++) {
  //    std::size_t idx = source_indices_group_[i][j];
  //    input_source_->push_back(tmp_cloud->at(idx));
  //    // informations_.push_back(group_informations_[i][idx]);
  //    informations_.push_back(Eigen::Matrix3f::Identity());
  //  }
  //}
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

// float PLICP2DRegistration::GetFitnessScore()
//{
//  float max_range = std::numeric_limits<float>::max();
//  float fitness_score = 0.0;
//
//  std::vector<int> nn_indices(1);
//  std::vector<float> nn_dists(1);
//  // For each point in the source dataset
//  int nr = 0;
//  size_t pc_size = align_cloud_->size();
//  for (size_t i = 0; i < pc_size; ++i) {
//    // Find its nearest neighbor in the target
//    int class_id = floorf(align_cloud_->at(i).intensity);
//    if (input_target_group_empty_[class_id]) {
//      continue;
//    }
//    input_target_group_kdtree_[class_id]->nearestKSearch(align_cloud_->at(i), 1, nn_indices, nn_dists);
//    // Deal with occlusions (incomplete targets)
//    if (nn_dists[0] <= max_range) {
//      // Add to the fitness score
//      fitness_score += nn_dists[0];
//      nr++;
//    }
//  }
//  if (nr > 0)
//    return (fitness_score / nr);
//  else
//    return (std::numeric_limits<float>::max());
//}
int PLICP2DRegistration::GetCorrespondence(const pcl::PointCloud<pcl::PointXY>::Ptr &source_cloud,
                                           const KdTreePtr &target_kdtree, const VecLine &source_lines,
                                           std::vector<int> &source_indices, std::vector<int> &target_indices)
{
  const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;
  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);

  std::size_t cnt = 0;
  source_indices.resize(source_cloud->size());
  target_indices.resize(source_cloud->size());
  // infos.resize(source_cloud->size());
  // Eigen::Matrix2f trans_R = transformation_.linear();

  // for (std::size_t i = 0; i < source_cloud->size(); i++) {
  for (std::size_t i = 0; i < source_lines.size(); i++) {
    // pcl::PointXY query = source_cloud->at(i);
    pcl::PointXY query = TransformPointXY(transformation_, source_lines[i].centroid);
    // query.x = source_lines[i].centroid[0];
    // query.y = source_lines[i].centroid[1];
    // Eigen::Vector2f query_tmp(query.x, query.y);
    // query_tmp = transformation_ * query_tmp;
    // query.x = query_tmp[0];
    // query.y = query_tmp[1];
    // query.getVector3fMap() = transformation_ * query.getVector3fMap();

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
      // Eigen::Matrix2f &M = infos[i];
      // M.setIdentity();
      // Eigen::Matrix3f &C1 = (*source_covs)[i];
      // Eigen::Matrix3f &C2 = (*target_covs)[nn_indices[0]];
      // M = trans_R * C1;
      // Eigen::Matrix3f tmp = M * trans_R.transpose();
      // tmp += C2;
      // M = tmp.inverse();
      source_indices[cnt] = static_cast<int>(i);
      target_indices[cnt] = nn_indices[0];
      cnt++;
    }
  }
  source_indices.resize(cnt);
  target_indices.resize(cnt);

  return cnt;
}

pcl::PointXY PLICP2DRegistration::TransformPointXY(const Eigen::Isometry2f &trans, const pcl::PointXY &point)
{
  Eigen::Vector2f tmp(point.x, point.y);
  tmp = trans * tmp;
  pcl::PointXY pt;
  pt.x = tmp[0];
  pt.y = tmp[1];
  return pt;
}
pcl::PointXY PLICP2DRegistration::TransformPointXY(const Eigen::Isometry2f &trans, const Eigen::Vector2f &point)
{
  Eigen::Vector2f tmp = trans * point;
  pcl::PointXY pt;
  pt.x = tmp[0];
  pt.y = tmp[1];
  return pt;
}
float PLICP2DRegistration::Point2LineDistance(const CloudData::POINTXYZI &pt, const LineFeature &line)
{
  Eigen::Vector3f p(pt.x, pt.y, pt.z);
  // Eigen::Vector3f tmp1 = p - line.endpoint_1;
  // Eigen::Vector3f tmp2 = p - line.endpoint_2;
  // Eigen::Vector3f den = line.endpoint_1 - line.endpoint_2;
  // return float(tmp1.cross(tmp2).norm() / den.norm());
  return float(line.direction.cross(p - line.centroid).norm());
}

bool PLICP2DRegistration::hasConverged(const Eigen::Isometry2f &transformation)
{
  // 1. The epsilon (difference) between the previous transformation and the current estimated transformation
  // a. translation magnitude -- squaredNorm:
  float translation_sqr = transformation.translation().squaredNorm();
  // b. rotation magnitude -- angle:
  // float cos_angle = (transformation.linear().trace() - 1.0f) / 2.0f;
  float cos_angle = transformation.linear().trace() / 2.0f;
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
bool PLICP2DRegistration::UpdateCorrMaxDist(const float &mean_dist, const float &max_dist, const size_t &pc_size,
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

// void PLICP2DRegistration::ApplyState(Eigen::Isometry3f &t, const Vector6f &x) const
//{
//  // Z Y X euler angles convention
//  Eigen::Matrix3f R;
//  R = Eigen::AngleAxisf(static_cast<float>(x[5]), Eigen::Vector3f::UnitZ()) *
//      Eigen::AngleAxisf(static_cast<float>(x[4]), Eigen::Vector3f::UnitY()) *
//      Eigen::AngleAxisf(static_cast<float>(x[3]), Eigen::Vector3f::UnitX());
//  t.linear() = R * t.linear();
//  t.translation()(0) += x[0];
//  t.translation()(1) += x[1];
//  t.translation()(2) += x[2];
//}

// void PLICP2DRegistration::ComputeRDerivative(const Vector6f &x, const Eigen::Vector3f &RX,
//                                           Eigen::Matrix3f &dP_dtheta) const
//{
//  Eigen::Matrix3f dR_dPhi;
//  Eigen::Matrix3f dR_dTheta;
//  Eigen::Matrix3f dR_dPsi;
//
//  float phi = x[3], theta = x[4], psi = x[5];
//
//  float cphi = std::cos(phi), sphi = sin(phi);
//  float ctheta = std::cos(theta), stheta = sin(theta);
//  float cpsi = std::cos(psi), spsi = sin(psi);
//
//  dR_dPhi(0, 0) = 0.;
//  dR_dPhi(1, 0) = 0.;
//  dR_dPhi(2, 0) = 0.;
//
//  dR_dPhi(0, 1) = sphi * spsi + cphi * cpsi * stheta;
//  dR_dPhi(1, 1) = -cpsi * sphi + cphi * spsi * stheta;
//  dR_dPhi(2, 1) = cphi * ctheta;
//
//  dR_dPhi(0, 2) = cphi * spsi - cpsi * sphi * stheta;
//  dR_dPhi(1, 2) = -cphi * cpsi - sphi * spsi * stheta;
//  dR_dPhi(2, 2) = -ctheta * sphi;
//
//  dR_dTheta(0, 0) = -cpsi * stheta;
//  dR_dTheta(1, 0) = -spsi * stheta;
//  dR_dTheta(2, 0) = -ctheta;
//
//  dR_dTheta(0, 1) = cpsi * ctheta * sphi;
//  dR_dTheta(1, 1) = ctheta * sphi * spsi;
//  dR_dTheta(2, 1) = -sphi * stheta;
//
//  dR_dTheta(0, 2) = cphi * cpsi * ctheta;
//  dR_dTheta(1, 2) = cphi * ctheta * spsi;
//  dR_dTheta(2, 2) = -cphi * stheta;
//
//  dR_dPsi(0, 0) = -ctheta * spsi;
//  dR_dPsi(1, 0) = cpsi * ctheta;
//  dR_dPsi(2, 0) = 0.;
//
//  dR_dPsi(0, 1) = -cphi * cpsi - sphi * spsi * stheta;
//  dR_dPsi(1, 1) = -cphi * spsi + cpsi * sphi * stheta;
//  dR_dPsi(2, 1) = 0.;
//
//  dR_dPsi(0, 2) = cpsi * sphi - cphi * spsi * stheta;
//  dR_dPsi(1, 2) = sphi * spsi + cphi * cpsi * stheta;
//  dR_dPsi(2, 2) = 0.;
//
//  // g[3] = MatricesInnerProd(dR_dPhi, R);
//  // g[4] = MatricesInnerProd(dR_dTheta, R);
//  // g[5] = MatricesInnerProd(dR_dPsi, R);
//  dP_dtheta.col(0) = dR_dPhi * RX;
//  dP_dtheta.col(1) = dR_dTheta * RX;
//  dP_dtheta.col(2) = dR_dPsi * RX;
//}

bool PLICP2DRegistration::BuildCeresProblem(ceres::Problem *problem, double *x, double *y, double *yaw,
                                            const std::vector<std::vector<int>> &source_indices_group,
                                            const std::vector<std::vector<int>> &target_indices_group)
{
  if (problem == NULL) {
    ERROR("[PLICP2D][BuildCeresProblem] invalid optimization problem!");
    return false;
  }

  ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();

  for (std::size_t i = 0; i < source_indices_group.size(); i++) {
    for (std::size_t j = 0; j < source_indices_group[i].size(); j++) {
      int idx_src = source_indices_group[i][j];
      int idx_tgt = target_indices_group[i][j];
      // int idx_line_tgt = input_target_group_lines_indices_[i][idx_tgt];

      // if (idx_line_tgt < 0) continue;

      Eigen::Vector2d pt(input_source_group_lines_[i][idx_src].centroid[0],
                         input_source_group_lines_[i][idx_src].centroid[1]);
      Eigen::Vector2d ep1(input_target_group_lines_[i][idx_tgt].endpoint_1[0],
                          input_target_group_lines_[i][idx_tgt].endpoint_1[1]);
      Eigen::Vector2d ep2(input_target_group_lines_[i][idx_tgt].endpoint_2[0],
                          input_target_group_lines_[i][idx_tgt].endpoint_2[1]);
      // Eigen::Vector2d pt(input_source_group_[i]->points[idx_src].x, input_source_group_[i]->points[idx_src].y);
      // Eigen::Vector2d ep1(input_target_group_lines_[i][idx_line_tgt].endpoint_1[0],
      //                    input_target_group_lines_[i][idx_line_tgt].endpoint_1[1]);
      // Eigen::Vector2d ep2(input_target_group_lines_[i][idx_line_tgt].endpoint_2[0],
      //                    input_target_group_lines_[i][idx_line_tgt].endpoint_2[1]);

      ceres::CostFunction *cost_function = CeresLinePointErrorTerm::Create(ep1, ep2, pt);
      problem->AddResidualBlock(cost_function, loss_function, x, y, yaw);

      problem->SetParameterization(yaw, angle_local_parameterization);
    }
  }

  return true;
}

bool PLICP2DRegistration::SolveCeresProblem(ceres::Problem *problem)
{
  if (problem == NULL) {
    ERROR("[PLICP2D][SolveCeresProblem] invalid optimization problem!");
    return false;
  }

  ceres::Solver::Options options;
  options.max_num_iterations = max_num_iterations_;
  options.linear_solver_type = linear_solver_type_;
  options.trust_region_strategy_type = trust_region_strategy_type_;
  options.minimizer_progress_to_stdout = minimizer_progress_to_stdout_;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  if (options.minimizer_progress_to_stdout) std::cout << summary.FullReport() << std::endl;

  return summary.IsSolutionUsable();
}

PLICP2DRegistration::Matrix6f PLICP2DRegistration::GetCovariance()
{
  float cos_theta = transformation_.linear()(0, 0);
  float sin_theta = transformation_.linear()(1, 0);
  Eigen::Matrix3f d2E_dxi2 = Eigen::Matrix3f::Zero();
  int cnt = 0;
  cost_min_ = 0.0;
  Eigen::Matrix2f scatter = Eigen::Matrix2f::Zero();
  for (std::size_t i = 0; i < source_indices_group_.size(); i++) {
    cnt += source_indices_group_[i].size();
    for (std::size_t j = 0; j < source_indices_group_[i].size(); j++) {
      std::size_t idx_src = source_indices_group_[i][j];
      std::size_t idx_tgt = target_indices_group_[i][j];

      const LineFeature2D &line_src = input_source_group_lines_[i][idx_src];
      Eigen::Vector2f pt = line_src.centroid;
      Eigen::Vector2f pt1 = transformation_ * pt;
      const LineFeature2D &line_tgt = input_target_group_lines_[i][idx_tgt];
      Eigen::Vector2f ep1 = line_tgt.endpoint_1;
      Eigen::Vector2f ep2 = line_tgt.endpoint_2;

      Eigen::Vector2f p_ep_1 = pt1 - ep1;
      Eigen::Vector2f p_ep_2 = pt1 - ep2;
      Eigen::Vector2f ep1_ep2 = ep1 - ep2;
      float le = ep1_ep2.norm();
      float dE = VectorCross2D(p_ep_1, p_ep_2) / le;
      cost_min_ += dE * dE;

      Eigen::Matrix3f dL_dxi_2;
      // float xe1 = ep1[0], ye1 = ep1[1], xe2 = ep2[0], ye2 = ep2[1];
      float x = pt[0], y = pt[1];
      float ye = ep1[1] - ep2[1];
      float xe = ep2[0] - ep1[0];
      float r1 = -x * sin_theta - y * cos_theta;
      float r2 = x * cos_theta - y * sin_theta;

      dL_dxi_2(0, 0) = ye * ye;  //(ye1 - ye2) * (ye1 - ye2);
      dL_dxi_2(1, 0) = xe * ye;  //(xe2 - xe1) * (ye1 - ye2);
      dL_dxi_2(2, 0) = dL_dxi_2(0, 0) * r1 + dL_dxi_2(1, 0) * r2;

      dL_dxi_2(0, 1) = dL_dxi_2(1, 0);
      dL_dxi_2(1, 1) = xe * xe;  //(xe2 - xe1) * (xe2 - xe1);
      dL_dxi_2(2, 1) = dL_dxi_2(0, 1) * r1 + dL_dxi_2(1, 1) * r2;

      dL_dxi_2(0, 2) = dL_dxi_2(2, 0);
      dL_dxi_2(1, 2) = dL_dxi_2(2, 1);
      dL_dxi_2(2, 2) = (ye * r1 + xe * r2) * (ye * r1 + xe * r2);

      // Eigen::Matrix3f d2L_dxi2 = Eigen::Matrix3f::Zero();
      // d2L_dxi2(2, 2) = ye * (-x * cos_theta + y * sin_theta) + xe * (-x * sin_theta - y * cos_theta);

      // d2E_dxi2 += (1.0 / le) * ((1.0 / le) * dL_dxi_2 + dE * d2E_dxi2);
      d2E_dxi2 += dL_dxi_2 / (le * le);

      scatter += ep1_ep2 * ep1_ep2.transpose();
    }
  }
  d2E_dxi2 *= 2.0 / float(cnt);
  cost_min_ /= float(cnt - 3);

  Eigen::Matrix<float, 3, 6> selection = Eigen::Matrix<float, 3, 6>::Zero();
  selection(0, 0) = 1.0;
  selection(1, 1) = 1.0;
  selection(2, 5) = 1.0;

  hessian_ = selection.transpose() * d2E_dxi2 * selection;

  Eigen::SelfAdjointEigenSolver<Matrix6f> es;
  es.compute(hessian_);
  Vector6f hessian_eigenvalues_ = es.eigenvalues().cwiseAbs();
  Matrix6f hessian_eigenvectors_ = es.eigenvectors();
  Matrix6f eigenvalues_inv = Matrix6f::Zero();
  for (int i = 0; i < 6; i++) {
    if (hessian_eigenvalues_(i) > 1e-7) {
      eigenvalues_inv(i, i) = 1.0 / hessian_eigenvalues_(i);
    }
  }
  has_hessian_computed_ = true;

  Matrix6f hessian_inv = hessian_eigenvectors_ * eigenvalues_inv * hessian_eigenvectors_.transpose();
  Matrix6f cov = hessian_inv * cost_min_;

  //DEBUG("[PLICP2D]");
  //std::cout << "[registration] cost_min_ = " << cost_min_ << std::endl;
  //std::cout << "[registration] hessian_ = " << std::endl << hessian_ << std::endl;
  //std::cout << "[registration] hessian_inv = " << std::endl << hessian_inv << std::endl;
  //std::cout << "[registration] cov = " << std::endl << cov << std::endl << std::endl;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> ess;
  ess.compute(scatter);
  Eigen::Vector2f scatter_eigenvalues_ = ess.eigenvalues().cwiseAbs();
  Eigen::Matrix2f scatter_eigenvectors_ = ess.eigenvectors();

  DEBUG("scatter");
  for (int i = 0; i < 2; i++) {
    std::cout << scatter_eigenvalues_[i] << "\t" << scatter_eigenvectors_.col(i).transpose() << std::endl;
  }
  std::cout << std::endl;

  return cov;
}
}  // namespace vision_localization
