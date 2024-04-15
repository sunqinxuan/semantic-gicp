/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2022-03-02 15:05
#
# Filename:		sgicp_registration.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_REGISTRATION_SGICP_REGISTRATION_HPP_
#define VSION_LOCALIZATION_MODELS_REGISTRATION_SGICP_REGISTRATION_HPP_

#include "models/registration/registration_interface.hpp"

#include "sensor_data/cloud_data.hpp"
#include <fstream>
#include <functional>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/bfgs.h>


namespace vision_localization
{
class SGICPRegistration : public RegistrationInterface
{
public:
  using Vector6f = Eigen::Matrix<float, 6, 1>;
  using VecCloud = std::vector<CloudData::CLOUD_PTR>;
  using KdTree = pcl::KdTreeFLANN<CloudData::POINTXYZI>;
  using KdTreePtr = pcl::KdTreeFLANN<CloudData::POINTXYZI>::Ptr;
  using VecKdTreePtr = std::vector<KdTreePtr>;
  using VecMat = std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>;
  using VecMatPtr = std::shared_ptr<VecMat>;

  SGICPRegistration(const YAML::Node &node);

  bool SetInputTarget(const CloudData::CLOUD_PTR &input_target) override;
  bool ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                 CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose) override;
  float GetFitnessScore() override;

private:
  const int SEMANTIC_NUMS = 7;  //语义种类数量(加上地面类)

  int max_iterations_ = 200;
  int max_inner_iterations_ = 20;
  bool converged_ = false;

  // temporary pointers for BFGS optimization;
  const VecCloud *tmp_src_;
  const VecCloud *tmp_tgt_;
  const std::vector<std::vector<int>> *tmp_idx_src_;
  const std::vector<std::vector<int>> *tmp_idx_tgt_;

  // threshold for convergence of ICP iteration;
  float rotation_epsilon_ = 2e-4;
  float transformation_epsilon_ = 5e-4;

  // epsilon in the adjusted covariance matrix;
  float sgicp_epsilon_ = 0.001;
  // number of neighbors for PCA;
  int num_neighbors_cov_ = 20;

  // threshold for point correspondences;
  float max_corr_dist_;
  float max_dist_;
  float min_dist_;
  float correspondences_cur_mse_ = 10000.0;
  float correspondences_prev_mse_ = 0.0;
  float correspondences_cur_max_;

  float translation_threshold_;   //(3e-4 * 3e-4)  // 0.0003 meters
  float rotation_threshold_;      //(0.99999)         // 0.256 degrees
  float mse_threshold_relative_;  //(0.00001)     // 0.001% of the previous MSE (relative error)
  float mse_threshold_absolute_;  //(1e-12)       // MSE (absolute error)

private:
  int GetCorrespondence(const CloudData::CLOUD_PTR &source_cloud, const KdTreePtr &target_kdtree,
                        const VecMatPtr &source_covs, const VecMatPtr &target_covs, std::vector<int> &source_indices,
                        std::vector<int> &target_indices, VecMat &infos);

  bool hasConverged(const Eigen::Isometry3f &transformation);
  bool UpdateCorrMaxDist(const float &mean_dist, const float &max_dist, const size_t &pc_size,
                         const size_t &inline_size);

  bool CloudClassify(const CloudData::CLOUD_PTR &input, VecCloud &group, VecKdTreePtr &group_kdtree,
                     std::vector<bool> &empty, std::vector<VecMatPtr> &group_cov);
  bool ComputeCovariance(const CloudData::CLOUD_PTR &cloud, const KdTreePtr &tree, VecMatPtr &covs);

  void ApplyState(Eigen::Isometry3f &t, const Vector6f &x) const;
  void ComputeRDerivative(const Vector6f &x, const Eigen::Matrix3f &R, Vector6f &g) const;
  float MatricesInnerProd(const Eigen::MatrixXf &mat1, const Eigen::MatrixXf &mat2) const;

  CloudData::CLOUD_PTR align_cloud_;

  Eigen::Isometry3f transformation_;
  Eigen::Isometry3f previous_transformation_;

  VecCloud input_target_group_;
  VecKdTreePtr input_target_group_kdtree_;
  std::vector<VecMatPtr> input_target_group_covariance_;
  std::vector<bool> input_target_group_empty;

  VecCloud input_source_group_;
  VecKdTreePtr input_source_group_kdtree_;
  std::vector<VecMatPtr> input_source_group_covariance_;
  std::vector<bool> input_source_group_empty;

  // information matrices for mahalanobis distances;
  std::vector<VecMat> group_informations_;

  // optimization functor structure
  struct OptimizationFunctor : public BFGSDummyFunctor<float, 6> {
    OptimizationFunctor(const SGICPRegistration *sgicp) : BFGSDummyFunctor<float, 6>(), sgicp_(sgicp)
    {
    }
    virtual double operator()(const Vector6f &x) override;
    virtual void df(const Vector6f &x, Vector6f &df) override;
    virtual void fdf(const Vector6f &x, float &f, Vector6f &df) override;

    const SGICPRegistration *sgicp_;
  };

  std::function<void(const VecCloud &cloud_src, const std::vector<std::vector<int>> &indices_src,
                     const VecCloud &cloud_tgt, const std::vector<std::vector<int>> &indices_tgt,
                     Eigen::Isometry3f &transformation)>
      rigid_transformation_estimation_;

  void estimateRigidTransformationBFGS(const VecCloud &cloud_src, const std::vector<std::vector<int>> &indices_src,
                                       const VecCloud &cloud_tgt, const std::vector<std::vector<int>> &indices_tgt,
                                       Eigen::Isometry3f &transformation);
};
}  // namespace vision_localization

#endif
