/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2022-07-05 09:12
#
# Filename:		plicp2d_registration.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_REGISTRATION_PLICP2D_REGISTRATION_HPP_
#define VSION_LOCALIZATION_MODELS_REGISTRATION_PLICP2D_REGISTRATION_HPP_

#include "models/line_feature_extraction/line_feature_extraction_rg.hpp"
#include "models/registration/registration_interface.hpp"

#include "sensor_data/cloud_data.hpp"
#include "sensor_data/line_feature.hpp"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <fstream>
#include <functional>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/bfgs.h>

namespace vision_localization
{
class PLICP2DRegistration : public RegistrationInterface
{
public:
  using Matrix6f = Eigen::Matrix<float, 6, 6>;
  using Vector6f = Eigen::Matrix<float, 6, 1>;
  using VecCloud = std::vector<pcl::PointCloud<pcl::PointXY>::Ptr>;
  using KdTree = pcl::KdTreeFLANN<pcl::PointXY>;
  using KdTreePtr = KdTree::Ptr;
  using VecKdTreePtr = std::vector<KdTreePtr>;
  using VecMat = std::vector<Eigen::Matrix2f, Eigen::aligned_allocator<Eigen::Matrix2f>>;
  using VecMatPtr = std::shared_ptr<VecMat>;
  using VecLine = std::vector<LineFeature2D>;

  PLICP2DRegistration(const YAML::Node &node);

  bool SetInputTarget(const CloudData::CLOUD_PTR &input_target) override;
  bool ScanMatch(const CloudData::CLOUD_PTR &input_source, const Eigen::Isometry3f &predict_pose,
                 CloudData::CLOUD_PTR &result_cloud_ptr, Eigen::Isometry3f &result_pose) override;
  float GetFitnessScore() override
  {
    return 0.0;
  }

  Matrix6f GetCovariance() override;

private:
  const int SEMANTIC_NUMS = 7;  //语义种类数量(加上地面类)

  int max_iterations_ = 200;
  int max_inner_iterations_ = 20;
  bool converged_ = false;

  // temporary pointers for BFGS optimization;
  // const VecCloud *tmp_src_;
  // const VecCloud *tmp_tgt_;
  // const std::vector<std::vector<int>> *tmp_idx_src_;
  // const std::vector<std::vector<int>> *tmp_idx_tgt_;
  // const std::vector<VecLine> *tmp_lines_tgt_;
  // const std::vector<std::vector<int>> *tmp_idx_lines_tgt_;
  // const std::vector<VecLine> *tmp_lines_src_;
  // const std::vector<std::vector<int>> *tmp_idx_lines_src_;

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
  int GetCorrespondence(const pcl::PointCloud<pcl::PointXY>::Ptr &source_cloud, const KdTreePtr &target_kdtree,
                        const VecLine &source_lines, std::vector<int> &source_indices,
                        std::vector<int> &target_indices);

  bool hasConverged(const Eigen::Isometry2f &transformation);
  bool UpdateCorrMaxDist(const float &mean_dist, const float &max_dist, const size_t &pc_size,
                         const size_t &inline_size);

  bool CloudClassify(const CloudData::CLOUD_PTR &input, VecCloud &group, std::vector<bool> &empty,
                     std::vector<VecLine> &group_lines, std::vector<std::vector<int>> &group_line_indices);
  // bool ComputeCovariance(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud, const KdTreePtr &tree, VecMatPtr &covs);

  // void ApplyState(Eigen::Isometry3f &t, const Vector6f &x) const;
  // void ComputeRDerivative(const Vector6f &x, const Eigen::Vector3f &RX, Eigen::Matrix3f &dP_dtheta) const;
  // float MatricesInnerProd(const Eigen::MatrixXf &mat1, const Eigen::MatrixXf &mat2) const;

  float Point2LineDistance(const CloudData::POINTXYZI &pt, const LineFeature &line);
  pcl::PointXY TransformPointXY(const Eigen::Isometry2f &trans, const pcl::PointXY &point);
  pcl::PointXY TransformPointXY(const Eigen::Isometry2f &trans, const Eigen::Vector2f &point);

  // CloudData::CLOUD_PTR align_cloud_;

  Eigen::Isometry2f transformation_;
  Eigen::Isometry2f previous_transformation_;

  VecCloud input_target_group_;
  VecKdTreePtr input_target_group_kdtree_;
  // std::vector<VecMatPtr> input_target_group_covariance_;
  std::vector<bool> input_target_group_empty_;
  std::vector<VecLine> input_target_group_lines_;
  std::vector<std::vector<int>> input_target_group_lines_indices_;
  //	VecCloud input_source_sample_group_;

  VecCloud input_source_group_;
  // VecKdTreePtr input_source_group_kdtree_;
  // std::vector<VecMatPtr> input_source_group_covariance_;
  std::vector<bool> input_source_group_empty_;
  std::vector<VecLine> input_source_group_lines_;
  std::vector<std::vector<int>> input_source_group_lines_indices_;
  //	VecCloud input_target_sample_group_;

  // information matrices for mahalanobis distances;
  // std::vector<VecMat> group_informations_;

  std::shared_ptr<LineFeatureExtractionInterface> line_extract_ptr_;

  std::vector<std::vector<int>> source_indices_group_;
  std::vector<std::vector<int>> target_indices_group_;

  bool BuildCeresProblem(ceres::Problem *problem, double *x, double *y, double *yaw,
                         const std::vector<std::vector<int>> &source_indices_group,
                         const std::vector<std::vector<int>> &target_indices_group);
  bool SolveCeresProblem(ceres::Problem *problem);

  ceres::LinearSolverType linear_solver_type_ = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::TrustRegionStrategyType trust_region_strategy_type_ = ceres::LEVENBERG_MARQUARDT;
  unsigned int max_num_iterations_ = 10;
  bool minimizer_progress_to_stdout_ = false;

  // optimization functor structure
  // struct OptimizationFunctor : public BFGSDummyFunctor<float, 6> {
  //  OptimizationFunctor(const PLICP2DRegistration *sgicp) : BFGSDummyFunctor<float, 6>(), plicp_(sgicp)
  //  {
  //  }
  //  virtual double operator()(const Vector6f &x) override;
  //  virtual void df(const Vector6f &x, Vector6f &df) override;
  //  virtual void fdf(const Vector6f &x, float &f, Vector6f &df) override;
  //
  //  const PLICP2DRegistration *plicp_;
  //};
  //
  // std::function<void(const VecCloud &cloud_src, const std::vector<std::vector<int>> &indices_src,
  //                   const VecCloud &cloud_tgt, const std::vector<std::vector<int>> &indices_tgt,
  //                   Eigen::Isometry3f &transformation)>
  //    rigid_transformation_estimation_;
  //
  // void estimateRigidTransformationBFGS(const VecCloud &cloud_src, const std::vector<std::vector<int>> &indices_src,
  //                                     const VecCloud &cloud_tgt, const std::vector<std::vector<int>> &indices_tgt,
  //                                     Eigen::Isometry3f &transformation);

  template <typename T> static Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw)
  {
    const T cos_yaw = ceres::cos(yaw);
    const T sin_yaw = ceres::sin(yaw);
    Eigen::Matrix<T, 2, 2> rotation;
    rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
    return rotation;
  }

  // return angle in [-pi,pi);
  template <typename T> inline static T NormalizeAngle(const T &angle)
  {
    T two_pi(2.0 * M_PI);
    return angle - two_pi * ceres::floor((angle + T(M_PI)) / two_pi);
  }

  template <typename T> inline static T VectorCross2D(const Eigen::Matrix<T, 2, 1> &a, const Eigen::Matrix<T, 2, 1> &b)
  {
    return a(0) * b(1) - a(1) * b(0);
  }

  template <typename T> inline static T VectorNorm2D(const Eigen::Matrix<T, 2, 1> &a)
  {
    return ceres::sqrt(a(0) * a(0) + a(1) * a(1));
  }


  class AngleLocalParameterization
  {
  public:
    template <typename T>
    bool operator()(const T *theta_radians, const T *delta_theta_radians, T *theta_radians_plus_delta) const
    {
      *theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);

      return true;
    }

    static ceres::LocalParameterization *Create()
    {
      return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
    }
  };

  class CeresLinePointErrorTerm
  {
  public:
    CeresLinePointErrorTerm(const Eigen::Vector2d &ep1, const Eigen::Vector2d &ep2, const Eigen::Vector2d &p)
        : endpoint_1(ep1), endpoint_2(ep2), point(p)
    {
    }

    template <typename T> bool operator()(const T *const x, const T *const y, const T *const yaw, T *residual_ptr) const
    {
      const Eigen::Matrix<T, 2, 1> translation(*x, *y);
      const Eigen::Matrix<T, 2, 2> rotation = RotationMatrix2D(*yaw);

      Eigen::Matrix<T, 2, 1> trans_point = rotation * point.template cast<T>() + translation;
      Eigen::Matrix<T, 2, 1> p_ep_1 = trans_point - endpoint_1.template cast<T>();
      Eigen::Matrix<T, 2, 1> p_ep_2 = trans_point - endpoint_2.template cast<T>();
      Eigen::Matrix<T, 2, 1> ep1_ep2 = endpoint_1.template cast<T>() - endpoint_2.template cast<T>();

      *residual_ptr = VectorCross2D(p_ep_1, p_ep_2) / VectorNorm2D(ep1_ep2);
      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d &ep1, const Eigen::Vector2d &ep2, const Eigen::Vector2d &p)
    {
      return new ceres::AutoDiffCostFunction<CeresLinePointErrorTerm, 1, 1, 1, 1>(
          new CeresLinePointErrorTerm(ep1, ep2, p));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    const Eigen::Vector2d endpoint_1, endpoint_2;
    const Eigen::Vector2d point;
  };
};
}  // namespace vision_localization

#endif
