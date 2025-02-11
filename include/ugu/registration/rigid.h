/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <functional>

#include "ugu/accel/kdtree.h"
#include "ugu/correspondence/correspondence_finder.h"
#include "ugu/mesh.h"

namespace ugu {

// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

Eigen::Affine3f FindRigidTransformFrom3dCorrespondencesWithNormals(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst,
    const std::vector<Eigen::Vector3f>& dst_normals);

Eigen::Affine3d FindRigidTransformFrom3dCorrespondencesWithNormals(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst,
    const std::vector<Eigen::Vector3d>& dst_normals);

// "Least-squares estimation of transformation parameters between two point
// patterns ", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
// implementation reference:
// https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py#L63
bool FindSimilarityTransformFromPointCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst,
    Eigen::MatrixXd& R_similarity, Eigen::MatrixXd& t_similarity,
    Eigen::MatrixXd& scale, Eigen::MatrixXd& T_similarity,
    Eigen::MatrixXd& R_rigid, Eigen::MatrixXd& t_rigid,
    Eigen::MatrixXd& T_rigid);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);

void DecomposeRts(const Eigen::Affine3f& T, Eigen::Matrix3f& R,
                  Eigen::Vector3f& t, Eigen::Vector3f& s);
void DecomposeRts(const Eigen::Affine3d& T, Eigen::Matrix3d& R,
                  Eigen::Vector3d& t, Eigen::Vector3d& s);

enum class IcpCorrespType { kPointToPoint = 0, kPointToPlane = 1 };
enum class IcpLossType { kPointToPoint = 0, kPointToPlane = 1 };

struct IcpTerminateCriteria {
  int iter_max = 20;
  double loss_min = 0.001;
  double loss_eps = 0.0001;
};

struct IcpCorrespCriteria {
  bool test_nearest = false;
  float normal_th = -1.f;
  float dist_th = -1.f;
};

struct IcpOutput {
  std::vector<Eigen::Affine3d> transform_histry;
  std::vector<double> loss_histroty;
};

using IcpCallbackFunc =
    std::function<void(const IcpTerminateCriteria&, const IcpOutput&)>;

bool RigidIcp(const std::vector<Eigen::Vector3f>& src_points,
              const std::vector<Eigen::Vector3f>& dst_points,
              const std::vector<Eigen::Vector3f>& src_normals,
              const std::vector<Eigen::Vector3f>& dst_normals,
              const std::vector<Eigen::Vector3i>& dst_faces,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale = false, KdTreePtr<float, 3> kdtree = nullptr,
              CorrespFinderPtr corresp_finder = nullptr, int num_theads = -1,
              IcpCallbackFunc callback = nullptr, uint32_t approx_nn_num = 10);

bool RigidIcp(const std::vector<Eigen::Vector3d>& src_points,
              const std::vector<Eigen::Vector3d>& dst_points,
              const std::vector<Eigen::Vector3d>& src_normals,
              const std::vector<Eigen::Vector3d>& dst_normals,
              const std::vector<Eigen::Vector3i>& dst_faces,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale = false, KdTreePtr<double, 3> kdtree = nullptr,
              CorrespFinderPtr corresp_finder = nullptr, int num_theads = -1,
              IcpCallbackFunc callback = nullptr, uint32_t approx_nn_num = 10);

bool RigidIcp(const Mesh& src, const Mesh& dst,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale = false, KdTreePtr<float, 3> kdtree = nullptr,
              CorrespFinderPtr corresp_finder = nullptr, int num_theads = -1,
              IcpCallbackFunc callback = nullptr, uint32_t approx_nn_num = 10);

struct TransformationGIcp {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double error;

  TransformationGIcp();
  ~TransformationGIcp();

  Eigen::Vector3d Transform(const Eigen::Vector3d& p) const;

  void Update(const Eigen::Matrix<double, 6, 1>& delta);
};

struct TransformationGIcpHistory {
  std::vector<TransformationGIcp> history;
  TransformationGIcp accumlated;
  TransformationGIcpHistory();
  ~TransformationGIcpHistory();
  void Update(const TransformationGIcp& latest);
};

TransformationGIcpHistory GeneralizedIcp(
    const std::vector<Eigen::Vector3f>& src_points,
    const std::vector<Eigen::Vector3f>& dst_points,
    const std::vector<Eigen::Matrix3f>& src_covs =
        std::vector<Eigen::Matrix3f>(),
    const std::vector<Eigen::Matrix3f>& dst_covs =
        std::vector<Eigen::Matrix3f>(),
    const IcpTerminateCriteria& terminate_criteria = IcpTerminateCriteria(),
    const IcpCorrespCriteria& corresp_criateria = IcpCorrespCriteria(),
    KdTreePtr<double, 3> kdtree = nullptr, int num_theads = -1,
    IcpCallbackFunc callback = nullptr, uint32_t nn_num = 20,
    uint32_t cov_nn_num = 20, uint32_t minimize_max_iterations = 100,
    double minimize_tolerance = 1e-6);

TransformationGIcpHistory GeneralizedIcp(
    const std::vector<Eigen::Vector3d>& src_points,
    const std::vector<Eigen::Vector3d>& dst_points,
    const std::vector<Eigen::Matrix3d>& src_covs =
        std::vector<Eigen::Matrix3d>(),
    const std::vector<Eigen::Matrix3d>& dst_covs =
        std::vector<Eigen::Matrix3d>(),
    const IcpTerminateCriteria& terminate_criteria = IcpTerminateCriteria(),
    const IcpCorrespCriteria& corresp_criateria = IcpCorrespCriteria(),
    KdTreePtr<double, 3> kdtree = nullptr, int num_theads = -1,
    IcpCallbackFunc callback = nullptr, uint32_t nn_num = 20,
    uint32_t cov_nn_num = 20, uint32_t minimize_max_iterations = 100,
    double minimize_tolerance = 1e-6);

}  // namespace ugu