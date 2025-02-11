/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include <filesystem>
#include <iostream>
#include <random>

#include "ugu/registration/rigid.h"
#include "ugu/timer.h"
#include "ugu/util/path_util.h"
#include "ugu/util/string_util.h"

namespace {

bool GetFileNames(std::string folderPath,
                  std::vector<std::string>& file_names) {
  using namespace std::filesystem;
  directory_iterator iter(folderPath), end;
  std::error_code err;

  for (; iter != end && !err; iter.increment(err)) {
    const directory_entry entry = *iter;
    std::string name = *(ugu::Split(entry.path().string(), '/').end() - 1);
    file_names.push_back(name);
  }

  if (err) {
    std::cout << err.value() << std::endl;
    std::cout << err.message() << std::endl;
    return false;
  }
  return true;
}

auto AddNoise(ugu::Mesh& mesh) {
  // Add noise
  ugu::Mesh noised_mesh = ugu::Mesh(mesh);
  mesh.CalcStats();
  const auto& stats = mesh.stats();
  std::mt19937 engine(0);
  float bb_mean = (stats.bb_max - stats.bb_min).mean();
  float sigma = bb_mean * 0.01f;
  std::normal_distribution<float> gauss(0.0f, sigma);
  auto org_vertices = mesh.vertices();
  auto noised_vertices = org_vertices;
  for (size_t i = 0; i < noised_vertices.size(); i++) {
    auto& v = noised_vertices[i];
    auto& n = mesh.normals()[i];
    v += gauss(engine) * n;
  }
  noised_mesh.set_vertices(noised_vertices);

  Eigen::Vector3f noise_t{bb_mean / 5, bb_mean * 2 / 5, bb_mean * -3 / 5};
  Eigen::Vector3f axis(5, 2, 1);
  axis.normalize();
  Eigen::AngleAxisf noise_R(ugu::radians(10.f), axis);
  float noise_s = 1.f;
  Eigen::Affine3f T_Rts_gt = Eigen::Translation3f(noise_t) * noise_R.matrix() *
                             Eigen::Scaling(noise_s);
  noised_mesh.Scale(noise_s);
  noised_mesh.Transform(T_Rts_gt.rotation(), T_Rts_gt.translation());
  std::cout << "GT Rts" << std::endl;
  std::cout << T_Rts_gt.matrix() << std::endl;

  return std::make_tuple(T_Rts_gt, noised_mesh);
}

void TestAlignmentWithCorresp() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny;
  std::string out_dir = "../out/ex23/corresp/";
  ugu::EnsureDirExists(out_dir);
  bunny.LoadObj(in_obj_path1, data1_dir);

  // Add noise
  ugu::Mesh noised_bunny = ugu::Mesh(bunny);
  bunny.CalcStats();
  const auto& stats = bunny.stats();
  std::mt19937 engine(0);
  float bb_mean = (stats.bb_max - stats.bb_min).mean();
  float sigma = bb_mean * 0.01f;
  std::normal_distribution<float> gauss(0.0f, sigma);
  auto org_vertices = bunny.vertices();
  auto noised_vertices = org_vertices;
  for (size_t i = 0; i < noised_vertices.size(); i++) {
    auto& v = noised_vertices[i];
    auto& n = bunny.normals()[i];
    v += gauss(engine) * n;
  }
  noised_bunny.set_vertices(noised_vertices);

  Eigen::Vector3f noise_t{bb_mean, bb_mean * 2, bb_mean * -3};
  Eigen::Vector3f axis(5, 2, 1);
  axis.normalize();
  Eigen::AngleAxisf noise_R(ugu::radians(30.f), axis);
  float noise_s = 0.45f;
  Eigen::Affine3f T_Rt_gt = Eigen::Translation3f(noise_t) * noise_R.matrix();
  ugu::Mesh noised_bunny_Rt = ugu::Mesh(noised_bunny);
  noised_bunny_Rt.Transform(T_Rt_gt.rotation(), T_Rt_gt.translation());
  noised_bunny_Rt.WriteObj(out_dir, "noised_Rt_gt");
  std::cout << "GT Rt" << std::endl;
  std::cout << T_Rt_gt.matrix() << std::endl;

  Eigen::Affine3f T_Rt_estimated = ugu::FindRigidTransformFrom3dCorrespondences(
                                       org_vertices, noised_bunny_Rt.vertices())
                                       .cast<float>();
  std::cout << "Estimated Rt" << std::endl;
  std::cout << T_Rt_estimated.matrix() << std::endl;
  bunny.Transform(T_Rt_estimated.rotation(), T_Rt_estimated.translation());
  bunny.WriteObj(out_dir, "noised_Rt_estimated");
  bunny.Transform(T_Rt_estimated.inverse().rotation(),
                  T_Rt_estimated.inverse().translation());

  ugu::Mesh noised_bunny_Rts = ugu::Mesh(noised_bunny);
  Eigen::Affine3f T_Rts_gt = Eigen::Translation3f(noise_t) * noise_R.matrix() *
                             Eigen::Scaling(noise_s);
  noised_bunny_Rts.Scale(noise_s);
  noised_bunny_Rts.Transform(T_Rts_gt.rotation(), T_Rts_gt.translation());
  noised_bunny_Rts.WriteObj(out_dir, "noised_Rts_gt");
  std::cout << "GT Rts" << std::endl;
  std::cout << T_Rts_gt.matrix() << std::endl;

  Eigen::Affine3f T_Rts_estimated =
      ugu::FindSimilarityTransformFrom3dCorrespondences(
          org_vertices, noised_bunny_Rts.vertices())
          .cast<float>();
  std::cout << "Estimated Rts" << std::endl;
  std::cout << T_Rts_estimated.matrix() << std::endl;
  float estimated_s_x =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(0).norm();
  float estimated_s_y =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(1).norm();
  float estimated_s_z =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(2).norm();
  bunny.Scale(estimated_s_x, estimated_s_y, estimated_s_z);
  bunny.Transform(T_Rts_estimated.rotation(), T_Rts_estimated.translation());
  bunny.WriteObj(out_dir, "noised_Rts_estimated");
}

void TestAlignmentWithoutCorresp() {
  std::string data1_dir = "../data/bunny/";

  std::string out_dir = "../out/ex23/icp/";
  ugu::EnsureDirExists(out_dir);
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny;
  bunny.LoadObj(in_obj_path1, data1_dir);

  auto [T_Rts_gt, noised_mesh] = AddNoise(bunny);

  noised_mesh.WriteObj(out_dir, "noised_init");

  ugu::Mesh org_noised_mesh = noised_mesh;
  ugu::Timer<double> timer;
  {
    noised_mesh = org_noised_mesh;
    ugu::IcpTerminateCriteria tmc;
    ugu::IcpCorrespCriteria crc;
    ugu::IcpOutput output;
    timer.Start();
    tmc.iter_max = 100;
    auto history =
        ugu::GeneralizedIcp(noised_mesh.vertices(), bunny.vertices(), {}, {},
                            tmc, crc, nullptr, -1, nullptr, 20, 20, 1, 1e-2);
    timer.End();
    ugu::LOGI("Generalized ICP: %fms\n", timer.elapsed_msec());
    for (size_t i = 0; i < history.history.size(); i++) {
      ugu::LOGI("iter %d: %f\n", i, history.history[i].error);
      noised_mesh = org_noised_mesh;
      noised_mesh.Transform(history.history[i].R.cast<float>(),
                            history.history[i].t.cast<float>());
      noised_mesh.WriteObj(out_dir, "noised_gicp_" + std::to_string(i));
    }
  }

  {
    noised_mesh = org_noised_mesh;
    ugu::IcpTerminateCriteria tmc;
    ugu::IcpCorrespCriteria crc;
    ugu::IcpOutput output;
    timer.Start();
    ugu::RigidIcp(noised_mesh, bunny, ugu::IcpCorrespType::kPointToPoint,
                  ugu::IcpLossType::kPointToPoint, tmc, crc, output, false);
    timer.End();
    ugu::LOGI(
        "Correspondence: Point-to-Point, Loss: Point-to-Point ICP: %fms\n",
        timer.elapsed_msec());
    for (size_t i = 0; i < output.transform_histry.size(); i++) {
      ugu::LOGI("iter %d: %f\n", i, output.loss_histroty[i]);
      noised_mesh = org_noised_mesh;
      noised_mesh.Transform(output.transform_histry[i].cast<float>());
      noised_mesh.WriteObj(out_dir,
                           "noised_icp_point_point_" + std::to_string(i));
    }
  }

  {
    noised_mesh = org_noised_mesh;
    ugu::IcpTerminateCriteria tmc;
    ugu::IcpCorrespCriteria crc;
    ugu::IcpOutput output;
    timer.Start();
    ugu::RigidIcp(noised_mesh, bunny, ugu::IcpCorrespType::kPointToPlane,
                  ugu::IcpLossType::kPointToPoint, tmc, crc, output, false);
    timer.End();
    ugu::LOGI(
        "Correspondence: Point-to-Plane, Loss: Point-to-Point ICP: %fms\n",
        timer.elapsed_msec());
    for (size_t i = 0; i < output.transform_histry.size(); i++) {
      ugu::LOGI("iter %d: %f\n", i, output.loss_histroty[i]);
      noised_mesh = org_noised_mesh;
      noised_mesh.Transform(output.transform_histry[i].cast<float>());
      noised_mesh.WriteObj(out_dir,
                           "noised_icp_plane_point_" + std::to_string(i));
    }
  }

  {
    noised_mesh = org_noised_mesh;
    ugu::IcpTerminateCriteria tmc;
    ugu::IcpCorrespCriteria crc;
    ugu::IcpOutput output;
    timer.Start();
    ugu::RigidIcp(noised_mesh, bunny, ugu::IcpCorrespType::kPointToPoint,
                  ugu::IcpLossType::kPointToPlane, tmc, crc, output, false);
    timer.End();

    ugu::LOGI(
        "Correspondence: Point-to-Point, Loss: Point-to-Plane ICP: %f ms\n",
        timer.elapsed_msec());
    for (size_t i = 0; i < output.transform_histry.size(); i++) {
      ugu::LOGI("iter %d: %f\n", i, output.loss_histroty[i]);
      noised_mesh = org_noised_mesh;
      noised_mesh.Transform(output.transform_histry[i].cast<float>());
      noised_mesh.WriteObj(out_dir,
                           "noised_icp_point_plane_" + std::to_string(i));
    }
  }

  {
    noised_mesh = org_noised_mesh;
    ugu::IcpTerminateCriteria tmc;
    ugu::IcpCorrespCriteria crc;
    ugu::IcpOutput output;
    timer.Start();
    ugu::RigidIcp(noised_mesh, bunny, ugu::IcpCorrespType::kPointToPlane,
                  ugu::IcpLossType::kPointToPlane, tmc, crc, output, false);
    timer.End();

    ugu::LOGI(
        "Correspondence: Point-to-Plane, Loss: Point-to-Plane ICP: %f ms\n",
        timer.elapsed_msec());
    for (size_t i = 0; i < output.transform_histry.size(); i++) {
      ugu::LOGI("iter %d: %f\n", i, output.loss_histroty[i]);
      noised_mesh = org_noised_mesh;
      noised_mesh.Transform(output.transform_histry[i].cast<float>());
      noised_mesh.WriteObj(out_dir,
                           "noised_icp_plane_plane_" + std::to_string(i));
    }
  }
}

}  // namespace

int main() {
  ugu::EnsureDirExists("../out/");
  ugu::EnsureDirExists("../out/ex23/");

  TestAlignmentWithCorresp();

  TestAlignmentWithoutCorresp();

  return 0;
}
