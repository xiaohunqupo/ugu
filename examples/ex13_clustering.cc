/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/clustering/clustering.h"
#include "ugu/mesh.h"

namespace {
std::default_random_engine engine(0);

void SavePoints(const std::string& ply_path,
                const std::vector<Eigen::VectorXf>& points, int num_clusters,
                const std::vector<size_t>& labels,
                const std::vector<Eigen::VectorXf>& centroids) {
  ugu::Mesh mesh;
  std::vector<Eigen::Vector3f> points3d;
  // Points
  std::transform(
      points.begin(), points.end(), std::back_inserter(points3d),
      [](const auto& p) { return Eigen::Vector3f(p[0], p[1], p[2]); });
  // Centroids
  std::transform(
      centroids.begin(), centroids.end(), std::back_inserter(points3d),
      [](const auto& p) { return Eigen::Vector3f(p[0], p[1], p[2]); });
  mesh.set_vertices(points3d);

  std::vector<Eigen::Vector3f> point_colors;
  std::vector<Eigen::Vector3f> label_colors;
  std::uniform_real_distribution<float> color_dstr(0.f, 255.f);
  // Random color for each cluster
  for (size_t i = 0; i < num_clusters; i++) {
    label_colors.emplace_back(color_dstr(engine), color_dstr(engine),
                              color_dstr(engine));
  }
  for (size_t i = 0; i < points.size(); i++) {
    point_colors.push_back(label_colors[labels[i]]);
  }
  for (size_t i = 0; i < num_clusters; i++) {
    // Red for cluster centroids
    point_colors.emplace_back(255.f, 0.f, 0.f);
  }
  mesh.set_vertex_colors(point_colors);
  mesh.WritePly(ply_path);
}

void KMeansTest() {
  std::vector<Eigen::VectorXf> points;
  float r = 1.f;
  std::normal_distribution<float> dstr(0.f, r);

  size_t pc = 3000;
  size_t gt_clusters = 10;
  for (size_t i = 0; i < pc; i++) {
    auto gt_cluster = i % gt_clusters;
    float offset = static_cast<float>(gt_cluster * r);
    float x = dstr(engine) + offset * 3.f;
    float y = dstr(engine);
    float z = dstr(engine) + offset * 3.f;
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }

  std::vector<size_t> labels;
  std::vector<Eigen::VectorXf> centroids;
  std::vector<float> dists;
  std::vector<std::vector<Eigen::VectorXf>> clustered_points;
  int num_clusters = 10;
  ugu::KMeans(points, num_clusters, labels, centroids, dists, clustered_points,
              100, 1.f, false, 0);
  SavePoints("kmeans_naive.ply", points, num_clusters, labels, centroids);

  ugu::KMeans(points, num_clusters, labels, centroids, dists, clustered_points,
              100, 1.f, true, 0);
  SavePoints("kmeans_plusplus.ply", points, num_clusters, labels, centroids);
}

void MeanShiftTest() {
  std::vector<Eigen::VectorXf> points;
  float r = 1.f;
  std::normal_distribution<float> dstr(0.f, r);

  size_t pc = 3000;
  for (size_t i = 0; i < pc; i++) {
    float x = dstr(engine);
    float y = dstr(engine);
    float z = dstr(engine);
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }

  Eigen::VectorXf init(3);
  init[0] = r;
  init[1] = r;
  init[2] = r;

  Eigen::VectorXf node(3);
  ugu::LOGI("init      (%f %f %f)\n", init[0], init[1], init[2]);
  ugu::MeanShift(points, init, r, 0.0001f, 100, node);
  ugu::LOGI("converged (%f %f %f)\n", node[0], node[1], node[2]);
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  KMeansTest();

  MeanShiftTest();

  return 0;
}
