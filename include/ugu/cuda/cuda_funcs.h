#pragma once

#include <vector>
#include <memory>

#include "Eigen/Core"
#include "ugu/image.h"

namespace ugu {

void BoxFilterCuda(Image3b& img, int kernel);

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel);

class kNnGridCuda {
 public:
  kNnGridCuda();
  ~kNnGridCuda();

  void SetData(const std::vector<Eigen::Vector3f>& data);
  void Build();
  void SearchNn(const Eigen::Vector3f& query, Eigen::Vector3f& nn) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};
}  // namespace ugu
