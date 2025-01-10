#include "ugu/cuda/cuda_funcs.h"

#ifdef UGU_USE_CUDA
#include "./cuda_funcs.cuh"
#endif

namespace ugu {

#ifdef UGU_USE_CUDA
void BoxFilterCuda(Image3b& img, int kernel) {
  BoxFilterCuda3b(img.cols, img.rows, img.data, kernel);
}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}

class kNnGridCuda::Impl {
 public:
  Impl();
  ~Impl();
  void SetData(const std::vector<Eigen::Vector3f>& data);
  void Build();
  void SearchNn(const Eigen::Vector3f& query, Eigen::Vector3f& nn) const;

 private:
};

kNnGridCuda::Impl::Impl() {}
kNnGridCuda::Impl::~Impl() {}

kNnGridCuda::kNnGridCuda() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}
kNnGridCuda::~kNnGridCuda() {}

void kNnGridCuda::SetData(const std::vector<Eigen::Vector3f>& data) {}

void kNnGridCuda::Build() {}

void kNnGridCuda::SearchNn(const Eigen::Vector3f& query,
                           Eigen::Vector3f& nn) const {}
#else
void BoxFilterCuda(Image3b& img, int kernel) {}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}

class kNnGridCuda::Impl {
 public:
  Impl();
  ~Impl();
  void SetData(const std::vector<Eigen::Vector3f>& data);
  void Build();
  void SearchNn(const Eigen::Vector3f& query, Eigen::Vector3f& nn) const;

 private:
};

kNnGridCuda::Impl::Impl() {}
kNnGridCuda::Impl::~Impl() {}

kNnGridCuda::kNnGridCuda() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}
kNnGridCuda::~kNnGridCuda() {}

void kNnGridCuda::SetData(const std::vector<Eigen::Vector3f>& data) {}

void kNnGridCuda::Build() {}

void kNnGridCuda::SearchNn(const Eigen::Vector3f& query,
                           Eigen::Vector3f& nn) const {}
#endif

}  // namespace ugu
