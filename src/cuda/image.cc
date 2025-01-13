#include "ugu/cuda/image.h"

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>

#include "./image.cuh"

#endif

namespace ugu {

#ifdef UGU_USE_CUDA
void BoxFilterCuda(Image3b& img, int kernel) {
  BoxFilterCuda3b(img.cols, img.rows, img.data, kernel);
}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}

#else
void BoxFilterCuda(Image3b& img, int kernel) {}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}
#endif

}  // namespace ugu
