#pragma once

#include <cuda_runtime.h>

#include <iostream>

#include "./helper_cuda.h"
#include "./knn.cuh"

namespace {

// CUDAカーネル: 各クエリ点に対するkNN検索
__global__ void kNNKernel(const Eigen::Vector3f* d_points,
                          const uint32_t* d_voxel_start_indices,
                          const uint32_t* d_voxel_point_indices,
                          uint32_t grid_size_x, uint32_t grid_size_y,
                          uint32_t grid_size_z, float voxel_len,
                          Eigen::Vector3f min_bound,
                          const Eigen::Vector3f* d_queries,
                          uint32_t num_queries, uint32_t k,
                          uint32_t* d_knn_indices, float* d_knn_dists) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_queries) return;

  Eigen::Vector3f query = d_queries[idx];

  uint32_t vx = static_cast<uint32_t>((query.x() - min_bound.x()) / voxel_len);
  uint32_t vy = static_cast<uint32_t>((query.y() - min_bound.y()) / voxel_len);
  uint32_t vz = static_cast<uint32_t>((query.z() - min_bound.z()) / voxel_len);

  vx = min(max(vx, 0), grid_size_x - 1);
  vy = min(max(vy, 0), grid_size_y - 1);
  vz = min(max(vz, 0), grid_size_z - 1);

  // extern __shared__ float sharedDistances[];
  float* local_distances;
  uint32_t* local_indices;
  cudaMalloc(&local_distances, k * sizeof(float));
  cudaMalloc(&local_indices, k * sizeof(uint32_t));
  for (int i = 0; i < k; ++i) {
    local_distances[i] = FLT_MAX;
    local_indices[i] = UINT32_MAX;
  }

  // Determine the range of search
  int range_count = 0;
  int range = 0;
  while (true) {
    if (range_count > (grid_size_x + grid_size_y + grid_size_z) / 3 / 2) {
      range = range_count;
      break;
    }
    int range_max = range_count + 1;
    int range_min = -range_max;
    uint32_t num_data_in_range = 0;
    for (int dx = range_min; dx <= range_max; ++dx) {
      for (int dy = range_min; dy <= range_max; ++dy) {
        for (int dz = range_min; dz <= range_max; ++dz) {
          int nx = vx + dx;
          int ny = vy + dy;
          int nz = vz + dz;

          if (nx < 0 || ny < 0 || nz < 0 || nx >= grid_size_x ||
              ny >= grid_size_y || nz >= grid_size_z)
            continue;

          uint32_t neighbor_voxel =
              nx + ny * grid_size_x + nz * grid_size_x * grid_size_y;
          uint32_t start = d_voxel_start_indices[neighbor_voxel];
          uint32_t end =
              (neighbor_voxel == (grid_size_x * grid_size_y * grid_size_z - 1))
                  ? d_voxel_start_indices[neighbor_voxel] +
                        (d_voxel_start_indices[neighbor_voxel + 1] -
                         d_voxel_start_indices[neighbor_voxel])
                  : d_voxel_start_indices[neighbor_voxel + 1];

          num_data_in_range += end - start;
        }
      }
    }
    if (k <= num_data_in_range) {
      range = range_count;
      break;
    }
    range_count++;
  }

  // TODO: This range is still not accurate. Accuracy depends on the postion of
  // the query point in the voxel.

  // Perform kNN search with the range
  int range_max = range + 1;
  int range_min = -range_max;
  for (int dx = range_min; dx <= range_max; ++dx) {
    for (int dy = range_min; dy <= range_max; ++dy) {
      for (int dz = range_min; dz <= range_max; ++dz) {
        int nx = vx + dx;
        int ny = vy + dy;
        int nz = vz + dz;

        if (nx < 0 || ny < 0 || nz < 0 || nx >= grid_size_x ||
            ny >= grid_size_y || nz >= grid_size_z)
          continue;

        int neighbor_voxel =
            nx + ny * grid_size_x + nz * grid_size_x * grid_size_y;
        int start = d_voxel_start_indices[neighbor_voxel];
        int end =
            (neighbor_voxel == (grid_size_x * grid_size_y * grid_size_z - 1))
                ? d_voxel_start_indices[neighbor_voxel] +
                      (d_voxel_start_indices[neighbor_voxel + 1] -
                       d_voxel_start_indices[neighbor_voxel])
                : d_voxel_start_indices[neighbor_voxel + 1];

        for (int p = start; p < end; ++p) {
          uint32_t point_idx = d_voxel_point_indices[p];
          if (point_idx == UINT32_MAX) {
            continue;
          }

          Eigen::Vector3f point = d_points[point_idx];
          float distance = (query - point).squaredNorm();

          // Update kNN
          for (int i = 0; i < k; ++i) {
            if (distance < local_distances[i]) {
              // Shift to the right
              for (int j = k - 1; j > i; --j) {
                local_distances[j] = local_distances[j - 1];
                local_indices[j] = local_indices[j - 1];
              }
              local_distances[i] = distance;
              local_indices[i] = point_idx;
              break;
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < k; ++i) {
    d_knn_indices[idx * k + i] = local_indices[i];
    d_knn_dists[idx * k + i] = local_distances[i];
  }

  cudaFree(local_distances);
  cudaFree(local_indices);
}

}  // namespace

namespace ugu {

void SearchKnnCuda(const float* d_data, const uint32_t* d_voxel_start_indices,
                   const uint32_t* d_voxel_point_indices, uint32_t grid_size_x,
                   uint32_t grid_size_y, uint32_t grid_size_z, float voxel_len,
                   Eigen::Vector3f min_bound, float* d_queries,
                   uint32_t num_queries, uint32_t k, uint32_t* d_knn_indices,
                   float* d_knn_dists) {
  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;

  // kNNカーネルの起動

  kNNKernel<<<blocks, threads>>>(
      reinterpret_cast<Eigen::Vector3f*>(const_cast<float*>(d_data)),
      d_voxel_start_indices, d_voxel_point_indices, grid_size_x, grid_size_y,
      grid_size_z, voxel_len, min_bound,
      reinterpret_cast<Eigen::Vector3f*>(const_cast<float*>(d_queries)),
      num_queries, k, d_knn_indices, d_knn_dists);

  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace ugu