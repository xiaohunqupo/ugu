#pragma once

#include <vector>

#include "Eigen/Core"

namespace ugu {

void SearchKnnCuda(const float* d_data, const uint32_t* d_voxel_start_indices,
                   const uint32_t* d_voxel_point_indices, uint32_t grid_size_x,
                   uint32_t grid_size_y, uint32_t grid_size_z, float voxel_len,
                   Eigen::Vector3f min_bound, float* d_queries,
                   uint32_t num_queries, uint32_t k, uint32_t* d_knn_indices,
                   float* d_knn_dists);
}
// namespace ugu