#include <cuda_runtime.h>

#include <iostream>

#include "./cuda_funcs.cuh"
#include "./helper_cuda.h"

namespace {

__global__ void BoxFilterNaive(const uint8_t* d_in, uint8_t* d_out, int width,
                               int height, int K) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;

    for (int dy = -R; dy <= R; dy++) {
      for (int dx = -R; dx <= R; dx++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int inIdx = 3 * (ny * width + nx) + c;
          sumVal += d_in[inIdx];
          count++;
        }
      }
    }
    d_out[outIdxBase + c] = (uint8_t)(sumVal / (float)count);
  }
}

__global__ void BoxFilterShared(const uint8_t* d_in, uint8_t* d_out, int width,
                                int height, int K) {
  extern __shared__ float tile[];
  // tile uses 3 channels
  // スレッドブロックが (blockDim.x * blockDim.y) 個のスレッドなら、
  // シェアードメモリの確保サイズは 3 * blockDim.x * blockDim.y (バイトではない)
  // になる想定

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  int R = (K - 1) / 2;
  int blockSize = blockDim.x * blockDim.y;

  // Write into shared memory if (x, y) is inside
  for (int c = 0; c < 3; c++) {
    int idxGlobal = 3 * (y * width + x) + c;
    int idxShared = c * blockSize + (ty * blockDim.x + tx);
    if (x < width && y < height) {
      tile[idxShared] = d_in[idxGlobal];
    } else {
      tile[idxShared] = 0.0f;  // 0 fill for outside
    }
  }

  __syncthreads();  // Wait shared memory writing

  if (x >= width || y >= height) {
    return;
  }

  // box filter 計算
  float outVal[3] = {0.0f, 0.0f, 0.0f};
  int count = 0;
  // Loop window
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      int xx = tx + dx;  // Index in shared memory
      int yy = ty + dy;
      // Refer to shared memory if inside
      if (xx >= 0 && xx < blockDim.x && yy >= 0 && yy < blockDim.y) {
        int idxSharedBase = (yy * blockDim.x + xx);
        for (int c = 0; c < 3; c++) {
          outVal[c] += tile[c * blockSize + idxSharedBase];
        }
        count++;
      } else {
        // Load from global memory if out of block boundary
        int gx = x + dx;
        int gy = y + dy;
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
          int idxGlobal = 3 * (gy * width + gx);
          for (int c = 0; c < 3; c++) {
            outVal[c] += d_in[idxGlobal + c];
          }
          count++;
        }
      }
    }
  }

  int outIdxBase = 3 * (y * width + x);
  for (int c = 0; c < 3; c++) {
    d_out[outIdxBase + c] = outVal[c] / (float)count;
  }
}

__global__ void BoxFilterRow(const uint8_t* d_in, uint8_t* d_out, int width,
                             int height, int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;
    for (int i = x - R; i <= x + R; i++) {
      if (i >= 0 && i < width) {
        int inIdx = 3 * (y * width + i) + c;
        sumVal += d_in[inIdx];
        count++;
      }
    }
    d_out[outIdxBase + c] = sumVal / (float)count;
  }
}

__global__ void BoxFilterCol(const uint8_t* d_in, uint8_t* d_out, int width,
                             int height, int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;
    for (int j = y - R; j <= y + R; j++) {
      if (j >= 0 && j < height) {
        int inIdx = 3 * (j * width + x) + c;
        sumVal += d_in[inIdx];
        count++;
      }
    }
    d_out[outIdxBase + c] = sumVal / (float)count;
  }
}

__global__ void Transpose(const uint8_t* d_in, uint8_t* d_out, int width,
                          int height) {
  //__shared__ float tile[16][16 * 3];
  extern __shared__ float tile[];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Write to shared memory
  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      tile[(threadIdx.y * blockDim.x + threadIdx.x) * 3 + c] =
          d_in[3 * (y * width + x) + c];
    }
  }

  __syncthreads();

  // Write to global memory from shared memory
  x = blockIdx.y * blockDim.x + threadIdx.x;  // transposed x
  y = blockIdx.x * blockDim.y + threadIdx.y;  // transposed y

  if (x < height && y < width) {
    for (int c = 0; c < 3; c++) {
      d_out[3 * (y * height + x) + c] =
          tile[(threadIdx.x * blockDim.y + threadIdx.y) * 3 + c];
    }
  }
}

}  // namespace

namespace ugu {

void BoxFilterCuda3b(int width, int height, void* data, int k) {
#if 1
  uint8_t *d_in, *d_out;
  size_t totalSize = sizeof(uint8_t) * 3 * width * height;
  checkCudaErrors(cudaMalloc(&d_in, totalSize));
  cudaMalloc(&d_out, totalSize);

  cudaMemcpy(d_in, data, totalSize, cudaMemcpyHostToDevice);

  int N = 1 << 20;
  int blocksize = 32;

  dim3 block(blocksize, blocksize);  // 32x32 = 1024 threads
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

#if 0
  BoxFilterNaive<<<grid, block>>>(d_in, d_out, width, height, k);
#endif

#if 0
  size_t sharedMemSize = block.x * block.y * 3 * sizeof(float);

  BoxFilterShared<<<grid, block, sharedMemSize>>>(d_in, d_out, width, height,
                                                  k);
#endif

#if 1
  uint8_t* d_temp;
  cudaMalloc(&d_temp, totalSize);

  BoxFilterRow<<<grid, block>>>(d_in, d_temp, width, height, k);
  cudaDeviceSynchronize();

  BoxFilterCol<<<grid, block>>>(d_temp, d_out, width, height, k);
#endif

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(data, d_out, totalSize, cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
#if 1
  cudaFree(d_temp);
#endif
#else

  uint8_t *d_in, *d_temp, *d_transposed, *d_out_transposed, *d_final_out;
  size_t totalSize = sizeof(uint8_t) * 3 * width * height;

  cudaMalloc(&d_in, totalSize);
  cudaMalloc(&d_temp, totalSize);
  cudaMalloc(&d_transposed, totalSize);
  cudaMalloc(&d_out_transposed, totalSize);
  cudaMalloc(&d_final_out, totalSize);

  cudaMemcpy(d_in, data, totalSize, cudaMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // 1st Pass: Row filtering
  BoxFilterRow<<<grid, block>>>(d_in, d_temp, width, height, k);
  cudaDeviceSynchronize();

  dim3 transposeBlock(16, 16);
  dim3 transposeGrid((width + transposeBlock.x - 1) / transposeBlock.x,
                     (height + transposeBlock.y - 1) / transposeBlock.y);

  size_t sharedMemSize = block.x * block.y * 3 * sizeof(float);

  // Transpose
  Transpose<<<transposeGrid, transposeBlock, sharedMemSize>>>(
      d_temp, d_transposed, width, height);
  cudaDeviceSynchronize();

  // 2nd Pass: Row filtering for transposed image
  dim3 grid2((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);

  BoxFilterRow<<<grid2, block>>>(d_transposed, d_out_transposed, height, width,
                                 k);
  cudaDeviceSynchronize();

  // Transpose again（d_out_transposed -> d_final_out）
  dim3 transposeGridDim2((height + transposeBlock.y - 1) / transposeBlock.y,
                         (width + transposeBlock.x - 1) / transposeBlock.x);

  Transpose<<<transposeGridDim2, transposeBlock, sharedMemSize>>>(
      d_out_transposed, d_final_out, height, width);
  cudaDeviceSynchronize();

  cudaMemcpy(data, d_final_out, totalSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_temp);
  cudaFree(d_transposed);
  cudaFree(d_out_transposed);
  cudaFree(d_final_out);
#endif
}

}  // namespace ugu