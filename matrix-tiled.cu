#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

__global__ void matrix_mul_tiled(const float* __restrict__ M, const float* __restrict__ N, float* __restrict__ P, int n) {
  assert(n % blockDim.x == 0); // For simplicity, assume n is divisible by blockDim.x.
  assert(n % blockDim.y == 0); // For simplicity, assume n is divisible by blockDim.y.
  assert(blockDim.x == blockDim.y); // For simplicity, assume square blocks.
  assert(blockDim.x == 16);

  int num_tiles = n / blockDim.x;
  float sum = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < num_tiles; ++i) {
    __shared__ float M_tile[16][16];
    __shared__ float N_tile[16][16];
    M_tile[threadIdx.y][threadIdx.x] = M[row * n + (i * blockDim.x + threadIdx.x)];
    N_tile[threadIdx.y][threadIdx.x] = N[(i * blockDim.y + threadIdx.y) * n + col];
    __syncthreads();
    for(int j = 0; j < blockDim.x; ++j) {
      sum += M_tile[threadIdx.y][j] * N_tile[j][threadIdx.x];
    }
    __syncthreads();
  }
  P[row * n + col] = sum;
}

// Compute P = M * N, where M,N,P are matrices of size n * n, stored in row-major order.
__global__ void matrix_mul_naive(const float* __restrict__ M, const float* __restrict__ N, float* __restrict__ P, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n * n) {
    float sum = 0;
    int row = tid / n;
    int col = tid % n;
    for(int i = 0; i < n; ++i) {
      sum += M[row * n + i] * N[i * n + col];
    }
    P[tid] = sum;
  }
}

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if(err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

void matrix_mul_cpu(const std::vector<float>& M, const std::vector<float>& N, std::vector<float>& P, int n) {
  #pragma omp parallel for
  for(int row = 0; row < n; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(int i = 0; i < n; ++i) {
        sum += M[row * n + i] * N[i * n + col];
      }
      P[row * n + col] = sum;
    }
  }
}

// Usage: ./matrix-tiling [n] [mode] [-test], where mode is "gpu" (default) or "cpu".
// Add the suffix "-tiled" to "gpu" to use matrix_mul_tiled instead of matrix_mul_naive (i.e. "gpu-tiled"); n must then be a multiple of 16.
// The option -test (disabled by default because it is slow) runs both the GPU and CPU versions and compares their results.
// The number of CPU threads is controlled by OMP_NUM_THREADS.
int main(int argc, char** argv) {
  bool test = false;
  const char* args[2] = {nullptr, nullptr};
  int num_args = 0;
  for(int i = 1; i < argc; ++i) {
    if(strcmp(argv[i], "-test") == 0) {
      test = true;
    }
    else if(num_args < 2) {
      args[num_args++] = argv[i];
    }
  }
  int n = args[0] ? atoi(args[0]) : 512;
  const char* mode = args[1] ? args[1] : "gpu";
  bool run_gpu = test || strcmp(mode, "cpu") != 0;
  bool run_cpu = test || strncmp(mode, "gpu", 3) != 0;
  bool tiled = strstr(mode, "-tiled") != nullptr;
  size_t bytes = (size_t)n * n * sizeof(float);

  std::vector<float> M(n * n), N(n * n), P(n * n), P_ref(n * n);
  srand(0);
  for(int i = 0; i < n * n; ++i) {
    M[i] = (float)rand() / RAND_MAX;
    N[i] = (float)rand() / RAND_MAX;
  }

  if(run_gpu) {
    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, bytes));
    CUDA_CHECK(cudaMalloc(&d_N, bytes));
    CUDA_CHECK(cudaMalloc(&d_P, bytes));
    CUDA_CHECK(cudaMemcpy(d_M, M.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, N.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_P, 0, bytes));

    // One thread per element of P: a 1D grid for the naive kernel, a 2D grid of 16 x 16 tiles for the tiled kernel.
    dim3 threads_per_block(256);
    dim3 num_blocks((n * n + threads_per_block.x - 1) / threads_per_block.x);
    if(tiled) {
      threads_per_block = dim3(16, 16);
      num_blocks = dim3(n / 16, n / 16);
    }
    auto launch = [&](dim3 blocks, dim3 threads, int size) {
      if(tiled) {
        matrix_mul_tiled<<<blocks, threads>>>(d_M, d_N, d_P, size);
      }
      else {
        matrix_mul_naive<<<blocks, threads>>>(d_M, d_N, d_P, size);
      }
    };
    // Warm-up: the first launch pays one-time initialization costs; with n = 0 the kernel does no work.
    launch(1, threads_per_block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Kernel launches are asynchronous: synchronize before stopping the timer.
    auto start = std::chrono::steady_clock::now();
    launch(num_blocks, threads_per_block, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    printf("Kernel time: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
    CUDA_CHECK(cudaMemcpy(P.data(), d_P, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
  }

  if(run_cpu) {
    // Warm-up: the first parallel region creates the OpenMP thread pool.
    #pragma omp parallel
    {}
    auto start = std::chrono::steady_clock::now();
    matrix_mul_cpu(M, N, P_ref, n);
    auto end = std::chrono::steady_clock::now();
    printf("CPU time: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
  }
  if(!test) {
    return EXIT_SUCCESS;
  }

  int errors = 0;
  for(int i = 0; i < n * n; ++i) {
    if(fabsf(P[i] - P_ref[i]) > 1e-3f * fmaxf(1.0f, fabsf(P_ref[i]))) {
      if(errors < 10) {
        printf("Mismatch at (%d, %d): GPU = %f, CPU = %f\n", i / n, i % n, P[i], P_ref[i]);
      }
      ++errors;
    }
  }
  if(errors == 0) {
    printf("Test PASSED (n = %d)\n", n);
  }
  else {
    printf("Test FAILED: %d / %d incorrect elements (n = %d)\n", errors, n * n, n);
  }
  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
