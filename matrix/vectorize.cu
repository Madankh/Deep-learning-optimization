#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}


// ── CPU reference ─────────────────────────────────────────────────────────────
void matrix_cpu(int M, int N, int K,
                float *A, float *B, float *C,
                float alpha, float beta) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int l = 0; l < K; l++)
                tmp += A[i * K + l] * B[l * N + j];
            C[i * N + j] = alpha * tmp + beta * C[i * N + j];
        }
}

// ── Verification ──────────────────────────────────────────────────────────────
void verify(float *cpu, float *gpu, int size, float tol) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = cpu[i] - gpu[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) errors++;
    }
    printf("Verification: %s (%d mismatches out of %d)\n",
           errors == 0 ? "PASS ✓" : "FAIL ✗", errors, size);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // Matrix dimensions
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    // Host memory sizes
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float *a_h   = (float *)malloc(size_A);
    float *b_h   = (float *)malloc(size_B);
    float *c_cpu = (float *)malloc(size_C);
    float *c_gpu = (float *)malloc(size_C);

    // Initialize matrices
    for (int i = 0; i < M * K; i++)
        a_h[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < K * N; i++)
        b_h[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < M * N; i++)
        c_cpu[i] = c_gpu[i] = 0.0f;

    // Allocate device memory
    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, size_A);
    cudaMalloc(&b_d, size_B);
    cudaMalloc(&c_d, size_C);

    // Copy input matrices to GPU
    cudaMemcpy(a_d, a_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_gpu, size_C, cudaMemcpyHostToDevice);

    // Kernel configuration
    constexpr uint BM = 64;
    constexpr uint BN = 64;
    constexpr uint BK = 8;
    constexpr uint TM = 8;
    constexpr uint TN = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        M, N, K,
        alpha,
        a_d,
        b_d,
        beta,
        c_d);
    
    cudaError_t err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Copy GPU result back to host
    cudaMemcpy(c_gpu, c_d, size_C, cudaMemcpyDeviceToHost);

    // CPU reference implementation
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    matrix_cpu(M, N, K, a_h, b_h, c_cpu, alpha, beta);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double cpu_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 +
        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // Performance metrics
    const double flops = 2.0 * M * N * K;
    const double gpu_gflops = (flops / 1e9) / (gpu_ms / 1000.0);
    const double cpu_gflops = (flops / 1e9) / (cpu_ms / 1000.0);

    printf("\n=== Performance ===\n");
    printf("Grid      : (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block     : %d\n", blockDim.x);
    printf("GPU time  : %.3f ms -> %.2f GFLOPS\n", gpu_ms, gpu_gflops);
    printf("CPU time  : %.3f ms -> %.2f GFLOPS\n", cpu_ms, cpu_gflops);
    printf("Speedup   : %.2fx\n\n", cpu_ms / gpu_ms);

    // Verify correctness
    verify(c_cpu, c_gpu, M * N, 1e-2f);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_cpu);
    free(c_gpu);

    return 0;
}