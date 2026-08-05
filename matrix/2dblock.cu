#include<stdio.h>
#include<cuda_runtime.h>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template<int BM, int BN, int BK, int TM, int TN>
__global__ void block2d(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    // assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (BN/TN);
    const int threadRow = threadIdx.x / (BN/TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    int innerColA = threadIdx.x % BK; 
    int innerRowA = threadIdx.x / BK;
    int strideA = numThreadsBlocktile / BK;
    
    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;
    int strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int bidx=0; bidx < K; bidx+=BK){
        for(uint loadOffset = 0; loadOffset < BM; loadOffset+=strideA){
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }

        for(uint loadOffset = 0; loadOffset < BK; loadOffset+=strideB){
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for(int dotIdx=0; dotIdx < BK; dotIdx++){
            for(int i=0; i<TM; i++){
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            };

            for(int j=0; j<TN; j++){
                regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
            };

            for(uint resIdxM = 0; resIdxM < TM; resIdxM++){
                for(uint resIdxN=0; resIdxN < TN; resIdxN++){
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
        
    }
    // write out the results
    for(uint resIdxM=0; resIdxM < TM; resIdxM++){
        for(uint resIdxN=0; resIdxN < TN; resIdxN++){
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadResults[resIdxM * TN + resIdxN] + 
            beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }

}

void matrix_cpu(int M, int N, int K, float* A, float* B, float* C, float beta, float alpha){
  for(int i=0; i < M; i++){
    for(int j=0; j < N; j++){
      float tmp=0.0;
      for(int l=0; l < K; l++){
        tmp+= A[i * K + l] * B[l * N + j];
      }
      C[i*N + j] = alpha * tmp + beta * C[i*N+j];
    }
  }
}

// ── verify GPU result against CPU reference ──────────────────────────────────
void verify(float* cpu, float* gpu, int size, float tol){
  int errors = 0;
  for(int i = 0; i < size; i++){
    float diff = cpu[i] - gpu[i];
    if(diff < 0) diff = -diff;
    if(diff > tol) errors++;
  }
  printf("Verification: %s (%d mismatches out of %d)\n",
         errors == 0 ? "PASS ✓" : "FAIL ✗", errors, size);
}


int main() {
    int M = 4096;
    int K = 1024;
    int N = 4096;

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *a_h = (float*)malloc(size_A);
    float *b_h = (float*)malloc(size_B);
    float *c_cpu = (float*)malloc(size_C);
    float *c_gpu = (float*)malloc(size_C);

    float *a_d, *b_d, *c_d;

    cudaMalloc(&a_d, size_A);
    cudaMalloc(&b_d, size_B);
    cudaMalloc(&c_d, size_C);

    // Initialize A
    for (int i = 0; i < M * K; i++)
        a_h[i] = (float)rand() / RAND_MAX;

    // Initialize B
    for (int i = 0; i < K * N; i++)
        b_h[i] = (float)rand() / RAND_MAX;

    // Initialize C
    for (int i = 0; i < M * N; i++) {
        c_cpu[i] = 0.0f;
        c_gpu[i] = 0.0f;
    }

    cudaMemcpy(a_d, a_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_gpu, size_C, cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // GPU Timing
    // --------------------------------------------------

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    cudaEventRecord(start);

    block2d<BM, BN, BK, TM,TN><<<gridDim, blockDim>>>(
        M, N, K,
        a_d, b_d, c_d,
        alpha,beta
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n",
               cudaGetErrorString(err));
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(
        c_gpu,
        c_d,
        size_C,
        cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------
    // CPU Timing
    // --------------------------------------------------

    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    matrix_cpu(
        M, N, K,
        a_h, b_h,
        c_cpu,
        beta, alpha
    );

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double cpu_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 +
        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // --------------------------------------------------
    // Performance
    // --------------------------------------------------

    double flops = 2.0 * M * N * K;

    double gpu_gflops =
        (flops / 1e9) / (gpu_ms / 1000.0);

    double cpu_gflops =
        (flops / 1e9) / (cpu_ms / 1000.0);

    printf("\n=== Performance ===\n");
    printf("Grid      : (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block     : %d\n", blockDim.x);
    printf("GPU time  : %.3f ms  -> %.2f GFLOPS\n",
           gpu_ms, gpu_gflops);
    printf("CPU time  : %.3f ms  -> %.2f GFLOPS\n",
           cpu_ms, cpu_gflops);
    printf("Speedup   : %.2fx\n\n",
           cpu_ms / gpu_ms);

    // --------------------------------------------------
    // Verification
    // --------------------------------------------------

    verify(c_cpu, c_gpu, M * N, 1e-2f);

    // --------------------------------------------------
    // Cleanup
    // --------------------------------------------------

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