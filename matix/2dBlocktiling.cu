#include<stdio.h>
#include<cuda_runtime.h>
#include<algorithm>
#define CEIL_DIV(M,N) (((M) + (N) -1) / (N))
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int totalThreads = BM * BN;
    const int threadBlocktile = totalThreads / (TM * TN);

    const int threadCol = threadIdx.x % (BN/TN);
    const int threadRow = threadIdx.x / (BN/TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    const int strideA = threadBlocktile / BK;
    const int strideB = threadBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int bkIdx=0; bkIdx < K; bkIdx += BK){
        for(int loadOffset = 0; loadOffset < BM; loadOffset+=strideA){
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }

        for(int loadOffset = 0; loadOffset < BK; loadOffset += strideB){
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for(int dotIdx=0; dotIdx < BK; ++dotIdx){
            for(int resIdx=0; resIdx < TM; resIdx++){
                regM[resIdx] = As[(threadRow * TM + resIdx) * BK + dotIdx];
            }

            for(int resIdx=0; resIdx < TN; resIdx++){
                regN[resIdx] = Bs[dotIdx * BN  + threadCol * TN + resIdx];
            }

            for(int resIdxM = 0; resIdxM < TM; resIdxM++){
                for(int resIdxN = 0; resIdxN < TN; resIdxN++){
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for(int resIdxM=0; resIdxM < TM; resIdxM++){
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }

}


void init(int n, float* vec){
    for(int i=0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to verify results
bool verify_result(float *cpu_result, float *gpu_result, int size, float tolerance=1e-3){
    for(int i=0; i<size; i++){
        if(abs(cpu_result[i] - gpu_result[i]) > tolerance){
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}


void sgemm_cpu(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float temp = 0.0;
            for(int k = 0; k < K; k++) {
                temp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * temp + beta * C[i * N + j];
        }
    }
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    int M = 1024;
    int N = 1024;
    int K = 1024;

    const uint BK_T = 8;
    const uint TM_T = 8;
    const uint TN_T = 8;
    const uint BM_T = 128;
    const uint BN_T = 128;
    
    float alpha = 1.0f;
    float beta = 0.0f;

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;

    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);
    
    if(!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        printf("Failed to allocate host memory\n");
        return -1;
    }

    // Initialize matrices
    init(M * K, h_A);
    init(K * N, h_B);
    init(M * N, h_C_cpu);  // Initialize C for CPU computation
    
    // Copy C to GPU version for consistent initial values
    memcpy(h_C_gpu, h_C_cpu, size_C);

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_A, size_A);
    if(err != cudaSuccess) {
        printf("Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_B, size_B);
    if(err != cudaSuccess) {
        printf("Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_C, size_C);
    if(err != cudaSuccess) {
        printf("Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_gpu, size_C, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    // const int blockSize = (BM_T / TM_T) * BN_T;  // = (64/8)*64 = 512
    const int blockSize = (BM_T * BN_T) / (TM_T * TN_T);
    dim3 blockDim(blockSize);
    dim3 gridDim(CEIL_DIV(N, BN_T), CEIL_DIV(M, BM_T));  // Swapped dimensions
    
    printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d)\n", blockDim.x);

    // Launch kernel
    printf("Launching GPU kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sgemm2DBlocktiling<BM_T, BN_T, BK_T, TM_T, TN_T><<<gridDim, blockDim>>>(
        M, N, K, d_A, d_B, d_C, alpha, beta
    );
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // Compute CPU reference
    printf("Computing CPU reference...\n");
    clock_t cpu_start = clock();
    sgemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_C_cpu);
    clock_t cpu_end = clock();
    float cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;

    // Verify results
    printf("Verifying results...\n");
    bool correct = verify_result(h_C_cpu, h_C_gpu, M * N);
    
    if(correct) {
        printf("✓ Results match!\n");
        printf("GPU time: %.2f ms\n", gpu_time);
        printf("CPU time: %.2f ms\n", cpu_time);
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
        
        // Calculate GFLOPS
        float gflops = (2.0f * M * N * K) / (gpu_time * 1e6);
        printf("GPU Performance: %.2f GFLOPS\n", gflops);
    } else {
        printf("✗ Results do not match!\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}