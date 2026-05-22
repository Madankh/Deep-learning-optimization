#include<stdio.h>
#include<algorithm>
#include<cuda_runtime.h>
#define BLOCK_SIZE 32

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha ,float *A, float *B,float beta, float *C){
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    int threadRow = threadIdx.x / BLOCKSIZE;
    int threadCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;
    float temp = 0.0;
    for(int bkIdx=0; bkIdx < K; bkIdx+=BLOCKSIZE){
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for(int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++){
            temp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }

    C[threadRow * N + threadCol] =
      alpha * temp + beta * C[threadRow * N +  threadCol];
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
    
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
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
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    
    printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d)\n", blockDim.x);

    // Launch kernel
    printf("Launching GPU kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sgemm_shared_mem_block<BLOCK_SIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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