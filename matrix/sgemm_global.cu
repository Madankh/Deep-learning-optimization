#include<stdio.h>
#include<cuda_runtime.h>

/// this version is slower then top  version because of memory accessing pattern
__global__ void sgemm_global(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float tmp = 0.0f;
        for(int i=0; i < K; i++){
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = tmp;
    }
}

int main()
{
    int M = 2048;
    int K = 1048;
    int N = 2048;

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *a_h = (float*)malloc(size_A);
    float *b_h = (float*)malloc(size_B);
    float *c_h = (float*)malloc(size_C);

    float *a_d, *b_d, *c_d;

    cudaMalloc(&a_d, size_A);
    cudaMalloc(&b_d, size_B);
    cudaMalloc(&c_d, size_C);

    // init A
    for (int i = 0; i < M * K; i++)
        a_h[i] = (float)rand() / RAND_MAX;

    // init B
    for (int i = 0; i < K * N; i++)
        b_h[i] = (float)rand() / RAND_MAX;

    // init C (important)
    for (int i = 0; i < M * N; i++)
        c_h[i] = 0.0f;

    cudaMemcpy(a_d, a_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, size_C, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    sgemm_global_mem_coalesce<<<grid, block>>>(
        M, N, K,
        alpha,
        a_d,
        b_d,
        beta,
        c_d
    );

    cudaDeviceSynchronize();

    cudaMemcpy(c_h, c_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}