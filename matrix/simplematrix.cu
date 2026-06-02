#include<stdio.h>
#include<cuda_runtime.h>

__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K){
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

int main(){
    const int M = 512;
    const int N = 512;
    const int K = 512;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for(int i=0; i < M*K; i++){
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for(int i=0; i < K*N; i++){
        h_B[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16,16);
    dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                          (M + threads_per_block.y - 1) / threads_per_block.y);
    
    matrix_multiply_kernel<<<number_of_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Optionally print some values from C to verify correctness
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[M-1][N-1] = %f\n", h_C[(M-1)*N + (N-1)]);

    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}