#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

__global__ void matrixMultiply(float *a, float *b, float *c, int N, int K, int M){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int i=0; i < K; i++){
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}


int main(){
    int M = 1024, K = 1024, N = 1024;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(M * K * sizeof(float));
    b = (float*)malloc(K * N * sizeof(float));
    c = (float*)malloc(M * N * sizeof(float));

    for(int i=0; i < M*K; i++) a[i] = 1.0f;
    for(int i=0; i < K*N; i++) b[i] = 1.0f;

    cudaMalloc(&d_a, M*K*sizeof(float));
    cudaMalloc(&d_b, K*N*sizeof(float));
    cudaMalloc(&d_c, M*N*sizeof(float));

    cudaMemcpy(d_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, N, K, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for matrix multiplication: %f ms\n", milliseconds);

    cudaMemcpy(c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Result of matrix multiplication:\n");
    // for(int i=0; i < M; i++){
    //     for(int j=0; j < N; j++){
    //         printf("%.2f ", c[i*N + j]);
    //     }
    //     printf("\n");
    // }

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}