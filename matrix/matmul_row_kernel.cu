#include<stdio.h>
#include<cuda_runtime.h>

__global__ void matrix_vector_multiply_kernel(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M){
        for(int col=0; col < N; col++){
            float tmp = 0.0f;
            for(int i=0; i < K; i++){
                tmp += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = tmp;
        }

    }

}