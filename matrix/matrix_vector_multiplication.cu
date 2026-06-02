#include<stdio.h>
#include<cuda_runtime.h>

__global__ void matrix_vector_multiply_kernel(float* B, float* C, float* A, int M, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M){
        float tmp = 0.0f;
        for(int i=0; i < N; i++){
            tmp += B[row * N + i] * C[i];
        }
        A[row] = tmp;
    }
}