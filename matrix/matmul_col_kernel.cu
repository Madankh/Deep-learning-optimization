#include<stdio.h>
#include<cuda_runtime.h>

__global__ void matmul_col_kernel(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K)
{
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   if(col < N){
     for(int row = 0; row < M; row++){
        float tmp = 0.0f;
        for(int k=0; k < K; k++){
            tmp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = tmp;
     }
   }

}