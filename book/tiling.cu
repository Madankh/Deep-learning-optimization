#include<stdio.h>
#include<cuda_runtime.h>
#include<math.h>
#include<stdlib.h>


__global__ void matrixMul(float *a, float *b, float *c, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
}