#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void relu(float *out, float *inp, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        out[i] = fmaxf(inp[i], 0);
    }
}