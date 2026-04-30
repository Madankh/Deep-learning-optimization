#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// CPU code reference
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward_cpu(float* out, const float* inp, int N){
    for(int i=0; i<N; i++){
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }

}

__global__ void gelu_forward_kernel1(float* out, const float* inp, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// elementwise ops on GPU

