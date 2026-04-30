#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include <omp.h>

void matmul_forward_cpu(float *out, const float *inp, const float *weight, const float *bias,
                        int B, int T, int C, int OC)
{

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *out_bt = out + b * T * OC + t * OC;
            const float *inp_bc = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float *wrow = weight + o * C;
                for (int c = 0; c < C; c++)
                {
                    val += inp_bc[c] * wrow[c];
                }
                out_bt[o] = val;
            }
        }
    }
}

__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp,
                                       const float* weight,
                                       const float* bias,
                                       int BT, int C, int OC){

            int bt = blockIdx.x * blockDim.x + threadIdx.x;
            int oc = blockIdx.y * blockDim.y + threadIdx.y;
            if(bt < BT && oc < OC){
                float val = (bias != NULL) ? bias[oc] : 0.0f;
                const float* wrow = weight + oc * C;
                const float* inp_bt = inp + bt * C;
                for(int c = 0; c < C; c++){
                    val += inp[bt * C + c] * wrow[oc * C + c];
                }
                out[bt * OC + oc] = val;
            }

}


__global__ void add_bias(float* out, const float* bias, int B, int T, int OC){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=idx; i<B*T*OC; i+=stride){
        int col = i%OC;
        out[i] += bias[col];
    }
}


__device__ float4 ld_vec(const float* address){
    return *reinterpret_cast<const float4*>(address);
}

__device__ float st_vec(float* address, float4 val){
    *reinterpret_cast<float4*>(address) = val;
}