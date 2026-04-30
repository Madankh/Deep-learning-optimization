#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Warp reduction kernel
template<typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val, int thread_group_width=32) {
    #pragma unroll
    for(int i=0; i<NUM; i++) {
    #pragma unroll
        for(int mask = thread_group_width / 2; mask > 0; mask >>=1) {
            val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
        }
    }
    return val[0];  // Return actual max value instead of 0
}

void softmax_forward_cpu(float *out, float *inp, int N, int C){
    for(int i=0; i<N; i++){
        float *inp_row = inp + i * C;
        float *out_row = out + i * C;
        float maxval = -INFINITY;
        for(int j=0; j<C; j++){
            if(inp_row[j] > maxval){
                maxval = inp_row[j];
            }
        }
        double sum = {0};
        for(int j=0; j < C; j++){
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.0f / (float)sum;
        for(int j=0; j<C; j++){
            out_row[j] *= norm;
        }
    }
}

// Online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C){
    // inp is (N,C)
    // out is (N,C), each row of inp will get softmax
    for(int i=0; i<N; i++){
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;
        float maxval = -INFINITY;
        float sum = 0.0f;
        for(int j=0; j<C; j++){
            float maxval_prev = maxval;
            if(inp_row[j] > maxval){
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            }else{
                sum += expf(inp_row[j] - maxval);
            }
        } 

        for(int j=0; j<C; j++){
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

void softmax_forward_online_cpu(float *out, float *inp, int N, int C){
    for(int i=0; i<N; i++){
        float *inp_row = inp + i * C;
        float *out_row = out + i * C;
        float sum = 0.0f;
        float maxVal = -INFINITY;
        for(int j=0; j < C; j++){
            float maxVal_previos = maxVal;
            if(inp_row[j] > maxVal){
                maxVal = inp_row[j];
                sum = sum * expf(maxVal_previos - maxVal) + expf(inp_row[j] - maxVal);
            }else{
                sum += expf(inp_row[j] - maxVal);
            }
        }

        for(int j=0; j<C; j++){
            out_row[j] = expf(inp_row[i] - maxVal) / sum;
        }
    }
}


// GPU kernels
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C){
    // inp is (N , C)
    // out is (N , C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;
        float maxval  = -INFINITY;
        for(int j=0; j<C; j++){
            if(inp_row[j] > maxval){
                maxval = inp_row[j];
            }
        }

        double sum = 0.0;
        for(int j = 0; j < C; j++){
            out_row[j]= expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        

        for(int j=0; j<C; j++){
            out_row[j] /= (float)sum;
        }
    }
    
}
__global__ void softmax_forward_kernel1(float *out, float *inp, int N, int C){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
    float *inp_row = inp + idx * C;
    float *out_row = out + idx * C;

    float maxVal = -INFINITY;

    for(int i=0; i < C; i++){
        if(inp_row[i] > maxVal){
            maxVal = inp_row[i];
        }
    }

    float sum = 0.0f;
    for(int i = 0; i < C; i++){
        out_row[i] = expf(inp_row[i] - maxVal);
        sum += out_row[i];
    }

    float norm = 1.0f / (float)sum;
    for(int i=0; i < C; i++){
        out_row[i] = out_row[i] * norm;
    }
        
  }
}

__global__ void softmax_forward_kernel2(float *out, float *inp, int N, int C){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    float maxVal = 0.0f;
    float *x = inp + idx * C;
    for(int i=tid; i <  C; i += block_size){
        maxVal = fmaxf(x[i] , maxVal);   
    }

    shared[tid] = maxVal;
    for(int stride=block_size/2; stride >= 1; stride / 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[stride + tid]);
        }
    }
    float offset = shared[0];

    for(int i=tid; i < C; i+=block_size){
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    x = out + idx * C;
    float sum = 0.0f;
    for(int i = tid; i < C; i += block_size){
        sum += x[i];
    }
    shared[tid] = sum;

    for(int stride=block_size/2; stride >= 1; stride / 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
    }

    __syncthreads();
    float sum = shared[0];
    for(int i=tid; i < C; i+=block_size){
        out[idx * C + i] = x[i]/sum;
    }
    
}

//warp level reducntion for finding the maximum value 
__device__ float warpReduceMax(int val){
    for(int offset = 16; offset > 0; offset/=2){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp level reducetion sum
__device__ float warpsReduceSum(float val){
    for(int offset=16; offset > 0; offset/=2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + idx * C;

    // Thread coarsening and within=warp reduction for maxval
    float maxval = -INFINITY;
    for(int i=tid; i<C; i+=blockDim.x){
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);

    // Broadcast maxval within the warp
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);

    // Compute expf and write thre result to global memory
    for(int i=tid; i<C; i+=blockDim.x){
        out[idx * C + i] = expf(x[i] - offset);
    }

    // Thread coarsening and within-warp reduction for sumval
    x = out + idx * C;
    float sumval = 0.0f;
    for(int i=tid; i<C; i+=blockDim.x){
        sumval+=x[i];
    }

    sumval = warpsReduceSum(sumval);

    // divide the input values by the sum
    for(int i=tid; i<C; i+=blockDim.x){
        out[idx * C + i] = x[i]/sumval;
    }
    
}

// first find global max
// compute exp(x[i] - offset(which is maxval)) ) for numerical stablility
// sum the exp values using warps redunction warpsReduceSum
// normaalize by divide each element by the total sum

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;
    // shared[] must be allocated to have warpsPerBlock elements
    float* max_or_sum_storage = shared;

    const float* x = inp + idx * C;
    float maxval = -INFINITY;

    for(int i=tid; i < C; i+=blockDim.x){
        maxval = fmaxf(maxval, x[i]);
    }

    maxval = warpReduceMax(maxval);


    if(laneIdx == 0) max_or_sum_storage[warpIdx] = maxval;
    __syncthreads();

    if(tid == 0){
        float val = max_or_sum_storage[0];
        for(int i=1; i<warpsPerBlock; i++){
            val = fmaxf(val, max_or_sum_storage[i]);
        }
        max_or_sum_storage[0] = val;
    }
    __syncthreads();

    float offset = max_or_sum_storage[0];
    for(int i=tid; i < C; i+=blockDim.x){
        out[idx * C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for(int i=tid; i < C; i+=blockDim.x){
        sumval += x[i];
    }
    sumval = warpsReduceSum(sumval);
    if(laneIdx == 0) max_or_sum_storage[warpIdx] = sumval;
    __syncthreads();

    if(tid == 0){
        float val = max_or_sum_storage[0];
        for(int i=1; i < warpsPerBlock; i++){
            val += max_or_sum_storage[i];
        }
        max_or_sum_storage[0] = val;
    }
    __syncthreads();

    float sum = max_or_sum_storage[0];
    for(int i=tid; i < C; i+=blockDim.x){
        out[idx * C + i]  = x[i] / sum;
    }
}



