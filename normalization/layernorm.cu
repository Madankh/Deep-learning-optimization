#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<assert.h>


void layernorm(float *out, float *mean, float *rstd, const float *inp, const float *weight, const float *bias, int B, int T, int C){
    float eps = 1e-5f;
    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            const float *x = inp + b * T * C + t * C;
            float m = 0.0f;
            for(int i=0; i<C; i++){
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for(int i = 0; i < C; i++){
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sqrtf(v + eps);
            float *out_bs = out + b * T * C + t * C;
            for(int i=0; i<C; i++){
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_bs[i] = o;
            }

            mean[b * T + t] = m;
            rstd[b * T + t] = s;

        }
    }
}

__global__ void layernorm_kernel1(float *out, float *mean, float *rstd, const float *inp, const float *weight, float *bias, int N, int C){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;
    if(idx < N){
        const float *x = inp + idx * C;
        float m = 0.0;
        for(int i=0; i<C; i++){
            m += x[i];
        }
        m = m/C;

        float v = 0.0f;
        for(int i=0; i < C; i++){
            float rshift = (x[i] - m);
            v += rshift * rshift;
        }

        v = v/C;
        float s = 1.0f / (v + eps);
        float *out_idx = out + idx * C;

        for(int i=0; i < C; i++){
            float n = (s * (x[i] - m));
            float o = n * weight[i] + bias[i];
            out_idx[i] = o;
        }
        mean[idx] = m;
        rstd[idx] = s;
    }
}

__global__ void mean_kernel(float *mean, const float *inp, int N, int C, int block_size){
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    const float *x = inp + idx * C;
    float m = mean[idx];
    float sum = 0.0f;

    for(int i=tid; i < C; i+=block_size){
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();

    for(int stride = block_size/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] += shared[tid + stride]; 
        }
    }

    if(tid == 0){
        mean[idx] = shared[0]/C;
    }
}

__global__ void rstd_kernel(float *rstd, float *inp, float *mean, int N, int C, int block_size){
    extern __shared__ float shared[];
    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float *x = inp + idx * C;
    float m = mean[idx];
    float sum = 0;

    for(int i=tid; i < C; i += block_size){
        float xshift = x[i] - m;
        sum += xshift * xshift;
    }

    shared[tid] = sum;
    __syncthreads();

    for(int stride = block_size/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
    }
    if(tid == 0){
        rstd[idx] = 1.0f / sqrtf(shared[0]/C + 1e-5f);
    }
}

__global__ void normalization_kernel(float *out, const float* inp, float *mean, float *rstd,
const float *weight, const float *bias, int B, int T, int C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

__global__void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd , const float *inp , float* __restrict__ weights, float* __restrict__ bias, int N, int C){
    namespace corgroup = cooperative_groups();
    corgroup::thread_block block = corgroup::this_thread_block();
    corgroup:thread_block_tile<32> warp = corgroup::tiled_partition<32>(block);

    const int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    const float *x = inp + idx * C;
    if(idx >= N){
        return;
    }
    float sum = 0.0f;
    for(int i = warp.thread_rank(); i < C; i+=warp.size()){
        sum += x[i];
    }
    sum = corgroup::reduce(warp, sum, corgroup::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr){
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for(int i = warp.thread_rank(); i < C; i+=warp.size()){
        float xshift = x[i] - m;
        sum += xshift * xshift;
    }

    sum = corgroup::reduce(warp, sum, corgroup::plus<float>{});
    // float s = 1.0f / sqrt(sum/C + 1e-5); -- this is slower 
    float s = rsqrtf(sum / C + 1e-5); // this is faster but lower in precise

    if(warp.thread_rank() == 0 && mean != nullptr){
        __stcs(rstd + idx, s);
    }

    float* o = out + idx * C;
    for(int c = warp.thread_rank(); c < C; i+=warp.size()){
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weights[c] + bias[c]);
    }

}

__global__ void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd, float* __restrict__ inp, float* __restrict__ weights, float* __restrict__ bias, int N, int C){
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    float eps = 1e-5;
    float* x = inp + idx * C;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for(int i=warp.thread_rank(); i < C; i+=warp.size()){
        float xi = x[i];
        sum += xi;
        sum += xi * xi;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>{}); // sum(x)
    sum2 = cg::reduce(warp, sum2. cg::plus<float>{}); // sum(x**2)
    sum /= C;
    sum2 /= C;

    float m = sum;
    float var = sum2 - (sum * sum);
    float s = rsqrtf(var + 1e-5);

    // store the mean, no need to cache it 
    if(warp.thread_rank() == 0 && mean != nullptr){
        __stcs(mean + idx, m);
    }

    if(warp.thread_rank() == 0 && rstd != nullptr){
        __stcs(rstd + idx, s);
    }

    // final normalization and scalling by weights and bias
    float* o = out + idx * C;
    for(int i=warp.thread_rank(); i < C; i+=warp.size()){
        float n = s * (__ldcs(x+i) - m);
        __stcs(o + i, n * weights[i] + bias[i])
    }
}

__global__ void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd, float* __restrict__ inp, float* __restrict__ weights, float* __restrict__ bias, int N, int C){
    
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float shared_sum[32]; 
    __shared__ float shared_sum2[32]; // warps will be writting into shared memory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    float idx = blockIdx.x; // one block per row 
    float* x = inp + idx * C;

    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;

    for(int i=threadIdx.x; i < C; i += blockDim.x){
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    
    // warp level reductioon
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{}); // sum(x)
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{});
    shared_sum[warp_idx] = warp_sum;
    shared_sum2[warp_idx] = warp_sum2;
    __syncthreads();

    warp_sum = (lane_idx < num_warps) ? shared_sum[lane_idx] : 0.0f;
    warp_sum2 = (lane_idx < num_warps) ? shared_sum2[lane_idx] : 0.0f;

    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{});
    block_sum/=C;
    block_sum2/=C;
    float m = block_sum;
    float var = block_sum2 - (m * m);
    float s = rsqrtf(var + 1e-5f);

    if(threadIdx.x == 0 && mean != nullptr){
        __stcs(mean + idx , m);
    }

    if(threadIdx.x == 0 && rstd != nullptr){
        __stcs(rstd + idx, s);
    }

    float* o = out + idx * C;
    for(int i=threadIdx.x; i < C; i+=blockDim.x){
        float n = s * (__ldcs(x + i) - m);
        __stcs(o+i, n * weights[i] + bias[i]);
    }

}