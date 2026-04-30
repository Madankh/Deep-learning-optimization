#include<stdio>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<time.h>

#define M 512
#define K 128
#define N 512

__global__ void matrixmul_gpu(float* A, float* B, float* C, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < n){
        float sum = 0.0f;
        for(int i=0; i < k; i++){
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
    
}
void matrixmul_cpu(float* A, float* B, float* C, int m, int k, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0f;
            for(int p=0; p<k; p++){
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void init_vector(float* vec, int n){
    srand(time(NULL));
    for(int i=0; i<n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}


double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_A, *h_B, *h_C_cpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    float* h_C_gpu = (float*)malloc(size_C);
    // allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);

    // init matrices
    init_vector(h_A, M * K);
    init_vector(h_B, K * N);
    // allocate device memory    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    
    
    return 0;
}