#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#define TILE_SIZE 16

__global__ void shared_matrixMul(float *a, float *b, float *c, int M, int K, int N){
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    
    float value = 0.0f;
    for(int tileIdx=0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; tileIdx++){
        
        tile_a[local_row][local_col] = (row < M && tileIdx * TILE_SIZE + local_col < K)? a[row * K + tileIdx * TILE_SIZE + local_col] : 0.0f;
        tile_b[local_row][local_col] = (col < N && tileIdx * TILE_SIZE + local_row < K) ? b[(tileIdx * TILE_SIZE + local_row) * N + col] : 0.0f;
 
        __syncthreads();
        for(int k=0; k < TILE_SIZE; k++){
            value += tile_a[local_row][k] * tile_b[k][local_col];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        c[row * N + col] = value;
    }
}


void initmaxtrix(float* A, int K,int N){
    for(int i=0; i < K * N; i++){
        A[i] = rand() / (float)RAND_MAX;
    }
}

int main(){
    int M = 1024, K = 512, N = 1024;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    a = (float*)malloc(M * K * sizeof(float));
    b = (float*)malloc(K * N * sizeof(float));
    c = (float*)malloc(M * N * sizeof(float));

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    initmaxtrix(a, M, K);
    initmaxtrix(b, K, N);

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) /  TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    shared_matrixMul<<<grid, block>>>(d_a, d_b, d_c, M, K, N);
    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}