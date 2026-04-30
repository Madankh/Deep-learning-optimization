#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<time.h>
#define N 1000000
#define BLOCK_SIZE 256

__global__ void vectorAdd(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        c[i] = a[i] + b[i];
    }
}

void vectorAddCPU(float* a, float* b, float* c, int n){
    for(int i=0; i<n; i++){
        c[i] = a[i] + b[i];
    }
}

void init_vector(float* vec, int n){
    srand(time(NULL));
    for(int i=0; i < n; i++){
        vec[i] = rand() / (float)RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    float* h_c_gpu = (float*)malloc(size);

    // allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);

   //init vectors
   init_vector(h_a, N);
   init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching kernel with %d blocks and %d threads per block\n", numBlocks, BLOCK_SIZE);
    for (int i = 0; i < 10; i++){
        vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU version
    printf("Benchmarking CPU version...\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 10; i++){
        double start = get_time();
        vectorAddCPU(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total_time += (end - start);
    }
    double cpu_avg_time = cpu_total_time / 10.0;
    
    printf("Benchmarking GPU version...\n");
    double gpu_total_time = 0.0;
    for(int i=0; i<10; i++){
        double start = get_time();
        vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a,d_b,d_c,N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total_time += (end - start);
    }

    double gpu_avg_time = gpu_total_time / 10.0;
    printf("Average CPU time: %f seconds\n", cpu_avg_time);
    printf("Average GPU time: %f seconds\n", gpu_avg_time);
    printf("speedup: %f\n", cpu_avg_time / gpu_avg_time);
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i=0; i<N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}