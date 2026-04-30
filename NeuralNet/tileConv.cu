#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define FILTER_RADIUS 3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, 
     int width, int height){
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        // loading input tile
        __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
        if(row >= 0 && row < height && col >= 0 && col < width){
            N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
        }else{
            N_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // Calculating output elements
        int tileCol = threadIdx.x - FILTER_RADIUS;
        int tileRow = threadIdx.y - FILTER_RADIUS;

        if(col >= 0 && col < width && row >= 0 && row < height){
            if(tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM){
               float Pvalue = 0.0f;
               for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++){
                     for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++){
                            Pvalue += F_c[i + FILTER_RADIUS][j + FILTER_RADIUS] * N_s[threadIdx.y + i][threadIdx.x + j];
                     }
               }
               P[row * width + col] = Pvalue;
            }
        }
   }

int main(){

    return 0;
}