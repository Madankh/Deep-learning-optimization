#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void mean_filter_kernel(unsigned char* input, unsigned char* output, int width, int height, int radius){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = threadIdx.z;

    int baseOffset = channel * width * height;
    if(col < width && row < height){
      int pixVal = 0;
      int pixels = 0;
      
      for(int blurRow = -radius; blurRow <= radius; blurRow++){
          for(int blurCol = -radius; blurCol <= radius; blurCol++){
              int curRow = row + blurRow;
              int curCol = col + blurCol;
  
              if(curRow >= 0 && curRow < height && curCol >=0 && curCol < width){
                  pixVal += input[baseOffset + curRow * width + curCol];
                  pixels++;
              }
          }
      }
      output[baseOffset + row * width + col] = pixVal / pixels;
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

torch::Tensor mean_filter(torch::Tensor image, int radius){
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels  = image.size(0);
    const auto height = image.size(1);
    const auto width  = image.size(2);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16,16,channels);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    mean_filter_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}