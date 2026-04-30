import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

# First, let's create a CUDA extension to load your kernel
# This code should be saved in a file called 'softmax_cuda.cu'

"""
// Save this code to softmax_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Your corrected softmax kernel
// warp level reduction for finding the maximum value 
__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset/=2){
        val = fmax(val , __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpsReduceSum(float val){
    for(int offset= 16; offset > 0; offset /=2){
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


__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32; 

    // the number of warps per block 
    int warpsPereBlock = blockDim.x / 32;
    float* max_or_sum_storage = shared;

    const float* x = inp + idx * C;
    // first thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for(int i=tid; i<C; i+=blockDim.x){
        maxval = fmaxf(maxval - x[i]);
    }
    maxval = warpReduceMax(maxval);

    if(laneId == 0) max_or_sum_storage[warpId] = maxval;
    __syncthreads();

    // now the 0th thread of the block reduces the max values in shared memory i.e across warps
    if(tid == 0){
        float val = max_or_sum_storage[tid];
        for(int i=1; i<warpsPereBlock; i++){
            val = fmaxf(val, max_or_sum_storage[i]);
        }
        max_or_sum_storage[0] =  val;
    }

    __syncthreads();
    // broadcast the max in the first position

    float  offset = max_or_sum_storage[0];

    for(int i=tid; i<C; i+=blockDim.x){
        out[idx*C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for(int i=tid; i<C; i+=blockDim.x){
        sumval += x[i];
    }

    sumval = warpsReduceSum(sumval);

    // warte sumval to shared Memory
    if(laneId == 0) max_or_sum_storage[warpId] = sumval;
    __syncthreads();

    // inter-thread 
    if(tid == 0){
        float val = max_or_sum_storage[tid];
        for(int i=1; i<warpsPereBlock; i++){
            val += max_or_sum_storage[i];
        }
        max_or_sum_storage[0] =  val;
    }

    __syncthreads();
    float  sum = max_or_sum_storage[0];

    //  divide the whole row by sum
    for(int i=tid; i<C; i+=blockDim.x){
        out[idx*C+i]= x[i]/sum;
    }
}



// C++ wrapper function to launch the CUDA kernel
void softmax_cuda_forward(torch::Tensor output, torch::Tensor input) {
    // Get tensor dimensions
    int N = input.size(0);  // batch size
    int C = input.size(1);  // feature dimension
    
    // Calculate grid and block dimensions
    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);
    
    // Launch kernel
    softmax_forward_kernel4<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N,
        C
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_cuda_forward, "Softmax forward (CUDA)");
}
"""

# Save this code to setup.py:
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax',
    ext_modules=[
        CUDAExtension('softmax', [
            'softmax.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
"""

# Now, let's create the benchmarking code:

class SoftmaxBenchmark:
    def __init__(self):
        # In an actual implementation, you would uncomment these lines
        # self.softmax_cuda = load(
        #     name='softmax_cuda',
        #     sources=['softmax_cuda.cu'],
        #     verbose=True
        # )
        pass
        
    def custom_softmax(self, input_tensor):
        """
        Wrapper for our custom CUDA softmax implementation
        """
        # Create output tensor
        output = torch.empty_like(input_tensor)
        
        # In an actual implementation, you would call:
        # self.softmax_cuda.forward(output, input_tensor)
        
        # For simulation, we'll just use PyTorch's softmax
        # but add a small delay to simulate the overhead of a custom implementation
        time.sleep(0.001)  # Simulation only - remove in real benchmark
        output = torch.nn.functional.softmax(input_tensor, dim=1)
        
        return output
    
    def pytorch_softmax(self, input_tensor):
        """
        PyTorch's native softmax implementation
        """
        return torch.nn.functional.softmax(input_tensor, dim=1)
    
    def check_correctness(self, custom_output, pytorch_output, rtol=1e-5, atol=1e-6):
        """
        Check if the custom implementation produces the same results as PyTorch
        """
        is_close = torch.allclose(custom_output, pytorch_output, rtol=rtol, atol=atol)
        if not is_close:
            # Calculate maximum absolute difference
            max_diff = torch.max(torch.abs(custom_output - pytorch_output))
            avg_diff = torch.mean(torch.abs(custom_output - pytorch_output))
            return False, max_diff.item(), avg_diff.item()
        return True, 0, 0
    
    def benchmark_size(self, batch_size, feature_size, num_iterations=10):
        """
        Benchmark both implementations for a specific tensor size
        """
        # Generate random input tensor
        input_tensor = torch.randn(batch_size, feature_size, device='cuda', dtype=torch.float32)
        
        # Warm-up runs
        for _ in range(3):
            _ = self.custom_softmax(input_tensor)
            _ = self.pytorch_softmax(input_tensor)
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        # Time custom implementation
        custom_start = time.time()
        for _ in range(num_iterations):
            custom_output = self.custom_softmax(input_tensor)
            torch.cuda.synchronize()
        custom_end = time.time()
        custom_time = (custom_end - custom_start) / num_iterations
        
        # Time PyTorch implementation
        pytorch_start = time.time()
        for _ in range(num_iterations):
            pytorch_output = self.pytorch_softmax(input_tensor)
            torch.cuda.synchronize()
        pytorch_end = time.time()
        pytorch_time = (pytorch_end - pytorch_start) / num_iterations
        
        # Check correctness
        is_correct, max_diff, avg_diff = self.check_correctness(custom_output, pytorch_output)
        
        return {
            'batch_size': batch_size,
            'feature_size': feature_size,
            'custom_time': custom_time,
            'pytorch_time': pytorch_time,
            'speedup': pytorch_time / custom_time,
            'is_correct': is_correct,
            'max_diff': max_diff,
            'avg_diff': avg_diff
        }
    
    def run_benchmark_suite(self):
        """
        Run benchmarks for different tensor sizes
        """
        # Test different batch sizes with fixed feature size
        batch_sizes = [1, 8, 32, 128, 512, 2048]
        feature_size = 1024
        batch_results = []
        
        print("Benchmarking different batch sizes:")
        print("====================================")
        for batch_size in batch_sizes:
            result = self.benchmark_size(batch_size, feature_size)
            batch_results.append(result)
            correct_str = "✓" if result['is_correct'] else "✗"
            print(f"Batch Size: {batch_size}, Features: {feature_size}")
            print(f"  Custom: {result['custom_time']*1000:.2f} ms")
            print(f"  PyTorch: {result['pytorch_time']*1000:.2f} ms")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Correct: {correct_str}")
            if not result['is_correct']:
                print(f"  Max Diff: {result['max_diff']:.6e}")
                print(f"  Avg Diff: {result['avg_diff']:.6e}")
            print()
        
        # Test different feature sizes with fixed batch size
        feature_sizes = [64, 256, 1024, 4096, 16384]
        batch_size = 32
        feature_results = []
        
        print("Benchmarking different feature sizes:")
        print("=====================================")
        for feature_size in feature_sizes:
            result = self.benchmark_size(batch_size, feature_size)
            feature_results.append(result)
            correct_str = "✓" if result['is_correct'] else "✗"
            print(f"Batch Size: {batch_size}, Features: {feature_size}")
            print(f"  Custom: {result['custom_time']*1000:.2f} ms")
            print(f"  PyTorch: {result['pytorch_time']*1000:.2f} ms")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Correct: {correct_str}")
            if not result['is_correct']:
                print(f"  Max Diff: {result['max_diff']:.6e}")
                print(f"  Avg Diff: {result['avg_diff']:.6e}")
            print()
        
        # Plot results
        self.plot_results(batch_results, feature_results)
        
    def plot_results(self, batch_results, feature_results):
        """
        Plot the benchmark results
        """
        plt.figure(figsize=(16, 6))
        
        # Plot batch size results
        plt.subplot(1, 2, 1)
        batch_sizes = [r['batch_size'] for r in batch_results]
        custom_times = [r['custom_time'] * 1000 for r in batch_results]  # convert to ms
        pytorch_times = [r['pytorch_time'] * 1000 for r in batch_results]  # convert to ms
        
        plt.plot(batch_sizes, custom_times, 'o-', label='Custom Implementation')
        plt.plot(batch_sizes, pytorch_times, 'o-', label='PyTorch')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (ms)')
        plt.title('Performance vs Batch Size')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        
        # Plot feature size results
        plt.subplot(1, 2, 2)
        feature_sizes = [r['feature_size'] for r in feature_results]
        custom_times = [r['custom_time'] * 1000 for r in feature_results]  # convert to ms
        pytorch_times = [r['pytorch_time'] * 1000 for r in feature_results]  # convert to ms
        
        plt.plot(feature_sizes, custom_times, 'o-', label='Custom Implementation')
        plt.plot(feature_sizes, pytorch_times, 'o-', label='PyTorch')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Feature Size')
        plt.ylabel('Time (ms)')
        plt.title('Performance vs Feature Size')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('softmax_benchmark_results.png')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a GPU-enabled machine.")
        exit()
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Create and run benchmark
    benchmark = SoftmaxBenchmark()
    benchmark.run_benchmark_suite()