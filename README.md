# CUDA ML Algorithms Implementation 🚀

A collection of fundamental machine learning algorithms implemented in CUDA from scratch, focusing on performance optimization and educational value. 
My GPU is GTX 1650 4G RAM

## 📋 Overview

This repository contains CUDA implementations of core ML operations commonly used in deep learning frameworks. Each implementation emphasizes:
- **Performance**: Optimized memory access patterns and compute utilization
- **Educational Value**: Clear, well-commented code for learning CUDA programming
- **Correctness**: CPU reference implementations for verification

## 🎯 Implemented Algorithms

### ✅ Currently Available

| Algorithm | Status | Performance | Description | Speedup
|-----------|---------|-------------|-------------|
| **SGEMM (Matrix Multiplication)** | ✅ Complete | 670.74 GFLOPS | 1D block tiling optimization | 1061.85x
| **SGEMM (Matrix Multiplication)** | ✅ Complete | 1190.30 GFLOPS | 2D block tiling optimization | 1780.25x
| **LayerNorm** | ✅ Complete | - | Layer normalization for transformers |
| **Softmax** | ✅ Complete | - | Numerically stable softmax implementation |

### 🚧 Planned Implementations

| Algorithm | Priority | Use Case |
|-----------|----------|----------|
| **Max Pooling** | High | CNN downsampling |
| **Attention Mechanism** | High | Transformer core operation |
| **Convolution 2D** | High | CNN feature extraction |
| **ReLU & Derivatives** | Medium | Activation functions |
| **Batch Normalization** | Medium | Training stabilization |
| **Cross Entropy Loss** | Medium | Classification training |
| **Adam Optimizer** | Low | Gradient-based optimization |

## 🏗️ Project Structure

```
cuda-ml-algorithms/
├── src/
│   ├── matrix/
│   │   ├── sgemm_basic.cu          # Basic matrix multiplication
│   │   └── sgemm_1d_tiling.cu      # Optimized 1D block tiling
│   ├── normalization/
│   │   └── layernorm.cu            # Layer normalization
│   ├── activation/
│   │   └── softmax.cu              # Softmax activation
│   └── utils/
│       ├── common.h                # Common utilities and macros
│       └── verification.cu         # CPU reference implementations
├── examples/
│   └── benchmarks/                 # Performance benchmarking scripts
├── tests/
│   └── unit_tests/                 # Unit tests for each algorithm
└── README.md
```

## 🚀 Performance Results

### Matrix Multiplication (1024 * 1024 * 1024)
- **GPU Time**: 3.20 ms
- **Performance**: 670.74 GFLOPS
- **Speedup vs CPU**: 1061.85x
- **Architecture**: Optimized 1D block tiling with shared memory

### Matrix Multiplication (1024 * 1024 * 1024)
- **GPU Time**: 1.80 ms
- **Performance**: 1190.30 GFLOPS
- **Speedup vs CPU**: 1780.25x
- **Architecture**: Optimized 2D block tiling with shared memory

### Memory Optimization Techniques Used
- **Shared Memory Tiling**: Reduces global memory accesses
- **Coalesced Memory Access**: Maximizes memory bandwidth utilization
- **Thread-level Parallelism**: Each thread computes multiple output elements
- **Register Blocking**: Minimizes shared memory bank conflicts

## 🛠️ Building and Running

### Prerequisites
- CUDA Toolkit (≥11.0)
- GCC/G++ compiler
- CMake (optional, for build automation)

### Compilation
```bash
# Individual algorithms
nvcc -o sgemm src/matrix/sgemm_1d_tiling.cu -O3
nvcc -o layernorm src/normalization/layernorm.cu -O3
nvcc -o softmax src/activation/softmax.cu -O3

# Suppress architecture warnings (optional)
nvcc -Wno-deprecated-gpu-targets -o program file.cu
```

### Running
```bash
./1dBlocktiling          # Matrix multiplication benchmark
./sgemm2Dblock   # Matrix multiplication benchmark
./layernorm      # Layer normalization test
./softmax        # Softmax activation test
```

## 📊 Benchmarking

Each implementation includes:
- **Correctness Verification**: Results compared against CPU reference
- **Performance Metrics**: Timing, GFLOPS, memory bandwidth utilization
- **Scalability Testing**: Performance across different input sizes

Example output:
```
Grid dimensions: (16, 16)
Block dimensions: (512)
Launching GPU kernel...
Computing CPU reference...
Verifying results...
✓ Results match!
GPU time: 3.20 ms
CPU time: 3399.69 ms
Speedup: 1061.85x
GPU Performance: 670.74 GFLOPS


Grid dimensions: (8, 8)
Block dimensions: (256)
Launching GPU kernel...
Computing CPU reference...
Verifying results...
✓ Results match!
GPU time: 1.80 ms
CPU time: 3211.86 ms
Speedup: 1780.25x
GPU Performance: 1190.30 GFLOPS
```

## 🎓 Learning Resources

### CUDA Optimization Techniques Demonstrated
1. **Memory Hierarchy Utilization**
   - Shared memory for data reuse
   - Register blocking for compute intensity
   - Coalesced global memory access

2. **Thread Organization**
   - 1D/2D block tiling strategies
   - Warp-level optimizations
   - Occupancy considerations

3. **Algorithmic Optimizations**
   - Numerically stable implementations
   - Reduced precision where appropriate
   - Minimized divergent branching

## 🤝 Contributing

Contributions are welcome! Please focus on:
- **Performance**: New optimization techniques
- **Correctness**: Robust error handling and verification
- **Documentation**: Clear explanations of algorithms and optimizations
- **Testing**: Comprehensive unit and integration tests

### Development Guidelines
- Follow CUDA best practices for memory access patterns
- Include CPU reference implementations for verification
- Add performance benchmarks for new algorithms
- Maintain consistent code style and documentation

## 📝 TODO List

- [ ] Implement max pooling with different kernel sizes
- [ ] Add multi-head attention mechanism
- [ ] Optimize softmax for large vocabulary sizes  
- [ ] Implement 2D convolution with im2col
- [ ] Add FP16 precision support
- [ ] Create automated benchmarking suite
- [ ] Add Docker container for consistent environment

## 📜 License

MIT License - Feel free to use this code for learning and research purposes.

## 🙏 Acknowledgments

Inspired by high-performance computing principles and modern deep learning frameworks. Special thanks to the CUDA programming community for optimization techniques and best practices.

---

**Note**: This is an educational project focused on understanding CUDA programming and ML algorithm implementation. For production use, consider optimized libraries like cuBLAS, cuDNN, and Thrust.
