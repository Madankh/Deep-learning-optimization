namespace M1{
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    namespace cde = cuda::device::experimental;

    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x){return (((x) & 0x3FFFF)>>0x4);}

    __device__ uint64_t make_smem_desc(bf16* ptr){
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        uint32_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(addr);
        desc |= matrix_descriptor_encode((uint64_t)16)<<16;
        desc |= matrix_descriptor_encode((uint64_t)1024)<<32;
        desc |= 1llu<<62;
        return desc
    }

    __device__ void warpgroup_arrive(){
        asm volatile("wgmma.fence.sync.aligned;\n"::: "memory");
    }

    __device__ void warpgroup_commit_batch(){
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }

    template<int N>
    __device__ void warpgroup_wait(){
        static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0,7]");
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }

    template <int BlockMajorSize, int BlockMinorSize>
    void create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width) {
        void* gmem_address = (void*)gmem_ptr;
        uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};
        uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0, 0, 0};
        uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
        uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
    
        CUresult result = cuTensorMapEncodeTiled(
            tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
            gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
        assert(result == CUDA_SUCCESS);
    }
}
