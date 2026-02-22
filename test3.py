
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

import struct


cuda_kernel = """
// Expected local size: (8, 1, 1)
#define VKDISPATCH_EXPECTED_LOCAL_SIZE_X 8
#define VKDISPATCH_EXPECTED_LOCAL_SIZE_Y 1
#define VKDISPATCH_EXPECTED_LOCAL_SIZE_Z 1

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define VKDISPATCH_ENABLE_SUBGROUP_OPS 1
#define VKDISPATCH_ENABLE_PRINTF 1

__device__ __forceinline__ float2 vkdispatch_make_float2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ float2 vkdispatch_make_float2(float x) { return make_float2(x, x); }
template <typename TVec> __device__ __forceinline__ float2 vkdispatch_make_float2(TVec v) { return make_float2((float)v.x, (float)v.y); }

__device__ __forceinline__ uint3 vkdispatch_local_invocation_id() {
    return make_uint3((unsigned int)threadIdx.x, (unsigned int)threadIdx.y, (unsigned int)threadIdx.z);
}

__device__ __forceinline__ uint3 vkdispatch_workgroup_id() {
    return make_uint3((unsigned int)blockIdx.x, (unsigned int)blockIdx.y, (unsigned int)blockIdx.z);
}

__device__ __forceinline__ unsigned int vkdispatch_local_invocation_index() {
    return (unsigned int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
}

__shared__ float2 sdata[68];

struct UniformObjectBuffer {
    uint4 exec_count;
    int4 sdata_shape;
    int4 buf1_shape;
};
struct Buffer1 { float2* data; };

extern "C" __global__ void vkdispatch_main(const UniformObjectBuffer* UBO_ptr, Buffer1 buf1) {
    const UniformObjectBuffer& UBO = *UBO_ptr;
    unsigned int workgroup_index = ((unsigned int)(vkdispatch_workgroup_id().x));
    unsigned int tid = vkdispatch_local_invocation_id().x;
    unsigned int input_batch_offset = ((unsigned int)(0));
    unsigned int output_batch_offset = ((unsigned int)(0));
    float2 omega_register = vkdispatch_make_float2(0);
    unsigned int subsequence_offset = ((unsigned int)(0));
    unsigned int io_index = ((unsigned int)(0));
    unsigned int io_index_2 = ((unsigned int)(0));
    float2 radix_register_0 = vkdispatch_make_float2(0);
    float2 radix_register_1 = vkdispatch_make_float2(0);
    float2 fft_reg_0 = vkdispatch_make_float2(0);
    float2 fft_reg_1 = vkdispatch_make_float2(0);
    float2 fft_reg_2 = vkdispatch_make_float2(0);
    float2 fft_reg_3 = vkdispatch_make_float2(0);
    float2 fft_reg_4 = vkdispatch_make_float2(0);
    float2 fft_reg_5 = vkdispatch_make_float2(0);
    float2 fft_reg_6 = vkdispatch_make_float2(0);
    float2 fft_reg_7 = vkdispatch_make_float2(0);
    
    /* Reading input samples from global memory into FFT registers. */
    input_batch_offset = ((workgroup_index + vkdispatch_local_invocation_id().y) << 6);
    io_index = (tid + input_batch_offset);
    fft_reg_0 = buf1.data[io_index];
    io_index = ((tid + 8) + input_batch_offset);
    fft_reg_1 = buf1.data[io_index];
    io_index = ((tid + 16) + input_batch_offset);
    fft_reg_2 = buf1.data[io_index];
    io_index = ((tid + 24) + input_batch_offset);
    fft_reg_3 = buf1.data[io_index];
    io_index = ((tid + 32) + input_batch_offset);
    fft_reg_4 = buf1.data[io_index];
    io_index = ((tid + 40) + input_batch_offset);
    fft_reg_5 = buf1.data[io_index];
    io_index = ((tid + 48) + input_batch_offset);
    fft_reg_6 = buf1.data[io_index];
    io_index = ((tid + 56) + input_batch_offset);
    fft_reg_7 = buf1.data[io_index];
    
    /*
     * FFT stage 1/2.
     * Prime group (2, 2, 2): execute 1 radix-8 sub-FFTs per invocation.
     * Register-group coverage this stage: 8.
     */
    
    /*
     * Starting mixed-radix FFT decomposition for this invocation on 8 register samples.
     * Radix factorization sequence: (2, 2, 2).
     * At each level: partition lanes into stage-local sub-sequences, apply twiddles,
     * run radix-P butterflies, then reassemble in stride-consistent order for downstream stages.
     */
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_4;
    fft_reg_4 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_5;
    fft_reg_5 = (fft_reg_1 - radix_register_0);
    fft_reg_1 = (fft_reg_1 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_6;
    fft_reg_6 = (fft_reg_2 - radix_register_0);
    fft_reg_2 = (fft_reg_2 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_3 - radix_register_0);
    fft_reg_3 = (fft_reg_3 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_2;
    fft_reg_2 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 4. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_6.x;
    fft_reg_6.x = fft_reg_6.y;
    fft_reg_6.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_6;
    fft_reg_6 = (fft_reg_4 - radix_register_0);
    fft_reg_4 = (fft_reg_4 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_3;
    fft_reg_3 = (fft_reg_1 - radix_register_0);
    fft_reg_1 = (fft_reg_1 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 4. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_7.x;
    fft_reg_7.x = fft_reg_7.y;
    fft_reg_7.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_5 - radix_register_0);
    fft_reg_5 = (fft_reg_5 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_1;
    fft_reg_1 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_5.x, 0.7071067811865476, ((-fft_reg_5.y) * -0.7071067811865475)), fma(fft_reg_5.x, -0.7071067811865475, (fft_reg_5.y * 0.7071067811865476)));
    fft_reg_5 = radix_register_0;
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_5;
    fft_reg_5 = (fft_reg_4 - radix_register_0);
    fft_reg_4 = (fft_reg_4 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 2.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_3.x;
    fft_reg_3.x = fft_reg_3.y;
    fft_reg_3.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_3;
    fft_reg_3 = (fft_reg_2 - radix_register_0);
    fft_reg_2 = (fft_reg_2 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 3.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_7.x, -0.7071067811865475, ((-fft_reg_7.y) * -0.7071067811865476)), fma(fft_reg_7.x, -0.7071067811865476, (fft_reg_7.y * -0.7071067811865475)));
    fft_reg_7 = radix_register_0;
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_6 - radix_register_0);
    fft_reg_6 = (fft_reg_6 + radix_register_0);
    
    /*
     * FFT stage 2/2.
     * Prime group (2, 2, 2): execute 1 radix-8 sub-FFTs per invocation.
     * Register-group coverage this stage: 8.
     */
    /* Register shuffle not possible, falling back to shared memory shuffle. */
    io_index = (tid * 8);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_0;
    io_index = (tid * 8 + 1);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_4;
    io_index = (tid * 8 + 2);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_2;
    io_index = (tid * 8 + 3);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_6;
    io_index = (tid * 8 + 4);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_1;
    io_index = (tid * 8 + 5);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_5;
    io_index = (tid * 8 + 6);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_3;
    io_index = (tid * 8 + 7);
    io_index = (io_index + (io_index >> 4));
    sdata[io_index] = fft_reg_7;
    __syncthreads();
    io_index = tid;
    io_index = (io_index + (io_index >> 4));
    fft_reg_0 = sdata[io_index];
    io_index = (tid + 8);
    io_index = (io_index + (io_index >> 4));
    fft_reg_4 = sdata[io_index];
    io_index = (tid + 16);
    io_index = (io_index + (io_index >> 4));
    fft_reg_2 = sdata[io_index];
    io_index = (tid + 24);
    io_index = (io_index + (io_index >> 4));
    fft_reg_6 = sdata[io_index];
    io_index = (tid + 32);
    io_index = (io_index + (io_index >> 4));
    fft_reg_1 = sdata[io_index];
    io_index = (tid + 40);
    io_index = (io_index + (io_index >> 4));
    fft_reg_5 = sdata[io_index];
    io_index = (tid + 48);
    io_index = (io_index + (io_index >> 4));
    fft_reg_3 = sdata[io_index];
    io_index = (tid + 56);
    io_index = (io_index + (io_index >> 4));
    fft_reg_7 = sdata[io_index];
    
    /*
     * Starting mixed-radix FFT decomposition for this invocation on 8 register samples.
     * Radix factorization sequence: (2, 2, 2).
     * At each level: partition lanes into stage-local sub-sequences, apply twiddles,
     * run radix-P butterflies, then reassemble in stride-consistent order for downstream stages.
     */
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 64. Twiddle index source: tid.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    omega_register.x = (tid * -0.09817477042468103);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_4.x, omega_register.x, ((-fft_reg_4.y) * omega_register.y)), fma(fft_reg_4.x, omega_register.y, (fft_reg_4.y * omega_register.x)));
    fft_reg_4 = radix_register_0;
    omega_register.x = (tid * -0.19634954084936207);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_2.x, omega_register.x, ((-fft_reg_2.y) * omega_register.y)), fma(fft_reg_2.x, omega_register.y, (fft_reg_2.y * omega_register.x)));
    fft_reg_2 = radix_register_0;
    omega_register.x = (tid * -0.2945243112740431);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_6.x, omega_register.x, ((-fft_reg_6.y) * omega_register.y)), fma(fft_reg_6.x, omega_register.y, (fft_reg_6.y * omega_register.x)));
    fft_reg_6 = radix_register_0;
    omega_register.x = (tid * -0.39269908169872414);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_1.x, omega_register.x, ((-fft_reg_1.y) * omega_register.y)), fma(fft_reg_1.x, omega_register.y, (fft_reg_1.y * omega_register.x)));
    fft_reg_1 = radix_register_0;
    omega_register.x = (tid * -0.4908738521234052);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_5.x, omega_register.x, ((-fft_reg_5.y) * omega_register.y)), fma(fft_reg_5.x, omega_register.y, (fft_reg_5.y * omega_register.x)));
    fft_reg_5 = radix_register_0;
    omega_register.x = (tid * -0.5890486225480862);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_3.x, omega_register.x, ((-fft_reg_3.y) * omega_register.y)), fma(fft_reg_3.x, omega_register.y, (fft_reg_3.y * omega_register.x)));
    fft_reg_3 = radix_register_0;
    omega_register.x = (tid * -0.6872233929727672);
    omega_register = vkdispatch_make_float2(cos(omega_register.x), sin(omega_register.x));
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_7.x, omega_register.x, ((-fft_reg_7.y) * omega_register.y)), fma(fft_reg_7.x, omega_register.y, (fft_reg_7.y * omega_register.x)));
    fft_reg_7 = radix_register_0;
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_1;
    fft_reg_1 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_5;
    fft_reg_5 = (fft_reg_4 - radix_register_0);
    fft_reg_4 = (fft_reg_4 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_3;
    fft_reg_3 = (fft_reg_2 - radix_register_0);
    fft_reg_2 = (fft_reg_2 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_6 - radix_register_0);
    fft_reg_6 = (fft_reg_6 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_2;
    fft_reg_2 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 4. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_3.x;
    fft_reg_3.x = fft_reg_3.y;
    fft_reg_3.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_3;
    fft_reg_3 = (fft_reg_1 - radix_register_0);
    fft_reg_1 = (fft_reg_1 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_6;
    fft_reg_6 = (fft_reg_4 - radix_register_0);
    fft_reg_4 = (fft_reg_4 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 4. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_7.x;
    fft_reg_7.x = fft_reg_7.y;
    fft_reg_7.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_5 - radix_register_0);
    fft_reg_5 = (fft_reg_5 + radix_register_0);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_4;
    fft_reg_4 = (fft_reg_0 - radix_register_0);
    fft_reg_0 = (fft_reg_0 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 1.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_5.x, 0.7071067811865476, ((-fft_reg_5.y) * -0.7071067811865475)), fma(fft_reg_5.x, -0.7071067811865475, (fft_reg_5.y * 0.7071067811865476)));
    fft_reg_5 = radix_register_0;
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_5;
    fft_reg_5 = (fft_reg_1 - radix_register_0);
    fft_reg_1 = (fft_reg_1 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 2.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0.x = fft_reg_6.x;
    fft_reg_6.x = fft_reg_6.y;
    fft_reg_6.y = (-radix_register_0.x);
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_6;
    fft_reg_6 = (fft_reg_2 - radix_register_0);
    fft_reg_2 = (fft_reg_2 + radix_register_0);
    
    /*
     * Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
     * Twiddle domain size: N = 8. Twiddle index source: 3.
     * For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
     * This phase-aligns each sub-FFT with its parent decomposition stage.
     */
    radix_register_0 = vkdispatch_make_float2(fma(fft_reg_7.x, -0.7071067811865475, ((-fft_reg_7.y) * -0.7071067811865476)), fma(fft_reg_7.x, -0.7071067811865476, (fft_reg_7.y * -0.7071067811865475)));
    fft_reg_7 = radix_register_0;
    /* Radix-2 butterfly base case */
    radix_register_0 = fft_reg_7;
    fft_reg_7 = (fft_reg_3 - radix_register_0);
    fft_reg_3 = (fft_reg_3 + radix_register_0);
    
    /*
     * Writing register-resident FFT outputs to global memory.
     * Addressing uses computed batch offsets plus FFT-lane stride.
     */
    output_batch_offset = ((workgroup_index + vkdispatch_local_invocation_id().y) << 6);
    io_index = (tid + output_batch_offset);
    buf1.data[io_index] = fft_reg_0;
    io_index = ((tid + 8) + output_batch_offset);
    buf1.data[io_index] = fft_reg_1;
    io_index = ((tid + 16) + output_batch_offset);
    buf1.data[io_index] = fft_reg_2;
    io_index = ((tid + 24) + output_batch_offset);
    buf1.data[io_index] = fft_reg_3;
    io_index = ((tid + 32) + output_batch_offset);
    buf1.data[io_index] = fft_reg_4;
    io_index = ((tid + 40) + output_batch_offset);
    buf1.data[io_index] = fft_reg_5;
    io_index = ((tid + 48) + output_batch_offset);
    buf1.data[io_index] = fft_reg_6;
    io_index = ((tid + 56) + output_batch_offset);
    buf1.data[io_index] = fft_reg_7;
}"""


mod = SourceModule(cuda_kernel, no_extern_c=True)
kernel = mod.get_function("vkdispatch_main")

# --- Set up UniformObjectBuffer on device ---
# uint4 = 4x uint32 (16 bytes), int4 = 4x int32 (16 bytes)
# Total: 48 bytes, 16-byte aligned

n = 64
ubo_bytes = struct.pack(
    "4I 4i 4i",
    # exec_count (uint4)
    n, 1, 1, 0,
    # sdata_shape (int4)
    n, 1, 1, 1,
    # buf1_shape (int4)
    n, 1, 1, 1,
)

ubo_gpu = cuda.mem_alloc(len(ubo_bytes))
cuda.memcpy_htod(ubo_gpu, ubo_bytes)

# --- Set up Buffer1 data (float2 = 2x float32 per element) ---

buf1_data = np.random.randn(n, 2).astype(np.float32)
buf1_gpu = cuda.mem_alloc(buf1_data.nbytes)
cuda.memcpy_htod(buf1_gpu, buf1_data)

# --- Pack the Buffer1 struct (just a device pointer, 8 bytes) ---
# Buffer1 { float2* data } is passed BY VALUE, so we pack the pointer

buf1_struct = struct.pack("P", int(buf1_gpu))  # "P" = pointer-sized uint

# --- Launch ---

kernel(
    ubo_gpu,          # const UniformObjectBuffer* — passed as pointer
    buf1_struct,      # Buffer1 — passed by value as raw bytes
    block=(256, 1, 1),
    grid=((n + 255) // 256, 1),
)

# --- Verify ---

result = np.empty_like(buf1_data)
cuda.memcpy_dtoh(result, buf1_gpu)
assert np.allclose(result, buf1_data * 2.0)
print("Success:", result[:4])