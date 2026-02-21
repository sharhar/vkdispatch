from typing import List, Optional

import vkdispatch.base.dtype as dtypes

from .base import CodeGenBackend


class CUDABackend(CodeGenBackend):
    name = "cuda"

    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self._kernel_params: List[str] = []
        self._entry_alias_lines: List[str] = []

    def _register_kernel_param(self, param_decl: str) -> None:
        if param_decl not in self._kernel_params:
            self._kernel_params.append(param_decl)

    def _register_alias_line(self, alias_line: str) -> None:
        if alias_line not in self._entry_alias_lines:
            self._entry_alias_lines.append(alias_line)

    def type_name(self, var_type: dtypes.dtype) -> str:
        if var_type == dtypes.int32:
            return "int"
        if var_type == dtypes.uint32:
            return "unsigned int"
        if var_type == dtypes.float32:
            return "float"
        if var_type == dtypes.complex64:
            return "float2"

        if var_type == dtypes.ivec2:
            return "int2"
        if var_type == dtypes.ivec3:
            return "int3"
        if var_type == dtypes.ivec4:
            return "int4"

        if var_type == dtypes.uvec2:
            return "uint2"
        if var_type == dtypes.uvec3:
            return "uint3"
        if var_type == dtypes.uvec4:
            return "uint4"

        if var_type == dtypes.vec2:
            return "float2"
        if var_type == dtypes.vec3:
            return "float3"
        if var_type == dtypes.vec4:
            return "float4"

        if var_type == dtypes.mat2:
            return "vkdispatch_mat2"
        if var_type == dtypes.mat3:
            return "vkdispatch_mat3"
        if var_type == dtypes.mat4:
            return "vkdispatch_mat4"

        raise ValueError(f"Unsupported CUDA type mapping for '{var_type.name}'")

    def constructor(self, var_type: dtypes.dtype, args: List[str]) -> str:
        target_type = self.type_name(var_type)

        if dtypes.is_scalar(var_type):
            assert len(args) > 0, f"Constructor for scalar type '{var_type.name}' needs at least one argument."
            return f"(({target_type})({args[0]}))"

        if var_type == dtypes.mat2:
            return f"vkdispatch_make_mat2({', '.join(args)})"
        if var_type == dtypes.mat3:
            return f"vkdispatch_make_mat3({', '.join(args)})"
        if var_type == dtypes.mat4:
            return f"vkdispatch_make_mat4({', '.join(args)})"

        helper_name = f"vkdispatch_make_{target_type}"
        return f"{helper_name}({', '.join(args)})"

    def pre_header(self, *, enable_subgroup_ops: bool, enable_printf: bool) -> str:
        self.reset_state()

        subgroup_support = "1" if enable_subgroup_ops else "0"
        printf_support = "1" if enable_printf else "0"

        header = (
            "#include <cuda_runtime.h>\n"
            "#include <math.h>\n"
            "#include <stdint.h>\n\n"
            f"#define VKDISPATCH_ENABLE_SUBGROUP_OPS {subgroup_support}\n"
            f"#define VKDISPATCH_ENABLE_PRINTF {printf_support}\n\n"
        )

        header += """struct vkdispatch_mat2 {
    float2 c0;
    float2 c1;
};

struct vkdispatch_mat3 {
    float3 c0;
    float3 c1;
    float3 c2;
};

struct vkdispatch_mat4 {
    float4 c0;
    float4 c1;
    float4 c2;
    float4 c3;
};

__device__ __forceinline__ vkdispatch_mat2 vkdispatch_make_mat2(float2 c0, float2 c1) { return {c0, c1}; }
__device__ __forceinline__ vkdispatch_mat3 vkdispatch_make_mat3(float3 c0, float3 c1, float3 c2) { return {c0, c1, c2}; }
__device__ __forceinline__ vkdispatch_mat4 vkdispatch_make_mat4(float4 c0, float4 c1, float4 c2, float4 c3) { return {c0, c1, c2, c3}; }

__device__ __forceinline__ int2 vkdispatch_make_int2(int x, int y) { return make_int2(x, y); }
__device__ __forceinline__ int2 vkdispatch_make_int2(int x) { return make_int2(x, x); }
template <typename TVec> __device__ __forceinline__ int2 vkdispatch_make_int2(TVec v) { return make_int2((int)v.x, (int)v.y); }

__device__ __forceinline__ int3 vkdispatch_make_int3(int x, int y, int z) { return make_int3(x, y, z); }
__device__ __forceinline__ int3 vkdispatch_make_int3(int x) { return make_int3(x, x, x); }
template <typename TVec> __device__ __forceinline__ int3 vkdispatch_make_int3(TVec v) { return make_int3((int)v.x, (int)v.y, (int)v.z); }

__device__ __forceinline__ int4 vkdispatch_make_int4(int x, int y, int z, int w) { return make_int4(x, y, z, w); }
__device__ __forceinline__ int4 vkdispatch_make_int4(int x) { return make_int4(x, x, x, x); }
template <typename TVec> __device__ __forceinline__ int4 vkdispatch_make_int4(TVec v) { return make_int4((int)v.x, (int)v.y, (int)v.z, (int)v.w); }

__device__ __forceinline__ uint2 vkdispatch_make_uint2(unsigned int x, unsigned int y) { return make_uint2(x, y); }
__device__ __forceinline__ uint2 vkdispatch_make_uint2(unsigned int x) { return make_uint2(x, x); }
template <typename TVec> __device__ __forceinline__ uint2 vkdispatch_make_uint2(TVec v) { return make_uint2((unsigned int)v.x, (unsigned int)v.y); }

__device__ __forceinline__ uint3 vkdispatch_make_uint3(unsigned int x, unsigned int y, unsigned int z) { return make_uint3(x, y, z); }
__device__ __forceinline__ uint3 vkdispatch_make_uint3(unsigned int x) { return make_uint3(x, x, x); }
template <typename TVec> __device__ __forceinline__ uint3 vkdispatch_make_uint3(TVec v) { return make_uint3((unsigned int)v.x, (unsigned int)v.y, (unsigned int)v.z); }

__device__ __forceinline__ uint4 vkdispatch_make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return make_uint4(x, y, z, w); }
__device__ __forceinline__ uint4 vkdispatch_make_uint4(unsigned int x) { return make_uint4(x, x, x, x); }
template <typename TVec> __device__ __forceinline__ uint4 vkdispatch_make_uint4(TVec v) { return make_uint4((unsigned int)v.x, (unsigned int)v.y, (unsigned int)v.z, (unsigned int)v.w); }

__device__ __forceinline__ float2 vkdispatch_make_float2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ float2 vkdispatch_make_float2(float x) { return make_float2(x, x); }
template <typename TVec> __device__ __forceinline__ float2 vkdispatch_make_float2(TVec v) { return make_float2((float)v.x, (float)v.y); }

__device__ __forceinline__ float3 vkdispatch_make_float3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 vkdispatch_make_float3(float x) { return make_float3(x, x, x); }
template <typename TVec> __device__ __forceinline__ float3 vkdispatch_make_float3(TVec v) { return make_float3((float)v.x, (float)v.y, (float)v.z); }

__device__ __forceinline__ float4 vkdispatch_make_float4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
__device__ __forceinline__ float4 vkdispatch_make_float4(float x) { return make_float4(x, x, x, x); }
template <typename TVec> __device__ __forceinline__ float4 vkdispatch_make_float4(TVec v) { return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w); }

__device__ __forceinline__ uint3 vkdispatch_global_invocation_id() {
    return make_uint3(
        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x),
        (unsigned int)(blockIdx.y * blockDim.y + threadIdx.y),
        (unsigned int)(blockIdx.z * blockDim.z + threadIdx.z)
    );
}

__device__ __forceinline__ uint3 vkdispatch_local_invocation_id() {
    return make_uint3((unsigned int)threadIdx.x, (unsigned int)threadIdx.y, (unsigned int)threadIdx.z);
}

__device__ __forceinline__ uint3 vkdispatch_workgroup_id() {
    return make_uint3((unsigned int)blockIdx.x, (unsigned int)blockIdx.y, (unsigned int)blockIdx.z);
}

__device__ __forceinline__ unsigned int vkdispatch_local_invocation_index() {
    return (unsigned int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
}

__device__ __forceinline__ unsigned int vkdispatch_subgroup_size() { return (unsigned int)warpSize; }
__device__ __forceinline__ unsigned int vkdispatch_num_subgroups() {
    unsigned int local_count = (unsigned int)(blockDim.x * blockDim.y * blockDim.z);
    return (local_count + vkdispatch_subgroup_size() - 1u) / vkdispatch_subgroup_size();
}
__device__ __forceinline__ unsigned int vkdispatch_subgroup_id() {
    return vkdispatch_local_invocation_index() / vkdispatch_subgroup_size();
}
__device__ __forceinline__ unsigned int vkdispatch_subgroup_invocation_id() {
    return vkdispatch_local_invocation_index() % vkdispatch_subgroup_size();
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_add(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        value += __shfl_xor_sync(mask, value, (int)offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_mul(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        value *= __shfl_xor_sync(mask, value, (int)offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_min(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        T other = __shfl_xor_sync(mask, value, (int)offset);
        value = other < value ? other : value;
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_max(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        T other = __shfl_xor_sync(mask, value, (int)offset);
        value = other > value ? other : value;
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_and(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        value &= __shfl_xor_sync(mask, value, (int)offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_or(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        value |= __shfl_xor_sync(mask, value, (int)offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T vkdispatch_subgroup_xor(T value) {
    unsigned int mask = 0xffffffffu;
    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {
        value ^= __shfl_xor_sync(mask, value, (int)offset);
    }
    return value;
}

__device__ __forceinline__ float mod(float x, float y) { return fmodf(x, y); }
__device__ __forceinline__ float fract(float x) { return x - floorf(x); }
__device__ __forceinline__ float roundEven(float x) { return nearbyintf(x); }
__device__ __forceinline__ float mix(float x, float y, float a) { return x + (y - x) * a; }
__device__ __forceinline__ float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__device__ __forceinline__ float radians(float x) { return x * (3.14159265358979323846f / 180.0f); }
__device__ __forceinline__ float degrees(float x) { return x * (180.0f / 3.14159265358979323846f); }
__device__ __forceinline__ float inversesqrt(float x) { return rsqrtf(x); }

__device__ __forceinline__ int floatBitsToInt(float x) { return __float_as_int(x); }
__device__ __forceinline__ unsigned int floatBitsToUint(float x) { return __float_as_uint(x); }
__device__ __forceinline__ float intBitsToFloat(int x) { return __int_as_float(x); }
__device__ __forceinline__ float uintBitsToFloat(unsigned int x) { return __uint_as_float(x); }

__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float coord) { return tex1D<float4>(tex, coord); }
__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float2 coord) { return tex2D<float4>(tex, coord.x, coord.y); }
__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float3 coord) { return tex3D<float4>(tex, coord.x, coord.y, coord.z); }
__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float coord, float lod) { return tex1DLod<float4>(tex, coord, lod); }
__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float2 coord, float lod) { return tex2DLod<float4>(tex, coord.x, coord.y, lod); }
__device__ __forceinline__ float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float3 coord, float lod) { return tex3DLod<float4>(tex, coord.x, coord.y, coord.z, lod); }
"""

        return header

    def make_source(self, header: str, body: str, x: int, y: int, z: int) -> str:
        expected_size_header = (
            f"// Expected local size: ({x}, {y}, {z})\n"
            f"#define VKDISPATCH_EXPECTED_LOCAL_SIZE_X {x}\n"
            f"#define VKDISPATCH_EXPECTED_LOCAL_SIZE_Y {y}\n"
            f"#define VKDISPATCH_EXPECTED_LOCAL_SIZE_Z {z}\n"
        )
        return f"{expected_size_header}\n{header}\n{body}"

    def constant_namespace(self) -> str:
        return "UBO"

    def variable_namespace(self) -> str:
        return "PC"

    def exec_bounds_guard(self, exec_count_expr: str) -> str:
        gid = self.global_invocation_id_expr()
        return (
            f"if (({exec_count_expr}).x <= ({gid}).x || "
            f"({exec_count_expr}).y <= ({gid}).y || "
            f"({exec_count_expr}).z <= ({gid}).z) {{ return; }}\n"
        )

    def shared_buffer_declaration(self, var_type: dtypes.dtype, name: str, size: int) -> str:
        return f"__shared__ {self.type_name(var_type)} {name}[{size}];"

    def uniform_block_declaration(self, contents: str) -> str:
        self._register_kernel_param("const UniformObjectBuffer* UBO_ptr")
        self._register_alias_line("const UniformObjectBuffer& UBO = *UBO_ptr;")
        return f"\nstruct UniformObjectBuffer {{\n{contents}\n}};\n"

    def storage_buffer_declaration(self, binding: int, var_type: dtypes.dtype, name: str) -> str:
        struct_name = f"Buffer{binding}"
        self._register_kernel_param(f"{struct_name} {name}")
        return f"struct {struct_name} {{ {self.type_name(var_type)}* data; }};\n"

    def sampler_declaration(self, binding: int, dimensions: int, name: str) -> str:
        self._register_kernel_param(f"cudaTextureObject_t {name}")
        return f"// sampler binding {binding}, dimensions={dimensions}\n"

    def push_constant_declaration(self, contents: str) -> str:
        self._register_kernel_param("const PushConstant* PC_ptr")
        self._register_alias_line("const PushConstant& PC = *PC_ptr;")
        return f"\nstruct PushConstant {{\n{contents}\n}};\n"

    def entry_point(self, body_contents: str) -> str:
        params = ", ".join(self._kernel_params)

        alias_block = ""
        for line in self._entry_alias_lines:
            alias_block += f"    {line}\n"

        return (
            f'extern "C" __global__ void vkdispatch_main({params}) {{\n'
            f"{alias_block}"
            f"{body_contents}"
            f"}}\n"
        )

    def inf_f32_expr(self) -> str:
        return "uintBitsToFloat(0x7F800000u)"

    def ninf_f32_expr(self) -> str:
        return "uintBitsToFloat(0xFF800000u)"

    def float_bits_to_int_expr(self, var_expr: str) -> str:
        return f"floatBitsToInt({var_expr})"

    def float_bits_to_uint_expr(self, var_expr: str) -> str:
        return f"floatBitsToUint({var_expr})"

    def int_bits_to_float_expr(self, var_expr: str) -> str:
        return f"intBitsToFloat({var_expr})"

    def uint_bits_to_float_expr(self, var_expr: str) -> str:
        return f"uintBitsToFloat({var_expr})"

    def global_invocation_id_expr(self) -> str:
        return "vkdispatch_global_invocation_id()"

    def local_invocation_id_expr(self) -> str:
        return "vkdispatch_local_invocation_id()"

    def local_invocation_index_expr(self) -> str:
        return "vkdispatch_local_invocation_index()"

    def workgroup_id_expr(self) -> str:
        return "vkdispatch_workgroup_id()"

    def workgroup_size_expr(self) -> str:
        return "vkdispatch_make_uint3((unsigned int)blockDim.x, (unsigned int)blockDim.y, (unsigned int)blockDim.z)"

    def num_workgroups_expr(self) -> str:
        return "vkdispatch_make_uint3((unsigned int)gridDim.x, (unsigned int)gridDim.y, (unsigned int)gridDim.z)"

    def num_subgroups_expr(self) -> str:
        return "vkdispatch_num_subgroups()"

    def subgroup_id_expr(self) -> str:
        return "vkdispatch_subgroup_id()"

    def subgroup_size_expr(self) -> str:
        return "vkdispatch_subgroup_size()"

    def subgroup_invocation_id_expr(self) -> str:
        return "vkdispatch_subgroup_invocation_id()"

    def barrier_statement(self) -> str:
        return "__syncthreads();"

    def memory_barrier_statement(self) -> str:
        return "__threadfence();"

    def memory_barrier_buffer_statement(self) -> str:
        return "__threadfence();"

    def memory_barrier_shared_statement(self) -> str:
        return "__threadfence_block();"

    def memory_barrier_image_statement(self) -> str:
        return "__threadfence();"

    def group_memory_barrier_statement(self) -> str:
        return "__threadfence_block();"

    def subgroup_add_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_add({arg_expr})"

    def subgroup_mul_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_mul({arg_expr})"

    def subgroup_min_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_min({arg_expr})"

    def subgroup_max_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_max({arg_expr})"

    def subgroup_and_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_and({arg_expr})"

    def subgroup_or_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_or({arg_expr})"

    def subgroup_xor_expr(self, arg_expr: str) -> str:
        return f"vkdispatch_subgroup_xor({arg_expr})"

    def subgroup_elect_expr(self) -> str:
        return "((int)(vkdispatch_subgroup_invocation_id() == 0u))"

    def subgroup_barrier_statement(self) -> str:
        return "__syncwarp();"

    def printf_statement(self, fmt: str, args: List[str]) -> str:
        safe_fmt = fmt.replace("\\", "\\\\").replace('"', '\\"')

        if len(args) == 0:
            return f'printf("{safe_fmt}");'

        return f'printf("{safe_fmt}", {", ".join(args)});'

    def texture_size_expr(self, texture_expr: str, lod: int, dimensions: int) -> str:
        # CUDA texture objects do not expose shape directly in device code.
        # The future CUDA backend should pass explicit texture shape parameters.
        if dimensions == 1:
            return "1.0f"
        if dimensions == 2:
            return "vkdispatch_make_float2(1.0f)"
        if dimensions == 3:
            return "vkdispatch_make_float3(1.0f)"

        raise ValueError(f"Unsupported texture dimensions '{dimensions}'")

    def sample_texture_expr(self, texture_expr: str, coord_expr: str, lod_expr: Optional[str] = None) -> str:
        if lod_expr is None:
            return f"vkdispatch_sample_texture({texture_expr}, {coord_expr})"

        return f"vkdispatch_sample_texture({texture_expr}, {coord_expr}, {lod_expr})"
