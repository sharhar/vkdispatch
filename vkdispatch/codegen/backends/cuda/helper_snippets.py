from typing import Dict, List


_HELPER_SNIPPETS: Dict[str, str] = {
    "composite_types": "",
    "mat2_type": "",
    "mat3_type": "",
    "mat4_type": "",
    "make_mat2": "",
    "make_mat3": "",
    "make_mat4": "",
    "make_short2": "",
    "make_short3": "",
    "make_short4": "",
    "make_ushort2": "",
    "make_ushort3": "",
    "make_ushort4": "",
    "make_int2": "",
    "make_int3": "",
    "make_int4": "",
    "make_uint2": "",
    "make_uint3": "",
    "make_uint4": "",
    "make_half2": "",
    "make_half3": "",
    "make_half4": "",
    "float2_ops": "",
    "make_float2": "",
    "make_float3": "",
    "make_float4": "",
    "make_double2": "",
    "make_double3": "",
    "make_double4": "",
    "global_invocation_id": (
        "__device__ __forceinline__ vkdispatch_uint3 vkdispatch_global_invocation_id() {\n"
        "    return vkdispatch_uint3(\n"
        "        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x),\n"
        "        (unsigned int)(blockIdx.y * blockDim.y + threadIdx.y),\n"
        "        (unsigned int)(blockIdx.z * blockDim.z + threadIdx.z)\n"
        "    );\n"
        "}"
    ),
    "local_invocation_id": (
        "__device__ __forceinline__ vkdispatch_uint3 vkdispatch_local_invocation_id() {\n"
        "    return vkdispatch_uint3((unsigned int)threadIdx.x, (unsigned int)threadIdx.y, (unsigned int)threadIdx.z);\n"
        "}"
    ),
    "workgroup_id": (
        "__device__ __forceinline__ vkdispatch_uint3 vkdispatch_workgroup_id() {\n"
        "    return vkdispatch_uint3((unsigned int)blockIdx.x, (unsigned int)blockIdx.y, (unsigned int)blockIdx.z);\n"
        "}"
    ),
    "local_invocation_index": (
        "__device__ __forceinline__ unsigned int vkdispatch_local_invocation_index() {\n"
        "    return (unsigned int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));\n"
        "}"
    ),
    "subgroup_size": "__device__ __forceinline__ unsigned int vkdispatch_subgroup_size() { return (unsigned int)warpSize; }",
    "num_subgroups": (
        "__device__ __forceinline__ unsigned int vkdispatch_num_subgroups() {\n"
        "    unsigned int local_count = (unsigned int)(blockDim.x * blockDim.y * blockDim.z);\n"
        "    return (local_count + vkdispatch_subgroup_size() - 1u) / vkdispatch_subgroup_size();\n"
        "}"
    ),
    "subgroup_id": (
        "__device__ __forceinline__ unsigned int vkdispatch_subgroup_id() {\n"
        "    return vkdispatch_local_invocation_index() / vkdispatch_subgroup_size();\n"
        "}"
    ),
    "subgroup_invocation_id": (
        "__device__ __forceinline__ unsigned int vkdispatch_subgroup_invocation_id() {\n"
        "    return vkdispatch_local_invocation_index() % vkdispatch_subgroup_size();\n"
        "}"
    ),
    "subgroup_shuffle_xor": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_shuffle_xor(unsigned int mask, T value, int lane_mask) {\n"
        "    return __shfl_xor_sync(mask, value, lane_mask);\n"
        "}"
    ),
    "subgroup_add": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_add(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        value = value + vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_mul": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_mul(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        value = value * vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_min": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_min(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        T other = vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "        value = other < value ? other : value;\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_max": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_max(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        T other = vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "        value = other > value ? other : value;\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_and": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_and(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        value = value & vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_or": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_or(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        value = value | vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "subgroup_xor": (
        "template <typename T>\n"
        "__device__ __forceinline__ T vkdispatch_subgroup_xor(T value) {\n"
        "    unsigned int mask = 0xffffffffu;\n"
        "    for (unsigned int offset = vkdispatch_subgroup_size() >> 1; offset > 0u; offset >>= 1u) {\n"
        "        value = value ^ vkdispatch_subgroup_shuffle_xor(mask, value, (int)offset);\n"
        "    }\n"
        "    return value;\n"
        "}"
    ),
    "mod": (
        "__device__ __forceinline__ float mod(float x, float y) { return fmodf(x, y); }\n"
        "__device__ __forceinline__ double mod(double x, double y) { return fmod(x, y); }"
    ),
    "fract": (
        "__device__ __forceinline__ float fract(float x) { return x - floorf(x); }\n"
        "__device__ __forceinline__ double fract(double x) { return x - floor(x); }"
    ),
    "roundEven": (
        "__device__ __forceinline__ float roundEven(float x) { return nearbyintf(x); }\n"
        "__device__ __forceinline__ double roundEven(double x) { return nearbyint(x); }"
    ),
    "mix": (
        "__device__ __forceinline__ float mix(float x, float y, float a) { return x + (y - x) * a; }\n"
        "__device__ __forceinline__ double mix(double x, double y, double a) { return x + (y - x) * a; }"
    ),
    "step": (
        "__device__ __forceinline__ float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }\n"
        "__device__ __forceinline__ double step(double edge, double x) { return x < edge ? 0.0 : 1.0; }"
    ),
    "smoothstep": (
        "__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {\n"
        "    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);\n"
        "    return t * t * (3.0f - 2.0f * t);\n"
        "}\n"
        "__device__ __forceinline__ double smoothstep(double edge0, double edge1, double x) {\n"
        "    double t = fmin(fmax((x - edge0) / (edge1 - edge0), 0.0), 1.0);\n"
        "    return t * t * (3.0 - 2.0 * t);\n"
        "}"
    ),
    "radians": (
        "__device__ __forceinline__ float radians(float x) { return x * (3.14159265358979323846f / 180.0f); }\n"
        "__device__ __forceinline__ double radians(double x) { return x * (3.14159265358979323846 / 180.0); }"
    ),
    "degrees": (
        "__device__ __forceinline__ float degrees(float x) { return x * (180.0f / 3.14159265358979323846f); }\n"
        "__device__ __forceinline__ double degrees(double x) { return x * (180.0 / 3.14159265358979323846); }"
    ),
    "inversesqrt": (
        "__device__ __forceinline__ float inversesqrt(float x) { return rsqrtf(x); }\n"
        "__device__ __forceinline__ double inversesqrt(double x) { return rsqrt(x); }"
    ),
    "floatBitsToInt": "__device__ __forceinline__ int floatBitsToInt(float x) { return __float_as_int(x); }",
    "floatBitsToUint": "__device__ __forceinline__ unsigned int floatBitsToUint(float x) { return __float_as_uint(x); }",
    "intBitsToFloat": "__device__ __forceinline__ float intBitsToFloat(int x) { return __int_as_float(x); }",
    "uintBitsToFloat": "__device__ __forceinline__ float uintBitsToFloat(unsigned int x) { return __uint_as_float(x); }",
    "longlong_as_double": "__device__ __forceinline__ double longlong_as_double(long long x) { return __longlong_as_double(x); }",
    "ushort_as_half": "__device__ __forceinline__ __half ushort_as_half(unsigned short x) { __half h; *reinterpret_cast<unsigned short*>(&h) = x; return h; }",
    "sample_texture": "",
}

_HELPER_ORDER: List[str] = [
    "composite_types",
    "global_invocation_id",
    "local_invocation_id",
    "workgroup_id",
    "local_invocation_index",
    "subgroup_size",
    "num_subgroups",
    "subgroup_id",
    "subgroup_invocation_id",
    "subgroup_shuffle_xor",
    "subgroup_add",
    "subgroup_mul",
    "subgroup_min",
    "subgroup_max",
    "subgroup_and",
    "subgroup_or",
    "subgroup_xor",
    "mod",
    "fract",
    "roundEven",
    "mix",
    "step",
    "smoothstep",
    "radians",
    "degrees",
    "inversesqrt",
    "floatBitsToInt",
    "floatBitsToUint",
    "intBitsToFloat",
    "uintBitsToFloat",
    "longlong_as_double",
    "ushort_as_half",
    "sample_texture",
]

_HELPER_DEPENDENCIES: Dict[str, List[str]] = {
    "mat2_type": ["composite_types"],
    "mat3_type": ["composite_types"],
    "mat4_type": ["composite_types"],
    "make_mat2": ["composite_types"],
    "make_mat3": ["composite_types"],
    "make_mat4": ["composite_types"],
    "make_short2": ["composite_types"],
    "make_short3": ["composite_types"],
    "make_short4": ["composite_types"],
    "make_ushort2": ["composite_types"],
    "make_ushort3": ["composite_types"],
    "make_ushort4": ["composite_types"],
    "make_int2": ["composite_types"],
    "make_int3": ["composite_types"],
    "make_int4": ["composite_types"],
    "make_uint2": ["composite_types"],
    "make_uint3": ["composite_types"],
    "make_uint4": ["composite_types"],
    "make_half2": ["composite_types"],
    "make_half3": ["composite_types"],
    "make_half4": ["composite_types"],
    "float2_ops": ["composite_types"],
    "make_float2": ["composite_types"],
    "make_float3": ["composite_types"],
    "make_float4": ["composite_types"],
    "make_double2": ["composite_types"],
    "make_double3": ["composite_types"],
    "make_double4": ["composite_types"],
    "global_invocation_id": ["composite_types"],
    "local_invocation_id": ["composite_types"],
    "workgroup_id": ["composite_types"],
    "sample_texture": ["composite_types"],
    "num_subgroups": ["subgroup_size"],
    "subgroup_id": ["local_invocation_index", "subgroup_size"],
    "subgroup_invocation_id": ["local_invocation_index", "subgroup_size"],
    "subgroup_add": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_mul": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_min": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_max": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_and": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_or": ["subgroup_size", "subgroup_shuffle_xor"],
    "subgroup_xor": ["subgroup_size", "subgroup_shuffle_xor"],
}


def initialize_feature_usage() -> Dict[str, bool]:
    return {feature_name: False for feature_name in _HELPER_SNIPPETS}
