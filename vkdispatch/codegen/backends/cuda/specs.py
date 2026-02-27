from typing import Dict, FrozenSet, Tuple

import vkdispatch.base.dtype as dtypes


_CUDA_VEC_TYPE_SPECS: Dict[str, Tuple[str, str, int, str, bool, bool]] = {
    "short2": ("vkdispatch_short2", "short", 2, "short2", True, True),
    "short3": ("vkdispatch_short3", "short", 3, "short3", True, True),
    "short4": ("vkdispatch_short4", "short", 4, "short4", True, True),
    "ushort2": ("vkdispatch_ushort2", "unsigned short", 2, "ushort2", False, True),
    "ushort3": ("vkdispatch_ushort3", "unsigned short", 3, "ushort3", False, True),
    "ushort4": ("vkdispatch_ushort4", "unsigned short", 4, "ushort4", False, True),
    "int2": ("vkdispatch_int2", "int", 2, "int2", True, True),
    "int3": ("vkdispatch_int3", "int", 3, "int3", True, True),
    "int4": ("vkdispatch_int4", "int", 4, "int4", True, True),
    "uint2": ("vkdispatch_uint2", "unsigned int", 2, "uint2", False, True),
    "uint3": ("vkdispatch_uint3", "unsigned int", 3, "uint3", False, True),
    "uint4": ("vkdispatch_uint4", "unsigned int", 4, "uint4", False, True),
    "half2": ("vkdispatch_half2", "__half", 2, "half2", True, False),
    "half3": ("vkdispatch_half3", "__half", 3, "half3", True, False),
    "half4": ("vkdispatch_half4", "__half", 4, "half4", True, False),
    "float2": ("vkdispatch_float2", "float", 2, "float2", True, False),
    "float3": ("vkdispatch_float3", "float", 3, "float3", True, False),
    "float4": ("vkdispatch_float4", "float", 4, "float4", True, False),
    "double2": ("vkdispatch_double2", "double", 2, "double2", True, False),
    "double3": ("vkdispatch_double3", "double", 3, "double3", True, False),
    "double4": ("vkdispatch_double4", "double", 4, "double4", True, False),
}

_CUDA_MAT_TYPE_SPECS: Dict[str, Tuple[str, str, str, int]] = {
    "mat2": ("vkdispatch_mat2", "vkdispatch_float2", "float2", 2),
    "mat3": ("vkdispatch_mat3", "vkdispatch_float3", "float3", 3),
    "mat4": ("vkdispatch_mat4", "vkdispatch_float4", "float4", 4),
}

_CUDA_VEC_ORDER = [
    "short2", "short3", "short4",
    "ushort2", "ushort3", "ushort4",
    "int2", "int3", "int4",
    "uint2", "uint3", "uint4",
    "half2", "half3", "half4",
    "float2", "float3", "float4",
    "double2", "double3", "double4",
]

_CUDA_MAT_ORDER = ["mat2", "mat3", "mat4"]

_DTYPE_TO_COMPOSITE_KEY = {
    dtypes.ihvec2: "short2",
    dtypes.ihvec3: "short3",
    dtypes.ihvec4: "short4",
    dtypes.uhvec2: "ushort2",
    dtypes.uhvec3: "ushort3",
    dtypes.uhvec4: "ushort4",
    dtypes.ivec2: "int2",
    dtypes.ivec3: "int3",
    dtypes.ivec4: "int4",
    dtypes.uvec2: "uint2",
    dtypes.uvec3: "uint3",
    dtypes.uvec4: "uint4",
    dtypes.hvec2: "half2",
    dtypes.hvec3: "half3",
    dtypes.hvec4: "half4",
    dtypes.complex32: "half2",
    dtypes.complex64: "float2",
    dtypes.complex128: "double2",
    dtypes.vec2: "float2",
    dtypes.vec3: "float3",
    dtypes.vec4: "float4",
    dtypes.dvec2: "double2",
    dtypes.dvec3: "double3",
    dtypes.dvec4: "double4",
    dtypes.mat2: "mat2",
    dtypes.mat3: "mat3",
    dtypes.mat4: "mat4",
}

_SCALAR_TYPE_NAMES = {
    dtypes.int16: "short",
    dtypes.uint16: "unsigned short",
    dtypes.int32: "int",
    dtypes.uint32: "unsigned int",
    dtypes.int64: "long long",
    dtypes.uint64: "unsigned long long",
    dtypes.float16: "__half",
    dtypes.float32: "float",
    dtypes.float64: "double",
}

_FLOAT_VEC_DTYPES: FrozenSet[dtypes.dtype] = frozenset(
    {
        dtypes.complex32,
        dtypes.complex64,
        dtypes.complex128,
        dtypes.hvec2,
        dtypes.hvec3,
        dtypes.hvec4,
        dtypes.vec2,
        dtypes.vec3,
        dtypes.vec4,
        dtypes.dvec2,
        dtypes.dvec3,
        dtypes.dvec4,
    }
)

_FLOAT_VEC_HELPER_SUFFIX_MAP = {
    dtypes.hvec2: "half2",
    dtypes.hvec3: "half3",
    dtypes.hvec4: "half4",
    dtypes.complex32: "half2",
    dtypes.complex64: "float2",
    dtypes.complex128: "double2",
    dtypes.vec2: "float2",
    dtypes.vec3: "float3",
    dtypes.vec4: "float4",
    dtypes.dvec2: "double2",
    dtypes.dvec3: "double3",
    dtypes.dvec4: "double4",
}
