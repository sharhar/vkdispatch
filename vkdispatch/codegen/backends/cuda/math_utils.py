from typing import Dict, List, Optional, Set

import vkdispatch.base.dtype as dtypes

from .composite_emitters import _cuda_vec_components
from .specs import _CUDA_VEC_TYPE_SPECS, _FLOAT_VEC_HELPER_SUFFIX_MAP


def cuda_fast_unary_math_name(func_name: str) -> str:
    if func_name == "sin":
        return "__sinf"
    if func_name == "cos":
        return "__cosf"
    if func_name == "tan":
        return "__tanf"
    if func_name == "exp":
        return "__expf"
    if func_name == "exp2":
        return "__exp2f"
    if func_name == "log":
        return "__logf"
    if func_name == "log2":
        return "__log2f"
    if func_name == "asin":
        return "asinf"
    if func_name == "acos":
        return "acosf"
    if func_name == "atan":
        return "atanf"
    if func_name == "sinh":
        return "sinhf"
    if func_name == "cosh":
        return "coshf"
    if func_name == "tanh":
        return "tanhf"
    if func_name == "asinh":
        return "asinhf"
    if func_name == "acosh":
        return "acoshf"
    if func_name == "atanh":
        return "atanhf"
    if func_name == "sqrt":
        return "sqrtf"

    return func_name


def cuda_fast_binary_math_name(func_name: str) -> str:
    if func_name == "atan2":
        return "atan2f"
    if func_name == "pow":
        return "__powf"

    return func_name


def cuda_scalar_unary_math_name(func_name: str, scalar_type: str) -> str:
    if scalar_type == "__half":
        half_math = {
            "sin": "hsin",
            "cos": "hcos",
            "exp": "hexp",
            "exp2": "hexp2",
            "log": "hlog",
            "log2": "hlog2",
            "sqrt": "hsqrt",
        }
        return half_math.get(func_name, func_name)
    if scalar_type == "double":
        return func_name
    return cuda_fast_unary_math_name(func_name)


def cuda_scalar_binary_math_name(func_name: str, scalar_type: str) -> str:
    if scalar_type == "__half":
        return func_name
    if scalar_type == "double":
        return func_name
    return cuda_fast_binary_math_name(func_name)


def cuda_float_vec_components_for_suffix(helper_suffix: str) -> List[str]:
    dim_char = helper_suffix[-1]
    if dim_char == "2":
        return ["x", "y"]
    if dim_char == "3":
        return ["x", "y", "z"]
    if dim_char == "4":
        return ["x", "y", "z", "w"]

    raise ValueError(f"Unsupported CUDA float vector helper suffix '{helper_suffix}'")


def cuda_float_vec_helper_suffix(var_type: dtypes.dtype) -> Optional[str]:
    return _FLOAT_VEC_HELPER_SUFFIX_MAP.get(var_type)


def emit_used_vec_math_helpers(
    composite_vec_unary_math_usage: Dict[str, Set[str]],
    composite_vec_binary_math_usage: Dict[str, Set[str]],
) -> str:
    helper_sections: List[str] = []

    unary_order = [
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "exp2",
        "log",
        "log2",
        "sqrt",
    ]
    binary_order = ["atan2", "pow"]
    signature_order = ["vv", "vs", "sv"]

    for key in ["half2", "half3", "half4", "float2", "float3", "float4", "double2", "double3", "double4"]:
        unary_funcs = composite_vec_unary_math_usage.get(key, set())
        binary_tokens = composite_vec_binary_math_usage.get(key, set())
        if len(unary_funcs) == 0 and len(binary_tokens) == 0:
            continue

        if key not in _CUDA_VEC_TYPE_SPECS:
            continue

        vec_name, scalar_type, dim, _, _, _ = _CUDA_VEC_TYPE_SPECS[key]
        comps = _cuda_vec_components(dim)
        lines: List[str] = []

        for func_name in unary_order:
            if func_name not in unary_funcs:
                continue
            scalar_func = cuda_scalar_unary_math_name(func_name, scalar_type)
            comp_args = ", ".join([f"{scalar_func}(v.v.{c})" for c in comps])
            lines.append(
                f"__device__ __forceinline__ {vec_name} {func_name}(const {vec_name}& v) {{ return vkdispatch_make_{key}({comp_args}); }}"
            )

        for func_name in binary_order:
            scalar_func = cuda_scalar_binary_math_name(func_name, scalar_type)
            for signature in signature_order:
                token = f"{func_name}:{signature}"
                if token not in binary_tokens:
                    continue

                if signature == "vv":
                    comp_args = ", ".join([f"{scalar_func}(a.v.{c}, b.v.{c})" for c in comps])
                    lines.append(
                        f"__device__ __forceinline__ {vec_name} {func_name}(const {vec_name}& a, const {vec_name}& b) {{ return vkdispatch_make_{key}({comp_args}); }}"
                    )
                elif signature == "vs":
                    comp_args = ", ".join([f"{scalar_func}(a.v.{c}, b)" for c in comps])
                    lines.append(
                        f"__device__ __forceinline__ {vec_name} {func_name}(const {vec_name}& a, {scalar_type} b) {{ return vkdispatch_make_{key}({comp_args}); }}"
                    )
                elif signature == "sv":
                    comp_args = ", ".join([f"{scalar_func}(a, b.v.{c})" for c in comps])
                    lines.append(
                        f"__device__ __forceinline__ {vec_name} {func_name}({scalar_type} a, const {vec_name}& b) {{ return vkdispatch_make_{key}({comp_args}); }}"
                    )

        if len(lines) > 0:
            helper_sections.append("\n".join(lines))

    return "\n\n".join(helper_sections)
