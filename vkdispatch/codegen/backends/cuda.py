from typing import Dict, List, Optional, Set, Tuple

import vkdispatch.base.dtype as dtypes

from .base import CodeGenBackend


def _cuda_vec_components(dim: int) -> List[str]:
    if dim < 2 or dim > 4:
        raise ValueError(f"Unsupported vector dimension '{dim}'")
    return list("xyzw"[:dim])


def _cuda_join_statements(statements: List[str]) -> str:
    if len(statements) == 0:
        return ""
    return " ".join(statements)


def _cuda_emit_vec_type(
    vec_name: str,
    scalar_type: str,
    dim: int,
    cuda_native_type: str,
    *,
    allow_unary_neg: bool,
    enable_bitwise: bool,
    needed_ops: Optional[Set[str]] = None,
) -> str:
    comps = _cuda_vec_components(dim)
    if needed_ops is None:
        needed_ops = set()
        if allow_unary_neg:
            needed_ops.add("un:-")
        if enable_bitwise:
            needed_ops.add("un:~")
        for op in ["+", "-", "*", "/"]:
            needed_ops.add(f"cmpd:{op}=:v")
            needed_ops.add(f"cmpd:{op}=:s")
            needed_ops.add(f"bin:{op}:vv")
            needed_ops.add(f"bin:{op}:vs")
            needed_ops.add(f"bin:{op}:sv")
        if enable_bitwise:
            for op in ["&", "|", "^", "<<", ">>"]:
                needed_ops.add(f"cmpd:{op}=:v")
                needed_ops.add(f"cmpd:{op}=:s")
                needed_ops.add(f"bin:{op}:vv")
                needed_ops.add(f"bin:{op}:vs")
                needed_ops.add(f"bin:{op}:sv")

    def has(token: str) -> bool:
        return token in needed_ops

    def self_comp(c: str) -> str:
        return f"v.{c}"

    def wrap_comp(obj: str, c: str) -> str:
        return f"{obj}.v.{c}"

    def native_comp(obj: str, c: str) -> str:
        return f"{obj}.{c}"

    def index_op_body() -> str:
        branches: List[str] = []
        for idx, c in enumerate(comps):
            prefix = "if" if idx == 0 else "else if"
            branches.append(f"{prefix} (i == {idx}) return v.{c};")
        branches.append(f"else return v.{comps[0]};")
        return " ".join(branches)

    lines: List[str] = [f"struct {vec_name} {{"]
    lines.append(f"    {cuda_native_type} v;")
    lines.append("")
    ctor_args = ", ".join([f"{scalar_type} {c}_" for c in comps])
    ctor_init = "{" + ", ".join([f"{c}_" for c in comps]) + "}"
    splat_init = "{" + ", ".join(["s" for _ in comps]) + "}"
    cast_init = "{" + ", ".join([f"({scalar_type}){native_comp('src', c)}" for c in comps]) + "}"
    lines.append(f"    __device__ __forceinline__ {vec_name}() = default;")
    lines.append(f"    __device__ __forceinline__ {vec_name}({ctor_args}) : v{ctor_init} {{}}")
    lines.append(f"    __device__ __forceinline__ explicit {vec_name}({scalar_type} s) : v{splat_init} {{}}")
    lines.append(f"    __device__ __forceinline__ explicit {vec_name}(const {cuda_native_type}& native) : v(native) {{}}")
    lines.append("    template <typename TVec>")
    lines.append(f"    __device__ __forceinline__ explicit {vec_name}(const TVec& src) : v{cast_init} {{}}")
    lines.append(f"    __device__ __forceinline__ {scalar_type}& operator[](int i) {{ {index_op_body()} }}")
    lines.append(f"    __device__ __forceinline__ const {scalar_type}& operator[](int i) const {{ {index_op_body()} }}")

    if allow_unary_neg and has("un:-"):
        neg_expr = ", ".join([f"-{self_comp(c)}" for c in comps])
        lines.append(f"    __device__ __forceinline__ {vec_name} operator-() const {{ return {vec_name}({neg_expr}); }}")

    if enable_bitwise and has("un:~"):
        not_expr = ", ".join([f"~{self_comp(c)}" for c in comps])
        lines.append(f"    __device__ __forceinline__ {vec_name} operator~() const {{ return {vec_name}({not_expr}); }}")

    for op in ["+", "-", "*", "/"]:
        op_assign = op + "="
        if has(f"cmpd:{op}=:v"):
            vv_ops = _cuda_join_statements([f"{self_comp(c)} {op_assign} {wrap_comp('b', c)};" for c in comps])
            lines.append(
                f"    __device__ __forceinline__ {vec_name}& operator{op_assign}(const {vec_name}& b) {{ {vv_ops} return *this; }}"
            )
        if has(f"cmpd:{op}=:s"):
            sv_ops = _cuda_join_statements([f"{self_comp(c)} {op_assign} b;" for c in comps])
            lines.append(
                f"    __device__ __forceinline__ {vec_name}& operator{op_assign}({scalar_type} b) {{ {sv_ops} return *this; }}"
            )

    if enable_bitwise:
        for op in ["&", "|", "^", "<<", ">>"]:
            op_assign = op + "="
            if has(f"cmpd:{op}=:v"):
                vv_ops = _cuda_join_statements([f"{self_comp(c)} {op_assign} {wrap_comp('b', c)};" for c in comps])
                lines.append(
                    f"    __device__ __forceinline__ {vec_name}& operator{op_assign}(const {vec_name}& b) {{ {vv_ops} return *this; }}"
                )
            if has(f"cmpd:{op}=:s"):
                sv_ops = _cuda_join_statements([f"{self_comp(c)} {op_assign} b;" for c in comps])
                lines.append(
                    f"    __device__ __forceinline__ {vec_name}& operator{op_assign}({scalar_type} b) {{ {sv_ops} return *this; }}"
                )

    lines.append("};")
    lines.append(
        f'static_assert(sizeof({vec_name}) == sizeof({cuda_native_type}), "{vec_name} size must match {cuda_native_type}");'
    )
    lines.append(
        f'static_assert(alignof({vec_name}) == alignof({cuda_native_type}), "{vec_name} alignment must match {cuda_native_type}");'
    )

    # Arithmetic operators (vector/vector, vector/scalar, scalar/vector)
    for op in ["+", "-", "*", "/"]:
        if has(f"bin:{op}:vv"):
            vv_expr = ", ".join([f"({wrap_comp('a', c)} {op} {wrap_comp('b', c)})" for c in comps])
            lines.append(
                f"__device__ __forceinline__ {vec_name} operator{op}(const {vec_name}& a, const {vec_name}& b) {{ return {vec_name}({vv_expr}); }}"
            )
        if has(f"bin:{op}:vs"):
            vs_expr = ", ".join([f"({wrap_comp('a', c)} {op} b)" for c in comps])
            lines.append(
                f"__device__ __forceinline__ {vec_name} operator{op}(const {vec_name}& a, {scalar_type} b) {{ return {vec_name}({vs_expr}); }}"
            )
        if has(f"bin:{op}:sv"):
            if op in ["+", "*"]:
                sv_expr = ", ".join([f"(a {op} {wrap_comp('b', c)})" for c in comps])
            else:
                sv_expr = ", ".join([f"({scalar_type})(a {op} {wrap_comp('b', c)})" for c in comps])
            lines.append(
                f"__device__ __forceinline__ {vec_name} operator{op}({scalar_type} a, const {vec_name}& b) {{ return {vec_name}({sv_expr}); }}"
            )

    if enable_bitwise:
        for op in ["&", "|", "^", "<<", ">>"]:
            if has(f"bin:{op}:vv"):
                vv_expr = ", ".join([f"({wrap_comp('a', c)} {op} {wrap_comp('b', c)})" for c in comps])
                lines.append(
                    f"__device__ __forceinline__ {vec_name} operator{op}(const {vec_name}& a, const {vec_name}& b) {{ return {vec_name}({vv_expr}); }}"
                )
            if has(f"bin:{op}:vs"):
                vs_expr = ", ".join([f"({wrap_comp('a', c)} {op} b)" for c in comps])
                lines.append(
                    f"__device__ __forceinline__ {vec_name} operator{op}(const {vec_name}& a, {scalar_type} b) {{ return {vec_name}({vs_expr}); }}"
                )
            if has(f"bin:{op}:sv"):
                sv_expr = ", ".join([f"({scalar_type})(a {op} {wrap_comp('b', c)})" for c in comps])
                lines.append(
                    f"__device__ __forceinline__ {vec_name} operator{op}({scalar_type} a, const {vec_name}& b) {{ return {vec_name}({sv_expr}); }}"
                )

    return "\n".join(lines)


def _cuda_emit_vec_helper(helper_suffix: str, vec_name: str, scalar_type: str, dim: int) -> str:
    comps = _cuda_vec_components(dim)
    args = ", ".join([f"{scalar_type} {c}" for c in comps])
    ctor_args = ", ".join(comps)
    return "\n".join(
        [
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}({args}) {{ return {vec_name}({ctor_args}); }}",
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}({scalar_type} x) {{ return {vec_name}(x); }}",
            "template <typename TVec>",
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}(TVec v) {{ return {vec_name}(v); }}",
        ]
    )


def _cuda_emit_vec_wrapper_conversion_helpers(
    helper_suffix: str,
    vec_name: str,
    scalar_type: str,
    dim: int,
    *,
    available_keys: Optional[Set[str]] = None,
) -> str:
    comps = _cuda_vec_components(dim)
    dim_keys = [key for key in _CUDA_VEC_TYPE_SPECS if key.endswith(str(dim))]
    if available_keys is not None:
        dim_keys = [key for key in dim_keys if key in available_keys]

    lines: List[str] = []
    for src_key in dim_keys:
        if src_key == helper_suffix:
            continue
        src_vec_name = _CUDA_VEC_TYPE_SPECS[src_key][0]
        ctor_args = ", ".join([f"({scalar_type})src.v.{c}" for c in comps])
        lines.append(
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}(const {src_vec_name}& src) {{ return {vec_name}({ctor_args}); }}"
        )

    return "\n".join(lines)


def _cuda_emit_mat_type(mat_name: str, vec_name: str, dim: int, needed_ops: Optional[Set[str]] = None) -> str:
    cols = [f"c{i}" for i in range(dim)]
    if needed_ops is None:
        needed_ops = {
            "un:-",
            "cmpd:+=:m", "cmpd:+=:s",
            "cmpd:-=:m", "cmpd:-=:s",
            "cmpd:*=:s", "cmpd:/=:s",
            "bin:+:mm", "bin:+:ms", "bin:+:sm",
            "bin:-:mm", "bin:-:ms", "bin:-:sm",
            "bin:*:ms", "bin:*:sm", "bin:/:ms", "bin:/:sm",
            "bin:*:mv", "bin:*:vm", "bin:*:mm",
        }

    def has(token: str) -> bool:
        return token in needed_ops

    lines: List[str] = [f"struct {mat_name} {{"]
    lines.extend([f"    {vec_name} {c};" for c in cols])
    lines.append("")
    lines.append(f"    __device__ __forceinline__ {mat_name}() = default;")
    ctor_args = ", ".join([f"{vec_name} {c}_" for c in cols])
    ctor_init = ", ".join([f"{c}({c}_)" for c in cols])
    lines.append(f"    __device__ __forceinline__ {mat_name}({ctor_args}) : {ctor_init} {{}}")

    zero = "0.0f"
    diag_init = ", ".join(
        [f"c{col_idx}({vec_name}({', '.join(['s' if row_idx == col_idx else zero for row_idx in range(dim)])}))" for col_idx in range(dim)]
    )
    lines.append(f"    __device__ __forceinline__ explicit {mat_name}(float s) : {diag_init} {{}}")
    lines.append(f"    __device__ __forceinline__ {vec_name}& operator[](int i) {{ return (&c0)[i]; }}")
    lines.append(f"    __device__ __forceinline__ const {vec_name}& operator[](int i) const {{ return (&c0)[i]; }}")
    if has("un:-"):
        lines.append(f"    __device__ __forceinline__ {mat_name} operator-() const {{ return {mat_name}({', '.join([f'-c{i}' for i in range(dim)])}); }}")

    for op in ["+", "-"]:
        op_assign = op + "="
        if has(f"cmpd:{op}=:m"):
            mm_ops = _cuda_join_statements([f"c{i} {op_assign} b.c{i};" for i in range(dim)])
            lines.append(
                f"    __device__ __forceinline__ {mat_name}& operator{op_assign}(const {mat_name}& b) {{ {mm_ops} return *this; }}"
            )
        if has(f"cmpd:{op}=:s"):
            ms_ops = _cuda_join_statements([f"c{i} {op_assign} b;" for i in range(dim)])
            lines.append(
                f"    __device__ __forceinline__ {mat_name}& operator{op_assign}(float b) {{ {ms_ops} return *this; }}"
            )

    for op in ["*", "/"]:
        op_assign = op + "="
        if has(f"cmpd:{op}=:s"):
            ms_ops = _cuda_join_statements([f"c{i} {op_assign} b;" for i in range(dim)])
            lines.append(
                f"    __device__ __forceinline__ {mat_name}& operator{op_assign}(float b) {{ {ms_ops} return *this; }}"
            )

    lines.append("};")

    # Basic arithmetic
    for op in ["+", "-"]:
        if has(f"bin:{op}:mm"):
            cols_expr = ", ".join([f"(a.c{i} {op} b.c{i})" for i in range(dim)])
            lines.append(
                f"__device__ __forceinline__ {mat_name} operator{op}(const {mat_name}& a, const {mat_name}& b) {{ return {mat_name}({cols_expr}); }}"
            )
        if has(f"bin:{op}:ms"):
            cols_expr = ", ".join([f"(a.c{i} {op} b)" for i in range(dim)])
            lines.append(
                f"__device__ __forceinline__ {mat_name} operator{op}(const {mat_name}& a, float b) {{ return {mat_name}({cols_expr}); }}"
            )
        if has(f"bin:{op}:sm"):
            cols_expr = ", ".join([f"(a {op} b.c{i})" for i in range(dim)])
            lines.append(
                f"__device__ __forceinline__ {mat_name} operator{op}(float a, const {mat_name}& b) {{ return {mat_name}({cols_expr}); }}"
            )

    for op in ["*", "/"]:
        if has(f"bin:{op}:ms"):
            cols_expr = ", ".join([f"(a.c{i} {op} b)" for i in range(dim)])
            lines.append(
                f"__device__ __forceinline__ {mat_name} operator{op}(const {mat_name}& a, float b) {{ return {mat_name}({cols_expr}); }}"
            )
        if has(f"bin:{op}:sm"):
            cols_expr = ", ".join([f"(a {op} b.c{i})" for i in range(dim)])
            lines.append(
                f"__device__ __forceinline__ {mat_name} operator{op}(float a, const {mat_name}& b) {{ return {mat_name}({cols_expr}); }}"
            )

    # GLSL-style matrix/vector products (column-major)
    vec_comps = _cuda_vec_components(dim)
    if has("bin:*:mv"):
        mat_vec_terms = [f"(m.c{i} * v.v.{vec_comps[i]})" for i in range(dim)]
        mat_vec_expr = " + ".join(mat_vec_terms)
        lines.append(
            f"__device__ __forceinline__ {vec_name} operator* (const {mat_name}& m, const {vec_name}& v) {{ return {mat_vec_expr}; }}"
        )

    if has("bin:*:vm"):
        row_exprs: List[str] = []
        for col_idx in range(dim):
            terms = [f"(v.v.{vec_comps[row_idx]} * m.c{col_idx}.v.{vec_comps[row_idx]})" for row_idx in range(dim)]
            row_exprs.append(" + ".join(terms))
        lines.append(
            f"__device__ __forceinline__ {vec_name} operator* (const {vec_name}& v, const {mat_name}& m) {{ return {vec_name}({', '.join(row_exprs)}); }}"
        )

    if has("bin:*:mm"):
        col_products = ", ".join([f"(a * b.c{i})" for i in range(dim)])
        lines.append(
            f"__device__ __forceinline__ {mat_name} operator* (const {mat_name}& a, const {mat_name}& b) {{ return {mat_name}({col_products}); }}"
        )

    return "\n".join(lines)


def _cuda_emit_mat_helpers(mat_name: str, helper_suffix: str, vec_name: str, vec_helper_suffix: str, dim: int) -> str:
    col_type = vec_name
    col_args = ", ".join([f"{col_type} c{i}" for i in range(dim)])
    col_ctor = ", ".join([f"c{i}" for i in range(dim)])

    flat_names = [f"m{col}{row}" for col in range(dim) for row in range(dim)]
    flat_args = ", ".join([f"float {name}" for name in flat_names])
    flat_cols: List[str] = []
    for col in range(dim):
        values = [f"m{col}{row}" for row in range(dim)]
        flat_cols.append(f"vkdispatch_make_{vec_helper_suffix}({', '.join(values)})")
    flat_ctor = ", ".join(flat_cols)

    cast_cols = ", ".join([f"vkdispatch_make_{vec_helper_suffix}(m[{i}])" for i in range(dim)])

    return "\n".join(
        [
            f"__device__ __forceinline__ {mat_name} vkdispatch_make_{helper_suffix}({col_args}) {{ return {mat_name}({col_ctor}); }}",
            f"__device__ __forceinline__ {mat_name} vkdispatch_make_{helper_suffix}(float s) {{ return {mat_name}(s); }}",
            f"__device__ __forceinline__ {mat_name} vkdispatch_make_{helper_suffix}({flat_args}) {{ return {mat_name}({flat_ctor}); }}",
            "template <typename TMat>",
            f"__device__ __forceinline__ {mat_name} vkdispatch_make_{helper_suffix}(TMat m) {{ return {mat_name}({cast_cols}); }}",
        ]
    )


def _cuda_emit_subgroup_shuffle_xor_vec_overloads(vec_keys: Set[str]) -> str:
    lines: List[str] = []
    vec_order = [
        "short2", "short3", "short4",
        "ushort2", "ushort3", "ushort4",
        "int2", "int3", "int4",
        "uint2", "uint3", "uint4",
        "half2", "half3", "half4",
        "float2", "float3", "float4",
        "double2", "double3", "double4",
    ]

    for key in vec_order:
        if key not in vec_keys:
            continue

        vec_name, _, dim, _, _, _ = _CUDA_VEC_TYPE_SPECS[key]
        comps = _cuda_vec_components(dim)
        comp_exprs = ", ".join([f"__shfl_xor_sync(mask, value.v.{c}, lane_mask)" for c in comps])
        lines.append(
            f"__device__ __forceinline__ {vec_name} vkdispatch_subgroup_shuffle_xor(unsigned int mask, const {vec_name}& value, int lane_mask) "
            f"{{ return vkdispatch_make_{key}({comp_exprs}); }}"
        )

    return "\n".join(lines)

_CUDA_VEC_TYPE_SPECS = {
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

_CUDA_MAT_TYPE_SPECS = {
    "mat2": ("vkdispatch_mat2", "vkdispatch_float2", "float2", 2),
    "mat3": ("vkdispatch_mat3", "vkdispatch_float3", "float3", 3),
    "mat4": ("vkdispatch_mat4", "vkdispatch_float4", "float4", 4),
}


class CUDABackend(CodeGenBackend):
    name = "cuda"
    _CUDA_BUILTIN_UVEC3_SENTINELS: Dict[str, Dict[str, str]] = {
        "global_invocation_id": {
            "sentinel": "VKDISPATCH_CUDA_GLOBAL_INVOCATION_ID_SENTINEL()",
            "x": "(unsigned int)(blockIdx.x * blockDim.x + threadIdx.x)",
            "y": "(unsigned int)(blockIdx.y * blockDim.y + threadIdx.y)",
            "z": "(unsigned int)(blockIdx.z * blockDim.z + threadIdx.z)",
        },
        "local_invocation_id": {
            "sentinel": "VKDISPATCH_CUDA_LOCAL_INVOCATION_ID_SENTINEL()",
            "x": "(unsigned int)threadIdx.x",
            "y": "(unsigned int)threadIdx.y",
            "z": "(unsigned int)threadIdx.z",
        },
        "workgroup_id": {
            "sentinel": "VKDISPATCH_CUDA_WORKGROUP_ID_SENTINEL()",
            "x": "(unsigned int)blockIdx.x",
            "y": "(unsigned int)blockIdx.y",
            "z": "(unsigned int)blockIdx.z",
        },
    }

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

    def __init__(self) -> None:
        self._fixed_preamble = ""
        self.reset_state()

    def reset_state(self) -> None:
        self._kernel_params: List[str] = []
        self._entry_alias_lines: List[str] = []
        self._composite_type_usage: Set[str] = set()
        self._composite_vec_op_usage: Dict[str, Set[str]] = {}
        self._composite_mat_op_usage: Dict[str, Set[str]] = {}
        self._composite_vec_unary_math_usage: Dict[str, Set[str]] = {}
        self._composite_vec_binary_math_usage: Dict[str, Set[str]] = {}
        self._sample_texture_dims: Set[int] = set()
        self._needs_cuda_fp16: bool = False
        self._feature_usage: Dict[str, bool] = {
            feature_name: False
            for feature_name in self._HELPER_SNIPPETS
        }

    def mark_feature_usage(self, feature_name: str) -> None:
        if feature_name in self._feature_usage:
            self._feature_usage[feature_name] = True

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

    def _composite_key_for_dtype(self, var_type: dtypes.dtype) -> Optional[str]:
        return self._DTYPE_TO_COMPOSITE_KEY.get(var_type)

    def _record_composite_type_key(self, key: str) -> None:
        self.mark_feature_usage("composite_types")
        self._composite_type_usage.add(key)

        if key in _CUDA_MAT_TYPE_SPECS:
            dim = _CUDA_MAT_TYPE_SPECS[key][3]
            self._composite_type_usage.add(f"float{dim}")

    def _record_composite_type(self, var_type: dtypes.dtype) -> Optional[str]:
        key = self._composite_key_for_dtype(var_type)
        if key is None:
            return None
        self._record_composite_type_key(key)
        return key

    def _record_vec_op(self, key: str, token: str) -> None:
        self._record_composite_type_key(key)
        self._composite_vec_op_usage.setdefault(key, set()).add(token)

    def _record_mat_op(self, key: str, token: str) -> None:
        self._record_composite_type_key(key)
        self._composite_mat_op_usage.setdefault(key, set()).add(token)

    def _record_vec_unary_math(self, key: str, func_name: str) -> None:
        self._record_composite_type_key(key)
        self._composite_vec_unary_math_usage.setdefault(key, set()).add(func_name)

    def _record_vec_binary_math(self, key: str, func_name: str, signature: str) -> None:
        self._record_composite_type_key(key)
        self._composite_vec_binary_math_usage.setdefault(key, set()).add(f"{func_name}:{signature}")

    def _propagate_matrix_vec_dependencies(self, mat_key: str, token: str) -> None:
        dim = _CUDA_MAT_TYPE_SPECS[mat_key][3]
        vec_key = f"float{dim}"

        if token == "un:-":
            self._record_vec_op(vec_key, "un:-")
            return

        if token.startswith("cmpd:"):
            if token.endswith(":m"):
                vec_token = token[:-1] + "v"
                self._record_vec_op(vec_key, vec_token)
                return
            if token.endswith(":s"):
                self._record_vec_op(vec_key, token)
                return

        if token.startswith("bin:"):
            parts = token.split(":")
            if len(parts) != 3:
                return
            _, op, shape = parts
            if shape == "mm":
                if op in ["+", "-"]:
                    self._record_vec_op(vec_key, f"bin:{op}:vv")
                elif op == "*":
                    self._record_mat_op(mat_key, "bin:*:mv")
                    self._propagate_matrix_vec_dependencies(mat_key, "bin:*:mv")
                return
            if shape == "ms":
                self._record_vec_op(vec_key, f"bin:{op}:vs")
                return
            if shape == "sm":
                self._record_vec_op(vec_key, f"bin:{op}:sv")
                return
            if shape == "mv":
                self._record_vec_op(vec_key, "bin:*:vs")
                self._record_vec_op(vec_key, "bin:+:vv")
                return
            if shape == "vm":
                return

    def mark_composite_unary_op(self, var_type: dtypes.dtype, op: str) -> None:
        key = self._record_composite_type(var_type)
        if key is None:
            return

        token = f"un:{op}"
        if key in _CUDA_VEC_TYPE_SPECS:
            self._record_vec_op(key, token)
            return
        if key in _CUDA_MAT_TYPE_SPECS:
            self._record_mat_op(key, token)
            self._propagate_matrix_vec_dependencies(key, token)

    def mark_composite_binary_op(
        self,
        lhs_type: dtypes.dtype,
        rhs_type: dtypes.dtype,
        op: str,
        *,
        inplace: bool = False,
    ) -> None:
        lhs_key = self._record_composite_type(lhs_type)
        rhs_key = self._record_composite_type(rhs_type)

        lhs_is_composite = lhs_key is not None
        rhs_is_composite = rhs_key is not None
        if not lhs_is_composite and not rhs_is_composite:
            return

        lhs_is_scalar = dtypes.is_scalar(lhs_type)
        rhs_is_scalar = dtypes.is_scalar(rhs_type)

        if lhs_key in _CUDA_VEC_TYPE_SPECS and (rhs_is_scalar or rhs_key in _CUDA_VEC_TYPE_SPECS):
            if inplace:
                suffix = "s" if rhs_is_scalar else "v"
                self._record_vec_op(lhs_key, f"cmpd:{op}=:{suffix}")
                return
            shape = "vs" if rhs_is_scalar else "vv"
            self._record_vec_op(lhs_key, f"bin:{op}:{shape}")
            return

        if rhs_key in _CUDA_VEC_TYPE_SPECS and lhs_is_scalar and not inplace:
            self._record_vec_op(rhs_key, f"bin:{op}:sv")
            return

        if lhs_key in _CUDA_MAT_TYPE_SPECS:
            if inplace:
                if rhs_is_scalar:
                    token = f"cmpd:{op}=:s"
                elif rhs_key in _CUDA_MAT_TYPE_SPECS:
                    token = f"cmpd:{op}=:m"
                else:
                    return
                self._record_mat_op(lhs_key, token)
                self._propagate_matrix_vec_dependencies(lhs_key, token)
                return

            if rhs_is_scalar:
                token = f"bin:{op}:ms"
                self._record_mat_op(lhs_key, token)
                self._propagate_matrix_vec_dependencies(lhs_key, token)
                return

            if rhs_key in _CUDA_MAT_TYPE_SPECS:
                token = "bin:*:mm" if op == "*" else f"bin:{op}:mm"
                self._record_mat_op(lhs_key, token)
                self._propagate_matrix_vec_dependencies(lhs_key, token)
                return

            if rhs_key in _CUDA_VEC_TYPE_SPECS and op == "*":
                token = "bin:*:mv"
                self._record_mat_op(lhs_key, token)
                self._propagate_matrix_vec_dependencies(lhs_key, token)
                return

        if rhs_key in _CUDA_MAT_TYPE_SPECS and lhs_is_scalar and not inplace:
            token = f"bin:{op}:sm"
            self._record_mat_op(rhs_key, token)
            self._propagate_matrix_vec_dependencies(rhs_key, token)
            return

        if lhs_key in _CUDA_VEC_TYPE_SPECS and rhs_key in _CUDA_MAT_TYPE_SPECS and op == "*" and not inplace:
            token = "bin:*:vm"
            self._record_mat_op(rhs_key, token)
            self._propagate_matrix_vec_dependencies(rhs_key, token)

    def mark_texture_sample_dimension(self, dimensions: int) -> None:
        self._sample_texture_dims.add(dimensions)
        self.mark_feature_usage("sample_texture")
        self._record_composite_type_key("float4")
        if dimensions == 2:
            self._record_composite_type_key("float2")
        elif dimensions == 3:
            self._record_composite_type_key("float3")

    def _emit_used_composite_helpers(self) -> str:
        if len(self._composite_type_usage) == 0:
            return ""

        parts: List[str] = []

        # Subgroup helpers use vector binary operators internally (e.g. value = value + shuffled)
        # even if user code never directly emits the corresponding operator on that vector type.
        subgroup_vec_op_requirements = [
            ("subgroup_add", "bin:+:vv"),
            ("subgroup_mul", "bin:*:vv"),
            ("subgroup_and", "bin:&:vv"),
            ("subgroup_or", "bin:|:vv"),
            ("subgroup_xor", "bin:^:vv"),
        ]
        for feature_name, token in subgroup_vec_op_requirements:
            if not self._feature_usage.get(feature_name, False):
                continue
            for key in self._composite_type_usage:
                if key in _CUDA_VEC_TYPE_SPECS:
                    self._composite_vec_op_usage.setdefault(key, set()).add(token)

        vec_order = [
            "short2", "short3", "short4",
            "ushort2", "ushort3", "ushort4",
            "int2", "int3", "int4",
            "uint2", "uint3", "uint4",
            "half2", "half3", "half4",
            "float2", "float3", "float4",
            "double2", "double3", "double4",
        ]
        emitted_vec_keys: Set[str] = set()
        for key in vec_order:
            if key not in self._composite_type_usage:
                continue
            vec_name, scalar_type, dim, cuda_native_type, allow_neg, enable_bitwise = _CUDA_VEC_TYPE_SPECS[key]
            emitted_vec_keys.add(key)
            parts.append(
                _cuda_emit_vec_type(
                    vec_name,
                    scalar_type,
                    dim,
                    cuda_native_type,
                    allow_unary_neg=allow_neg,
                    enable_bitwise=enable_bitwise,
                    needed_ops=self._composite_vec_op_usage.get(key, set()),
                )
            )
            parts.append(_cuda_emit_vec_helper(key, vec_name, scalar_type, dim))
        for key in vec_order:
            if key not in emitted_vec_keys:
                continue
            vec_name, scalar_type, dim, _, _, _ = _CUDA_VEC_TYPE_SPECS[key]
            conversion_helpers = _cuda_emit_vec_wrapper_conversion_helpers(
                key,
                vec_name,
                scalar_type,
                dim,
                available_keys=emitted_vec_keys,
            )
            if len(conversion_helpers) > 0:
                parts.append(conversion_helpers)

        subgroup_shuffle_overloads = _cuda_emit_subgroup_shuffle_xor_vec_overloads(emitted_vec_keys)
        if len(subgroup_shuffle_overloads) > 0:
            parts.append(subgroup_shuffle_overloads)

        mat_order = ["mat2", "mat3", "mat4"]
        for key in mat_order:
            if key not in self._composite_type_usage:
                continue
            mat_name, vec_name, vec_helper_suffix, dim = _CUDA_MAT_TYPE_SPECS[key]
            parts.append(_cuda_emit_mat_type(mat_name, vec_name, dim, self._composite_mat_op_usage.get(key, set())))
            parts.append(_cuda_emit_mat_helpers(mat_name, key, vec_name, vec_helper_suffix, dim))

        vec_math_helpers = self._emit_used_vec_math_helpers()
        if len(vec_math_helpers) > 0:
            parts.append(vec_math_helpers)

        return "\n\n".join(parts)

    @staticmethod
    def _cuda_scalar_unary_math_name(func_name: str, scalar_type: str) -> str:
        """Return the CUDA device-side scalar math function for a given type."""
        if scalar_type == "__half":
            _HALF_MATH = {
                "sin": "hsin", "cos": "hcos", "exp": "hexp", "exp2": "hexp2",
                "log": "hlog", "log2": "hlog2", "sqrt": "hsqrt",
            }
            return _HALF_MATH.get(func_name, func_name)
        if scalar_type == "double":
            return func_name  # standard C math names work for double
        # float  ->  fast intrinsics
        return CUDABackend._cuda_fast_unary_math_name(func_name)

    @staticmethod
    def _cuda_scalar_binary_math_name(func_name: str, scalar_type: str) -> str:
        if scalar_type == "__half":
            return func_name
        if scalar_type == "double":
            return func_name
        return CUDABackend._cuda_fast_binary_math_name(func_name)

    def _emit_used_vec_math_helpers(self) -> str:
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
            unary_funcs = self._composite_vec_unary_math_usage.get(key, set())
            binary_tokens = self._composite_vec_binary_math_usage.get(key, set())
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
                scalar_func = self._cuda_scalar_unary_math_name(func_name, scalar_type)
                comp_args = ", ".join([f"{scalar_func}(v.v.{c})" for c in comps])
                lines.append(
                    f"__device__ __forceinline__ {vec_name} {func_name}(const {vec_name}& v) {{ return vkdispatch_make_{key}({comp_args}); }}"
                )

            for func_name in binary_order:
                scalar_func = self._cuda_scalar_binary_math_name(func_name, scalar_type)
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

    def _emit_sample_texture_helpers(self) -> str:
        dims = set(self._sample_texture_dims)
        if len(dims) == 0:
            dims = {1, 2, 3}

        lines: List[str] = []
        if 1 in dims:
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float coord) { return vkdispatch_make_float4(tex1D<float4>(tex, coord)); }"
            )
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, float coord, float lod) { return vkdispatch_make_float4(tex1DLod<float4>(tex, coord, lod)); }"
            )
            self._record_composite_type_key("float4")
        if 2 in dims:
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, vkdispatch_float2 coord) { return vkdispatch_make_float4(tex2D<float4>(tex, coord.v.x, coord.v.y)); }"
            )
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, vkdispatch_float2 coord, float lod) { return vkdispatch_make_float4(tex2DLod<float4>(tex, coord.v.x, coord.v.y, lod)); }"
            )
            self._record_composite_type_key("float2")
            self._record_composite_type_key("float4")
        if 3 in dims:
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, vkdispatch_float3 coord) { return vkdispatch_make_float4(tex3D<float4>(tex, coord.v.x, coord.v.y, coord.v.z)); }"
            )
            lines.append(
                "__device__ __forceinline__ vkdispatch_float4 vkdispatch_sample_texture(cudaTextureObject_t tex, vkdispatch_float3 coord, float lod) { return vkdispatch_make_float4(tex3DLod<float4>(tex, coord.v.x, coord.v.y, coord.v.z, lod)); }"
            )
            self._record_composite_type_key("float3")
            self._record_composite_type_key("float4")

        return "\n".join(lines)

    def _register_kernel_param(self, param_decl: str) -> None:
        if param_decl not in self._kernel_params:
            self._kernel_params.append(param_decl)

    def _register_alias_line(self, alias_line: str) -> None:
        if alias_line not in self._entry_alias_lines:
            self._entry_alias_lines.append(alias_line)

    @staticmethod
    def _is_plain_integer_literal(expr: str) -> bool:
        if len(expr) == 0:
            return False
        if expr[0] in "+-":
            return len(expr) > 1 and expr[1:].isdigit()
        return expr.isdigit()

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

    def type_name(self, var_type: dtypes.dtype) -> str:
        scalar_name = self._SCALAR_TYPE_NAMES.get(var_type)
        if scalar_name is not None:
            if var_type == dtypes.float16:
                self._needs_cuda_fp16 = True
            return scalar_name

        key = self._composite_key_for_dtype(var_type)
        if key is not None:
            self._record_composite_type(var_type)
            if key in _CUDA_VEC_TYPE_SPECS:
                # Track fp16 header need when half vector types are used.
                if _CUDA_VEC_TYPE_SPECS[key][1] == "__half":
                    self._needs_cuda_fp16 = True
                return _CUDA_VEC_TYPE_SPECS[key][0]
            if key in _CUDA_MAT_TYPE_SPECS:
                return _CUDA_MAT_TYPE_SPECS[key][0]

        raise ValueError(f"Unsupported CUDA type mapping for '{var_type.name}'")

    _FLOAT_VEC_DTYPES = frozenset({
        dtypes.complex32,
        dtypes.complex64,
        dtypes.complex128,
        dtypes.hvec2, dtypes.hvec3, dtypes.hvec4,
        dtypes.vec2, dtypes.vec3, dtypes.vec4,
        dtypes.dvec2, dtypes.dvec3, dtypes.dvec4,
    })

    def constructor(self, var_type: dtypes.dtype, args: List[str]) -> str:
        if (
            len(args) == 1
            and var_type in self._FLOAT_VEC_DTYPES
            and self._is_plain_integer_literal(args[0])
        ):
            scalar_type = None
            if dtypes.is_complex(var_type):
                scalar_type = var_type.child_type
            elif dtypes.is_vector(var_type):
                scalar_type = var_type.scalar

            if scalar_type == dtypes.float64:
                args = [f"{args[0]}.0"]
            else:
                args = [f"{args[0]}.0f"]

        target_type = self.type_name(var_type)

        if dtypes.is_scalar(var_type):
            assert len(args) > 0, f"Constructor for scalar type '{var_type.name}' needs at least one argument."
            return f"(({target_type})({args[0]}))"

        if var_type == dtypes.mat2:
            self.mark_feature_usage("make_mat2")
            return f"vkdispatch_make_mat2({', '.join(args)})"
        if var_type == dtypes.mat3:
            self.mark_feature_usage("make_mat3")
            return f"vkdispatch_make_mat3({', '.join(args)})"
        if var_type == dtypes.mat4:
            self.mark_feature_usage("make_mat4")
            return f"vkdispatch_make_mat4({', '.join(args)})"

        helper_suffix = target_type[len("vkdispatch_"):] if target_type.startswith("vkdispatch_") else target_type
        helper_name = f"vkdispatch_make_{helper_suffix}"
        self.mark_feature_usage(f"make_{helper_suffix}")
        return f"{helper_name}({', '.join(args)})"

    def component_access_expr(self, expr: str, component: str, base_type: dtypes.dtype) -> str:
        if dtypes.is_scalar(base_type):
            if component == "x":
                return expr
            return super().component_access_expr(expr, component, base_type)

        if dtypes.is_vector(base_type) or dtypes.is_complex(base_type):
            direct_builtin_component = self._cuda_builtin_uvec3_component_expr(expr, component, base_type)
            if direct_builtin_component is not None:
                return direct_builtin_component
            return f"{expr}.v.{component}"

        return super().component_access_expr(expr, component, base_type)

    def pre_header(self, *, enable_subgroup_ops: bool, enable_printf: bool) -> str:
        subgroup_support = "1" if enable_subgroup_ops else "0"
        printf_support = "1" if enable_printf else "0"

        self._enable_subgroup_ops = enable_subgroup_ops
        self._enable_printf = enable_printf

        helper_header = self._helper_header()
        fp16_include = "#include <cuda_fp16.h>\n" if self._needs_cuda_fp16 else ""



        self._fixed_preamble = (
            "#include <cuda_runtime.h>\n"
            f"{fp16_include}\n"
            f"#define VKDISPATCH_ENABLE_SUBGROUP_OPS {subgroup_support}\n"
            f"#define VKDISPATCH_ENABLE_PRINTF {printf_support}\n\n"
            f"{helper_header}\n\n"
        )

        return self._fixed_preamble

    def _resolve_helper_dependencies(self, helpers: Set[str]) -> Set[str]:
        pending = list(helpers)
        resolved = set(helpers)

        while len(pending) > 0:
            helper_name = pending.pop()

            for dependency in self._HELPER_DEPENDENCIES.get(helper_name, []):
                if dependency not in resolved:
                    resolved.add(dependency)
                    pending.append(dependency)

        return resolved

    def _helper_header(self) -> str:
        enabled_helpers = {
            helper_name
            for helper_name, is_enabled in self._feature_usage.items()
            if is_enabled
        }

        resolved_helpers = self._resolve_helper_dependencies(enabled_helpers)

        if len(resolved_helpers) == 0:
            return ""

        helper_sections: List[str] = []

        for helper_name in self._HELPER_ORDER:
            if helper_name in resolved_helpers:
                if helper_name == "composite_types":
                    composite_helpers = self._emit_used_composite_helpers()
                    if len(composite_helpers) > 0:
                        helper_sections.append(composite_helpers)
                    continue
                if helper_name == "sample_texture":
                    texture_helpers = self._emit_sample_texture_helpers()
                    if len(texture_helpers) > 0:
                        helper_sections.append(texture_helpers)
                    continue

                snippet = self._HELPER_SNIPPETS[helper_name]
                if len(snippet) > 0:
                    helper_sections.append(snippet)

        return "\n\n".join(helper_sections) + "\n\n"

    def make_source(self, header: str, body: str, x: int, y: int, z: int) -> str:
        header, body = self._finalize_cuda_builtin_uvec3_sentinels(header, body)

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
        return "UBO"

    def exec_bounds_guard(self, exec_count_expr: str) -> str:
        gid = self.global_invocation_id_expr()
        exec_expr = f"({exec_count_expr})"
        gid_expr = f"({gid})"
        return (
            f"if ({self.component_access_expr(exec_expr, 'x', dtypes.uvec4)} <= {self.component_access_expr(gid_expr, 'x', dtypes.uvec3)} || "
            f"{self.component_access_expr(exec_expr, 'y', dtypes.uvec4)} <= {self.component_access_expr(gid_expr, 'y', dtypes.uvec3)} || "
            f"{self.component_access_expr(exec_expr, 'z', dtypes.uvec4)} <= {self.component_access_expr(gid_expr, 'z', dtypes.uvec3)}) {{ return; }}\n"
        )

    def shared_buffer_declaration(self, var_type: dtypes.dtype, name: str, size: int) -> str:
        return f"__shared__ {self.type_name(var_type)} {name}[{size}];"

    def uniform_block_declaration(self, contents: str) -> str:
        self._register_kernel_param("const UniformObjectBuffer* vkdispatch_uniform_ptr")
        self._register_alias_line("const UniformObjectBuffer& UBO = *vkdispatch_uniform_ptr;")
        return f"\nstruct UniformObjectBuffer {{\n{contents}\n}};\n"

    def storage_buffer_declaration(self, binding: int, var_type: dtypes.dtype, name: str) -> str:
        struct_name = f"Buffer{binding}"
        param_name = f"vkdispatch_binding_{binding}_ptr"
        self._register_kernel_param(f"{self.type_name(var_type)}* {param_name}")
        self._register_alias_line(f"{struct_name} {name} = {{{param_name}}};")
        return f"struct {struct_name} {{ {self.type_name(var_type)}* data; }};\n"

    def sampler_declaration(self, binding: int, dimensions: int, name: str) -> str:
        param_name = f"vkdispatch_sampler_{binding}"
        self._register_kernel_param(f"cudaTextureObject_t {param_name}")
        self._register_alias_line(f"cudaTextureObject_t {name} = {param_name};")
        return f"// sampler binding {binding}, dimensions={dimensions}\n"

    def push_constant_declaration(self, contents: str) -> str:
        raise NotImplementedError("Push constants are not supported in the CUDA backend.")

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
        self.mark_feature_usage("uintBitsToFloat")
        return "uintBitsToFloat(0x7F800000u)"

    def ninf_f32_expr(self) -> str:
        self.mark_feature_usage("uintBitsToFloat")
        return "uintBitsToFloat(0xFF800000u)"

    def fma_function_name(self, var_type: dtypes.dtype) -> str:
        if var_type == dtypes.float16:
            return "__hfma"
        if var_type == dtypes.float32:
            return "fmaf"
        return "fma"

    def math_func_name(self, func_name: str, var_type: dtypes.dtype) -> str:
        scalar = var_type
        if dtypes.is_vector(var_type) or dtypes.is_matrix(var_type):
            scalar = var_type.scalar
        elif dtypes.is_complex(var_type):
            scalar = var_type.child_type

        if scalar == dtypes.float16:
            return self._cuda_scalar_unary_math_name(func_name, "__half")
        if scalar == dtypes.float32:
            return self._cuda_fast_unary_math_name(func_name)
        # double and integer types use standard C names
        return func_name

    @staticmethod
    def _cuda_fast_unary_math_name(func_name: str) -> str:
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

    @staticmethod
    def _cuda_fast_binary_math_name(func_name: str) -> str:
        if func_name == "atan2":
            return "atan2f"
        if func_name == "pow":
            return "__powf"

        return func_name

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

    @staticmethod
    def _cuda_float_vec_helper_suffix(var_type: dtypes.dtype) -> Optional[str]:
        return CUDABackend._FLOAT_VEC_HELPER_SUFFIX_MAP.get(var_type)

    @staticmethod
    def _cuda_float_vec_components_for_suffix(helper_suffix: str) -> List[str]:
        # Extract the dimension from the suffix (e.g. "float3" -> 3, "half2" -> 2)
        dim_char = helper_suffix[-1]
        if dim_char == "2":
            return ["x", "y"]
        if dim_char == "3":
            return ["x", "y", "z"]
        if dim_char == "4":
            return ["x", "y", "z", "w"]

        raise ValueError(f"Unsupported CUDA float vector helper suffix '{helper_suffix}'")

    def _cuda_componentwise_unary_math_expr(self, func_name: str, arg_type: dtypes.dtype, arg_expr: str) -> Optional[str]:
        helper_suffix = self._cuda_float_vec_helper_suffix(arg_type)
        if helper_suffix is None:
            return None

        self._record_vec_unary_math(helper_suffix, func_name)
        return f"{func_name}({arg_expr})"

    def _cuda_componentwise_binary_math_expr(
        self,
        func_name: str,
        lhs_type: dtypes.dtype,
        lhs_expr: str,
        rhs_type: dtypes.dtype,
        rhs_expr: str,
    ) -> Optional[str]:
        lhs_helper = self._cuda_float_vec_helper_suffix(lhs_type)
        rhs_helper = self._cuda_float_vec_helper_suffix(rhs_type)

        if lhs_helper is None and rhs_helper is None:
            return None

        if lhs_helper is not None and rhs_helper is not None and lhs_helper != rhs_helper:
            return None

        helper_suffix = lhs_helper if lhs_helper is not None else rhs_helper
        assert helper_suffix is not None

        signature = ("v" if lhs_helper is not None else "s") + ("v" if rhs_helper is not None else "s")
        self._record_vec_binary_math(helper_suffix, func_name, signature)
        return f"{func_name}({lhs_expr}, {rhs_expr})"

    def unary_math_expr(self, func_name: str, arg_type: dtypes.dtype, arg_expr: str) -> str:
        vector_expr = self._cuda_componentwise_unary_math_expr(func_name, arg_type, arg_expr)
        if vector_expr is not None:
            return vector_expr

        mapped = self.math_func_name(func_name, arg_type)
        return f"{mapped}({arg_expr})"

    def binary_math_expr(
        self,
        func_name: str,
        lhs_type: dtypes.dtype,
        lhs_expr: str,
        rhs_type: dtypes.dtype,
        rhs_expr: str,
    ) -> str:
        vector_expr = self._cuda_componentwise_binary_math_expr(
            func_name,
            lhs_type,
            lhs_expr,
            rhs_type,
            rhs_expr,
        )
        if vector_expr is not None:
            return vector_expr

        if func_name == "atan2":
            mapped = self.math_func_name("atan", lhs_type)
            return f"{mapped}({lhs_expr}, {rhs_expr})"

        if dtypes.is_scalar(lhs_type) and dtypes.is_scalar(rhs_type):
            scalar = lhs_type
            scalar_name = self._SCALAR_TYPE_NAMES.get(scalar, "float")
            return f"{self._cuda_scalar_binary_math_name(func_name, scalar_name)}({lhs_expr}, {rhs_expr})"

        return f"{func_name}({lhs_expr}, {rhs_expr})"

    def float_bits_to_int_expr(self, var_expr: str) -> str:
        self.mark_feature_usage("floatBitsToInt")
        return f"floatBitsToInt({var_expr})"

    def float_bits_to_uint_expr(self, var_expr: str) -> str:
        self.mark_feature_usage("floatBitsToUint")
        return f"floatBitsToUint({var_expr})"

    def int_bits_to_float_expr(self, var_expr: str) -> str:
        self.mark_feature_usage("intBitsToFloat")
        return f"intBitsToFloat({var_expr})"

    def uint_bits_to_float_expr(self, var_expr: str) -> str:
        self.mark_feature_usage("uintBitsToFloat")
        return f"uintBitsToFloat({var_expr})"

    def global_invocation_id_expr(self) -> str:
        return self._CUDA_BUILTIN_UVEC3_SENTINELS["global_invocation_id"]["sentinel"]

    def local_invocation_id_expr(self) -> str:
        return self._CUDA_BUILTIN_UVEC3_SENTINELS["local_invocation_id"]["sentinel"]

    def local_invocation_index_expr(self) -> str:
        self.mark_feature_usage("local_invocation_index")
        return "vkdispatch_local_invocation_index()"

    def workgroup_id_expr(self) -> str:
        return self._CUDA_BUILTIN_UVEC3_SENTINELS["workgroup_id"]["sentinel"]

    def workgroup_size_expr(self) -> str:
        self._record_composite_type_key("uint3")
        self.mark_feature_usage("make_uint3")
        return "vkdispatch_make_uint3((unsigned int)blockDim.x, (unsigned int)blockDim.y, (unsigned int)blockDim.z)"

    def num_workgroups_expr(self) -> str:
        self._record_composite_type_key("uint3")
        self.mark_feature_usage("make_uint3")
        return "vkdispatch_make_uint3((unsigned int)gridDim.x, (unsigned int)gridDim.y, (unsigned int)gridDim.z)"

    def num_subgroups_expr(self) -> str:
        self.mark_feature_usage("num_subgroups")
        return "vkdispatch_num_subgroups()"

    def subgroup_id_expr(self) -> str:
        self.mark_feature_usage("subgroup_id")
        return "vkdispatch_subgroup_id()"

    def subgroup_size_expr(self) -> str:
        self.mark_feature_usage("subgroup_size")
        return "vkdispatch_subgroup_size()"

    def subgroup_invocation_id_expr(self) -> str:
        self.mark_feature_usage("subgroup_invocation_id")
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

    @staticmethod
    def _strip_outer_parens(expr: str) -> str:
        stripped = expr.strip()
        while len(stripped) >= 2 and stripped[0] == "(" and stripped[-1] == ")":
            depth = 0
            balanced = True
            for idx, ch in enumerate(stripped):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
                    if depth == 0 and idx != len(stripped) - 1:
                        balanced = False
                        break
            if not balanced or depth != 0:
                break
            stripped = stripped[1:-1].strip()
        return stripped

    def _cuda_builtin_uvec3_component_expr(
        self,
        expr: str,
        component: str,
        base_type: dtypes.dtype,
    ) -> Optional[str]:
        if base_type != dtypes.uvec3 or component not in ("x", "y", "z"):
            return None

        stripped_expr = self._strip_outer_parens(expr)
        for builtin_spec in self._CUDA_BUILTIN_UVEC3_SENTINELS.values():
            if stripped_expr == builtin_spec["sentinel"]:
                return builtin_spec[component]

        return None

    def _finalize_cuda_builtin_uvec3_sentinels(self, header: str, body: str) -> Tuple[str, str]:
        for builtin_spec in self._CUDA_BUILTIN_UVEC3_SENTINELS.values():
            sentinel = builtin_spec["sentinel"]
            if sentinel not in header and sentinel not in body:
                continue

            self._record_composite_type_key("uint3")
            self.mark_feature_usage("make_uint3")
            replacement = (
                "vkdispatch_make_uint3("
                f"{builtin_spec['x']}, {builtin_spec['y']}, {builtin_spec['z']}"
                ")"
            )
            header = header.replace(sentinel, replacement)
            body = body.replace(sentinel, replacement)

        return header, body

    def subgroup_add_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_add")
        return f"vkdispatch_subgroup_add({arg_expr})"

    def subgroup_mul_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_mul")
        return f"vkdispatch_subgroup_mul({arg_expr})"

    def subgroup_min_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_min")
        return f"vkdispatch_subgroup_min({arg_expr})"

    def subgroup_max_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_max")
        return f"vkdispatch_subgroup_max({arg_expr})"

    def subgroup_and_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_and")
        return f"vkdispatch_subgroup_and({arg_expr})"

    def subgroup_or_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_or")
        return f"vkdispatch_subgroup_or({arg_expr})"

    def subgroup_xor_expr(self, arg_expr: str) -> str:
        self.mark_feature_usage("subgroup_xor")
        return f"vkdispatch_subgroup_xor({arg_expr})"

    def subgroup_elect_expr(self) -> str:
        self.mark_feature_usage("subgroup_invocation_id")
        return "((int)(vkdispatch_subgroup_invocation_id() == 0u))"

    def subgroup_barrier_statement(self) -> str:
        return "__syncwarp();"

    def printf_statement(self, fmt: str, args: List[str]) -> str:
        #safe_fmt = fmt.replace("\\", "\\\\").replace('"', '\\"')

        if len(args) == 0:
            return f'printf("{fmt}");'

        return f'printf("{fmt}", {", ".join(args)});'

    def texture_size_expr(self, texture_expr: str, lod: int, dimensions: int) -> str:
        # CUDA texture objects do not expose shape directly in device code.
        # The future CUDA backend should pass explicit texture shape parameters.
        if dimensions == 1:
            return "1.0f"
        if dimensions == 2:
            self.mark_feature_usage("make_float2")
            return "vkdispatch_make_float2(1.0f)"
        if dimensions == 3:
            self.mark_feature_usage("make_float3")
            return "vkdispatch_make_float3(1.0f)"

        raise ValueError(f"Unsupported texture dimensions '{dimensions}'")

    def sample_texture_expr(self, texture_expr: str, coord_expr: str, lod_expr: Optional[str] = None) -> str:
        self.mark_feature_usage("sample_texture")
        if lod_expr is None:
            return f"vkdispatch_sample_texture({texture_expr}, {coord_expr})"

        return f"vkdispatch_sample_texture({texture_expr}, {coord_expr}, {lod_expr})"

    def atomic_add_expr(self, mem_expr: str, value_expr: str, var_type: dtypes.dtype) -> str:
        if var_type not in (dtypes.int32, dtypes.uint32):
            raise NotImplementedError(f"CUDA atomic_add only supports int32/uint32, got '{var_type.name}'")

        return f"atomicAdd(&({mem_expr}), {value_expr})"
