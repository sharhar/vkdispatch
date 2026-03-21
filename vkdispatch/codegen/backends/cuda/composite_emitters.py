from typing import List, Optional, Set

from .specs import _CUDA_MAT_TYPE_SPECS, _CUDA_VEC_ORDER, _CUDA_VEC_TYPE_SPECS


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
    member_guard = ", ".join([f"(void)(((const TVec*)0)->{c})" for c in comps])
    lines.append(f"    __device__ __forceinline__ {vec_name}() = default;")
    lines.append(f"    __device__ __forceinline__ {vec_name}({ctor_args}) : v{ctor_init} {{}}")
    lines.append(f"    __device__ __forceinline__ explicit {vec_name}({scalar_type} s) : v{splat_init} {{}}")
    lines.append(f"    __device__ __forceinline__ explicit {vec_name}(const {cuda_native_type}& native) : v(native) {{}}")
    lines.append(f"    template <typename TVec, typename = decltype({member_guard})>")
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
    member_guard = ", ".join([f"(void)(((const TVec*)0)->{c})" for c in comps])
    return "\n".join(
        [
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}({args}) {{ return {vec_name}({ctor_args}); }}",
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}({scalar_type} x) {{ return {vec_name}(x); }}",
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}(const {vec_name}& v) {{ return v; }}",
            f"template <typename TVec, typename = decltype({member_guard})>",
            f"__device__ __forceinline__ {vec_name} vkdispatch_make_{helper_suffix}(const TVec& v) {{ return {vec_name}(v); }}",
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
            "cmpd:+=:m",
            "cmpd:+=:s",
            "cmpd:-=:m",
            "cmpd:-=:s",
            "cmpd:*=:s",
            "cmpd:/=:s",
            "bin:+:mm",
            "bin:+:ms",
            "bin:+:sm",
            "bin:-:mm",
            "bin:-:ms",
            "bin:-:sm",
            "bin:*:ms",
            "bin:*:sm",
            "bin:/:ms",
            "bin:/:sm",
            "bin:*:mv",
            "bin:*:vm",
            "bin:*:mm",
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
        lines.append(
            f"    __device__ __forceinline__ {mat_name} operator-() const {{ return {mat_name}({', '.join([f'-c{i}' for i in range(dim)])}); }}"
        )

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

    for key in _CUDA_VEC_ORDER:
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
