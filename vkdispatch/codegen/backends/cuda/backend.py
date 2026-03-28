from typing import Dict, List, Optional, Set, Tuple

import vkdispatch.base.dtype as dtypes

from ..base import CodeGenBackend
from .composite_emitters import (
    _cuda_emit_mat_helpers,
    _cuda_emit_mat_type,
    _cuda_emit_subgroup_shuffle_xor_vec_overloads,
    _cuda_emit_vec_helper,
    _cuda_emit_vec_type,
    _cuda_emit_vec_wrapper_conversion_helpers,
)
from .helper_snippets import (
    _HELPER_DEPENDENCIES as _CUDA_HELPER_DEPENDENCIES,
    _HELPER_ORDER as _CUDA_HELPER_ORDER,
    _HELPER_SNIPPETS as _CUDA_HELPER_SNIPPETS,
    initialize_feature_usage,
)
from .math_utils import (
    cuda_fast_binary_math_name,
    cuda_fast_unary_math_name,
    cuda_float_vec_components_for_suffix,
    cuda_float_vec_helper_suffix,
    cuda_scalar_binary_math_name,
    cuda_scalar_unary_math_name,
    emit_used_vec_math_helpers,
)
from .specs import (
    _CUDA_MAT_ORDER,
    _CUDA_MAT_TYPE_SPECS,
    _CUDA_VEC_ORDER,
    _CUDA_VEC_TYPE_SPECS,
    _DTYPE_TO_COMPOSITE_KEY as _CUDA_DTYPE_TO_COMPOSITE_KEY,
    _FLOAT_VEC_DTYPES as _CUDA_FLOAT_VEC_DTYPES,
    _FLOAT_VEC_HELPER_SUFFIX_MAP as _CUDA_FLOAT_VEC_HELPER_SUFFIX_MAP,
    _SCALAR_TYPE_NAMES as _CUDA_SCALAR_TYPE_NAMES,
)

class CUDABackend(CodeGenBackend):
    name = "cuda"
    _SUBGROUP_FEATURE_NAMES = {
        "num_subgroups",
        "subgroup_id",
        "subgroup_size",
        "subgroup_invocation_id",
        "subgroup_add",
        "subgroup_mul",
        "subgroup_min",
        "subgroup_max",
        "subgroup_and",
        "subgroup_or",
        "subgroup_xor",
    }
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

    _HELPER_SNIPPETS: Dict[str, str] = _CUDA_HELPER_SNIPPETS
    _HELPER_ORDER: List[str] = _CUDA_HELPER_ORDER
    _HELPER_DEPENDENCIES: Dict[str, List[str]] = _CUDA_HELPER_DEPENDENCIES

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
        self._feature_usage: Dict[str, bool] = initialize_feature_usage()
        self._printf_used: bool = False

    def mark_feature_usage(self, feature_name: str) -> None:
        if feature_name in self._feature_usage:
            self._feature_usage[feature_name] = True

    def uses_feature(self, feature_name: str) -> bool:
        if feature_name == "subgroup_ops":
            return any(
                self._feature_usage.get(name, False)
                for name in self._SUBGROUP_FEATURE_NAMES
            )

        if feature_name == "printf":
            return self._printf_used

        return False

    _DTYPE_TO_COMPOSITE_KEY = _CUDA_DTYPE_TO_COMPOSITE_KEY

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

        emitted_vec_keys: Set[str] = set()
        for key in _CUDA_VEC_ORDER:
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
        for key in _CUDA_VEC_ORDER:
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

        for key in _CUDA_MAT_ORDER:
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
        return cuda_scalar_unary_math_name(func_name, scalar_type)

    @staticmethod
    def _cuda_scalar_binary_math_name(func_name: str, scalar_type: str) -> str:
        return cuda_scalar_binary_math_name(func_name, scalar_type)

    def _emit_used_vec_math_helpers(self) -> str:
        return emit_used_vec_math_helpers(
            self._composite_vec_unary_math_usage,
            self._composite_vec_binary_math_usage,
        )

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

    _SCALAR_TYPE_NAMES = _CUDA_SCALAR_TYPE_NAMES

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

    _FLOAT_VEC_DTYPES = _CUDA_FLOAT_VEC_DTYPES

    def constructor(
        self,
        var_type: dtypes.dtype,
        args: List[str],
        arg_types: Optional[List[Optional[dtypes.dtype]]] = None,
    ) -> str:
        _ = arg_types
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
            if len(args) == 0:
                raise ValueError(f"Constructor for scalar type '{var_type.name}' needs at least one argument.")
            
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
        return "PC"

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
        self._register_kernel_param("const UniformObjectBuffer vkdispatch_uniform_value")
        self._register_alias_line("const UniformObjectBuffer& UBO = vkdispatch_uniform_value;")
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
        self._register_kernel_param("const PushConstant vkdispatch_pc_value")
        self._register_alias_line("const PushConstant& PC = vkdispatch_pc_value;")
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
        self.mark_feature_usage("uintBitsToFloat")
        return "uintBitsToFloat(0x7F800000u)"

    def ninf_f32_expr(self) -> str:
        self.mark_feature_usage("uintBitsToFloat")
        return "uintBitsToFloat(0xFF800000u)"

    def inf_f64_expr(self) -> str:
        self.mark_feature_usage("longlong_as_double")
        return "__longlong_as_double(0x7FF0000000000000LL)"

    def ninf_f64_expr(self) -> str:
        self.mark_feature_usage("longlong_as_double")
        return "__longlong_as_double(0xFFF0000000000000LL)"

    def inf_f16_expr(self) -> str:
        self.mark_feature_usage("ushort_as_half")
        return "__ushort_as_half(0x7C00u)"

    def ninf_f16_expr(self) -> str:
        self.mark_feature_usage("ushort_as_half")
        return "__ushort_as_half(0xFC00u)"

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
        return cuda_fast_unary_math_name(func_name)

    @staticmethod
    def _cuda_fast_binary_math_name(func_name: str) -> str:
        return cuda_fast_binary_math_name(func_name)

    _FLOAT_VEC_HELPER_SUFFIX_MAP = _CUDA_FLOAT_VEC_HELPER_SUFFIX_MAP

    @staticmethod
    def _cuda_float_vec_helper_suffix(var_type: dtypes.dtype) -> Optional[str]:
        return cuda_float_vec_helper_suffix(var_type)

    @staticmethod
    def _cuda_float_vec_components_for_suffix(helper_suffix: str) -> List[str]:
        return cuda_float_vec_components_for_suffix(helper_suffix)

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

        if helper_suffix is None:
            raise ValueError("At least one of the argument types should have a float vector helper suffix")

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

    def subgroup_add_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_add")
        return f"vkdispatch_subgroup_add({arg_expr})"

    def subgroup_mul_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_mul")
        return f"vkdispatch_subgroup_mul({arg_expr})"

    def subgroup_min_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_min")
        return f"vkdispatch_subgroup_min({arg_expr})"

    def subgroup_max_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_max")
        return f"vkdispatch_subgroup_max({arg_expr})"

    def subgroup_and_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_and")
        return f"vkdispatch_subgroup_and({arg_expr})"

    def subgroup_or_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_or")
        return f"vkdispatch_subgroup_or({arg_expr})"

    def subgroup_xor_expr(self, arg_expr: str, arg_type: Optional[dtypes.dtype] = None) -> str:
        _ = arg_type
        self.mark_feature_usage("subgroup_xor")
        return f"vkdispatch_subgroup_xor({arg_expr})"

    def subgroup_elect_expr(self) -> str:
        self.mark_feature_usage("subgroup_invocation_id")
        return "((int)(vkdispatch_subgroup_invocation_id() == 0u))"

    def subgroup_barrier_statement(self) -> str:
        return "__syncwarp();"

    def printf_statement(self, fmt: str, args: List[str]) -> str:
        self._printf_used = True
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
        raise NotImplementedError("Direct texture sampling is not supported in CUDA backend. Use vkdispatch_sample_texture helper functions instead.")

    def atomic_add_expr(self, mem_expr: str, value_expr: str, var_type: dtypes.dtype) -> str:
        if var_type not in (dtypes.int32, dtypes.uint32):
            raise NotImplementedError(f"CUDA atomic_add only supports int32/uint32, got '{var_type.name}'")

        return f"atomicAdd(&({mem_expr}), {value_expr})"
