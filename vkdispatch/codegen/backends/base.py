from typing import List, Optional

import vkdispatch.base.dtype as dtypes


class CodeGenBackend:
    """
    Interface for backend-specific code generation.

    Subclasses should override all methods that are used by the codegen
    pipeline. The base implementation raises NotImplementedError so placeholder
    backends can be defined incrementally.
    """

    name: str = "base"

    def reset_state(self) -> None:
        # Stateless backends can ignore this.
        return

    def mark_feature_usage(self, feature_name: str) -> None:
        # Backends that emit optional helper code can override this.
        return

    def mark_composite_unary_op(self, var_type: dtypes.dtype, op: str) -> None:
        # Backends with composite helper/operator code can override this.
        return

    def mark_composite_binary_op(
        self,
        lhs_type: dtypes.dtype,
        rhs_type: dtypes.dtype,
        op: str,
        *,
        inplace: bool = False,
    ) -> None:
        # Backends with composite helper/operator code can override this.
        return

    def type_name(self, var_type: dtypes.dtype) -> str:
        raise NotImplementedError

    def constructor(self, var_type: dtypes.dtype, args: List[str]) -> str:
        raise NotImplementedError

    def component_access_expr(self, expr: str, component: str, base_type: dtypes.dtype) -> str:
        return f"{expr}.{component}"

    def fma_function_name(self, var_type: dtypes.dtype) -> str:
        return "fma"

    def math_func_name(self, func_name: str, var_type: dtypes.dtype) -> str:
        """Return the backend-specific function name for a math operation.

        Backends can override this to remap function names for specific types
        (e.g. CUDA __half intrinsics).
        """
        return func_name

    def unary_math_expr(self, func_name: str, arg_type: dtypes.dtype, arg_expr: str) -> str:
        return f"{self.math_func_name(func_name, arg_type)}({arg_expr})"

    def binary_math_expr(
        self,
        func_name: str,
        lhs_type: dtypes.dtype,
        lhs_expr: str,
        rhs_type: dtypes.dtype,
        rhs_expr: str,
    ) -> str:
        mapped = self.math_func_name(func_name, lhs_type)
        if func_name == "atan2":
            mapped_atan = self.math_func_name("atan", lhs_type)
            return f"{mapped_atan}({lhs_expr}, {rhs_expr})"

        return f"{mapped}({lhs_expr}, {rhs_expr})"

    def arithmetic_unary_expr(self, op: str, var_type: dtypes.dtype, var_expr: str) -> Optional[str]:
        """Optional backend override for unary arithmetic expressions."""
        _ = (op, var_type, var_expr)
        return None

    def arithmetic_binary_expr(
        self,
        op: str,
        lhs_type: dtypes.dtype,
        lhs_expr: str,
        rhs_type: dtypes.dtype,
        rhs_expr: str,
    ) -> Optional[str]:
        """Optional backend override for binary arithmetic expressions."""
        _ = (op, lhs_type, lhs_expr, rhs_type, rhs_expr)
        return None

    def pre_header(self, *, enable_subgroup_ops: bool, enable_printf: bool) -> str:
        raise NotImplementedError

    def make_source(self, header: str, body: str, x: int, y: int, z: int) -> str:
        raise NotImplementedError

    def constant_namespace(self) -> str:
        raise NotImplementedError

    def variable_namespace(self) -> str:
        raise NotImplementedError

    def exec_bounds_guard(self, exec_count_expr: str) -> str:
        raise NotImplementedError

    def shared_buffer_declaration(self, var_type: dtypes.dtype, name: str, size: int) -> str:
        raise NotImplementedError

    def uniform_block_declaration(self, contents: str) -> str:
        raise NotImplementedError

    def storage_buffer_declaration(self, binding: int, var_type: dtypes.dtype, name: str) -> str:
        raise NotImplementedError

    def sampler_declaration(self, binding: int, dimensions: int, name: str) -> str:
        raise NotImplementedError

    def push_constant_declaration(self, contents: str) -> str:
        raise NotImplementedError

    def entry_point(self, body_contents: str) -> str:
        raise NotImplementedError

    def inf_f32_expr(self) -> str:
        raise NotImplementedError

    def ninf_f32_expr(self) -> str:
        raise NotImplementedError

    def float_bits_to_int_expr(self, var_expr: str) -> str:
        raise NotImplementedError

    def float_bits_to_uint_expr(self, var_expr: str) -> str:
        raise NotImplementedError

    def int_bits_to_float_expr(self, var_expr: str) -> str:
        raise NotImplementedError

    def uint_bits_to_float_expr(self, var_expr: str) -> str:
        raise NotImplementedError

    def global_invocation_id_expr(self) -> str:
        raise NotImplementedError

    def local_invocation_id_expr(self) -> str:
        raise NotImplementedError

    def local_invocation_index_expr(self) -> str:
        raise NotImplementedError

    def workgroup_id_expr(self) -> str:
        raise NotImplementedError

    def workgroup_size_expr(self) -> str:
        raise NotImplementedError

    def num_workgroups_expr(self) -> str:
        raise NotImplementedError

    def num_subgroups_expr(self) -> str:
        raise NotImplementedError

    def subgroup_id_expr(self) -> str:
        raise NotImplementedError

    def subgroup_size_expr(self) -> str:
        raise NotImplementedError

    def subgroup_invocation_id_expr(self) -> str:
        raise NotImplementedError

    def barrier_statement(self) -> str:
        raise NotImplementedError

    def memory_barrier_statement(self) -> str:
        raise NotImplementedError

    def memory_barrier_buffer_statement(self) -> str:
        raise NotImplementedError

    def memory_barrier_shared_statement(self) -> str:
        raise NotImplementedError

    def memory_barrier_image_statement(self) -> str:
        raise NotImplementedError

    def group_memory_barrier_statement(self) -> str:
        raise NotImplementedError

    def subgroup_add_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_mul_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_min_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_max_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_and_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_or_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_xor_expr(self, arg_expr: str) -> str:
        raise NotImplementedError

    def subgroup_elect_expr(self) -> str:
        raise NotImplementedError

    def subgroup_barrier_statement(self) -> str:
        raise NotImplementedError

    def printf_statement(self, fmt: str, args: List[str]) -> str:
        raise NotImplementedError

    def texture_size_expr(self, texture_expr: str, lod: int, dimensions: int) -> str:
        raise NotImplementedError

    def sample_texture_expr(self, texture_expr: str, coord_expr: str, lod_expr: Optional[str] = None) -> str:
        raise NotImplementedError

    def mark_texture_sample_dimension(self, dimensions: int) -> None:
        return

    def atomic_add_expr(self, mem_expr: str, value_expr: str, var_type: dtypes.dtype) -> str:
        raise NotImplementedError(
            f"atomic_add is not supported for backend '{self.name}' and type '{var_type.name}'"
        )
