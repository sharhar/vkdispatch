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

    def type_name(self, var_type: dtypes.dtype) -> str:
        raise NotImplementedError

    def constructor(self, var_type: dtypes.dtype, args: List[str]) -> str:
        raise NotImplementedError

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
