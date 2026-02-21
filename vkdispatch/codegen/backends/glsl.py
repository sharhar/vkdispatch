from typing import List, Optional

import vkdispatch.base.dtype as dtypes

from .base import CodeGenBackend


class GLSLBackend(CodeGenBackend):
    name = "glsl"

    def type_name(self, var_type: dtypes.dtype) -> str:
        return var_type.glsl_type

    def constructor(self, var_type: dtypes.dtype, args: List[str]) -> str:
        return f"{self.type_name(var_type)}({', '.join(args)})"

    def pre_header(self, *, enable_subgroup_ops: bool, enable_printf: bool) -> str:
        header = "#version 450\n"
        header += "#extension GL_EXT_scalar_block_layout : require\n"

        if enable_subgroup_ops:
            header += "#extension GL_KHR_shader_subgroup_arithmetic : require\n"

        if enable_printf:
            header += "#extension GL_EXT_debug_printf : require\n"

        return header

    def make_source(self, header: str, body: str, x: int, y: int, z: int) -> str:
        layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"
        return f"{header}\n{layout_str}\n{body}"

    def constant_namespace(self) -> str:
        return "UBO"

    def variable_namespace(self) -> str:
        return "PC"

    def exec_bounds_guard(self, exec_count_expr: str) -> str:
        return f"if(any(lessThanEqual({exec_count_expr}.xyz, {self.global_invocation_id_expr()}))) {{ return; }}\n"

    def shared_buffer_declaration(self, var_type: dtypes.dtype, name: str, size: int) -> str:
        return f"shared {self.type_name(var_type)} {name}[{size}];"

    def uniform_block_declaration(self, contents: str) -> str:
        return f"\nlayout(set = 0, binding = 0, scalar) uniform UniformObjectBuffer {{\n{contents}\n}} UBO;\n"

    def storage_buffer_declaration(self, binding: int, var_type: dtypes.dtype, name: str) -> str:
        return f"layout(set = 0, binding = {binding}, scalar) buffer Buffer{binding} {{ {self.type_name(var_type)} data[]; }} {name};\n"

    def sampler_declaration(self, binding: int, dimensions: int, name: str) -> str:
        return f"layout(set = 0, binding = {binding}) uniform sampler{dimensions}D {name};\n"

    def push_constant_declaration(self, contents: str) -> str:
        return f"\nlayout(push_constant, scalar) uniform PushConstant {{\n{contents}\n}} PC;\n"

    def entry_point(self, body_contents: str) -> str:
        return f"void main() {{\n{body_contents}}}\n"

    def inf_f32_expr(self) -> str:
        return "uintBitsToFloat(0x7F800000)"

    def ninf_f32_expr(self) -> str:
        return "uintBitsToFloat(0xFF800000)"

    def global_invocation_id_expr(self) -> str:
        return "gl_GlobalInvocationID"

    def local_invocation_id_expr(self) -> str:
        return "gl_LocalInvocationID"

    def local_invocation_index_expr(self) -> str:
        return "gl_LocalInvocationIndex"

    def workgroup_id_expr(self) -> str:
        return "gl_WorkGroupID"

    def workgroup_size_expr(self) -> str:
        return "gl_WorkGroupSize"

    def num_workgroups_expr(self) -> str:
        return "gl_NumWorkGroups"

    def num_subgroups_expr(self) -> str:
        return "gl_NumSubgroups"

    def subgroup_id_expr(self) -> str:
        return "gl_SubgroupID"

    def subgroup_size_expr(self) -> str:
        return "gl_SubgroupSize"

    def subgroup_invocation_id_expr(self) -> str:
        return "gl_SubgroupInvocationID"

    def barrier_statement(self) -> str:
        return "barrier();"

    def memory_barrier_statement(self) -> str:
        return "memoryBarrier();"

    def memory_barrier_buffer_statement(self) -> str:
        return "memoryBarrierBuffer();"

    def memory_barrier_shared_statement(self) -> str:
        return "memoryBarrierShared();"

    def memory_barrier_image_statement(self) -> str:
        return "memoryBarrierImage();"

    def group_memory_barrier_statement(self) -> str:
        return "groupMemoryBarrier();"

    def subgroup_add_expr(self, arg_expr: str) -> str:
        return f"subgroupAdd({arg_expr})"

    def subgroup_mul_expr(self, arg_expr: str) -> str:
        return f"subgroupMul({arg_expr})"

    def subgroup_min_expr(self, arg_expr: str) -> str:
        return f"subgroupMin({arg_expr})"

    def subgroup_max_expr(self, arg_expr: str) -> str:
        return f"subgroupMax({arg_expr})"

    def subgroup_and_expr(self, arg_expr: str) -> str:
        return f"subgroupAnd({arg_expr})"

    def subgroup_or_expr(self, arg_expr: str) -> str:
        return f"subgroupOr({arg_expr})"

    def subgroup_xor_expr(self, arg_expr: str) -> str:
        return f"subgroupXor({arg_expr})"

    def subgroup_elect_expr(self) -> str:
        return "subgroupElect()"

    def subgroup_barrier_statement(self) -> str:
        return "subgroupBarrier();"

    def printf_statement(self, fmt: str, args: List[str]) -> str:
        args_suffix = ""

        if len(args) > 0:
            args_suffix = ", " + ", ".join(args)

        return f'debugPrintfEXT("{fmt}"{args_suffix});'

    def texture_size_expr(self, texture_expr: str, lod: int, dimensions: int) -> str:
        return f"textureSize({texture_expr}, {lod})"

    def sample_texture_expr(self, texture_expr: str, coord_expr: str, lod_expr: Optional[str] = None) -> str:
        if lod_expr is None:
            return f"texture({texture_expr}, {coord_expr})"

        return f"texture({texture_expr}, {coord_expr}, {lod_expr})"
