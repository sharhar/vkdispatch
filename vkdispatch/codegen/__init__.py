from .global_codegen_callbacks import append_contents, new_name

from .arguments import Constant, Variable, ConstantArray, VariableArray
from .arguments import Buffer, Image1D, Image2D, Image3D

from .arguments import _ArgType
from .struct_builder import StructBuilder, StructElement

from .variables.variables import ShaderVariable, SharedBuffer
from .variables.variables import ShaderDescription

from .variables.bound_variables import BufferVariable, ImageVariable, BoundVariable

from .functions.common_builtins import abs, sign, floor, ceil, trunc, round, round_even, comment
from .functions.common_builtins import fract, mod, modf, min, max, clip, clamp, mix
from .functions.common_builtins import step, smoothstep, isnan, isinf, float_bits_to_int
from .functions.common_builtins import float_bits_to_uint, int_bits_to_float, uint_bits_to_float, fma

from .functions.trigonometry import sin, cos, tan, asin, acos, atan, atan2
from .functions.trigonometry import sinh, cosh, tanh, asinh, acosh, atanh, radians, degrees

from .functions.complex_numbers import complex_from_euler_angle

from .functions.exponential import exp, exp2, log, log2, pow, sqrt, inversesqrt

from .functions.geometric import length, distance, dot, cross, normalize

from .functions.block_synchonization import barrier, memory_barrier, memory_barrier_buffer
from .functions.block_synchonization import memory_barrier_shared, memory_barrier_image, group_memory_barrier

from .functions.matrix import matrix_comp_mult, outer_product, transpose
from .functions.matrix import determinant, inverse

from .functions.atomic_memory import atomic_add

from .functions.type_casting import to_dtype, str_to_dtype, to_float, to_int, to_uint
from .functions.type_casting import to_vec2, to_vec3, to_vec4, to_complex
from .functions.type_casting import to_uvec2, to_uvec3, to_uvec4
from .functions.type_casting import to_ivec2, to_ivec3, to_ivec4
from .functions.type_casting import to_mat2, to_mat3, to_mat4

from .functions.registers import new_register, new_float_register, new_int_register, new_uint_register
from .functions.registers import new_vec2_register, new_ivec2_register, new_uvec2_register
from .functions.registers import new_vec3_register, new_ivec3_register, new_uvec3_register
from .functions.registers import new_vec4_register, new_ivec4_register, new_uvec4_register
from .functions.registers import new_mat2_register, new_mat3_register, new_mat4_register

from .functions.subgroups import subgroup_add, subgroup_mul
from .functions.subgroups import subgroup_min, subgroup_max, subgroup_and
from .functions.subgroups import subgroup_or, subgroup_xor, subgroup_elect
from .functions.subgroups import subgroup_barrier

from .functions.control_flow import if_statement, if_any, if_all, else_statement
from .functions.control_flow import else_if_statement, else_if_any, else_if_all
from .functions.control_flow import return_statement, while_statement, new_scope, end
from .functions.control_flow import logical_and, logical_or

from .functions.complex_numbers import mult_complex, mult_complex_conj, complex_conjugate, complex_from_euler_angle
from .functions.complex_numbers import mult_complex_fma, mult_complex_conj_fma

from .functions.builtin_constants import global_invocation_id, local_invocation_id, workgroup_id
from .functions.builtin_constants import workgroup_size, num_workgroups, num_subgroups, subgroup_id
from .functions.builtin_constants import subgroup_size, subgroup_invocation_id, inf_f32, ninf_f32

from .functions.index_raveling import ravel_index, unravel_index

from .builder import ShaderBinding
from .builder import ShaderBuilder, ShaderFlags

from .global_builder import set_global_builder, get_global_builder, make_var

from .global_builder import mapping_index, kernel_index, mapping_registers
from .global_builder import set_kernel_index, set_mapping_index, set_mapping_registers
from .global_builder import printf
from .global_builder import print_vars as print


from .abreviations import *