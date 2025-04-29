import vkdispatch as vd

from .variables import ShaderVariable
from .builder import ShaderBuilder

import contextlib

from typing import List, Union, Optional

inf_f32 = "uintBitsToFloat(0x7F800000)"
ninf_f32 = "uintBitsToFloat(0xFF800000)"

class GlobalBuilder:
    obj = ShaderBuilder()

def set_global_builder(builder: ShaderBuilder):
    old_value = GlobalBuilder.obj
    GlobalBuilder.obj = builder  # Update the global reference.
    return old_value

@contextlib.contextmanager
def builder_context(
    enable_subgroup_ops: bool = True,
    enable_atomic_float_ops: bool = True,
    enable_printf: bool = True,
    enable_exec_bounds: bool = True,
    disable_UBO: bool = False):
    builder = ShaderBuilder(
        enable_atomic_float_ops=enable_atomic_float_ops,
        enable_subgroup_ops=enable_subgroup_ops,
        enable_printf=enable_printf,
        enable_exec_bounds=enable_exec_bounds,
        disable_UBO=disable_UBO
    )
    old_builder = set_global_builder(builder)

    try:
        yield builder
    finally:
        set_global_builder(old_builder)

def comment(text: str):
    GlobalBuilder.obj.comment(text)

def global_invocation():
    return GlobalBuilder.obj.global_invocation

def local_invocation():
    return GlobalBuilder.obj.local_invocation

def workgroup():
    return GlobalBuilder.obj.workgroup

def workgroup_size():
    return GlobalBuilder.obj.workgroup_size

def num_workgroups():
    return GlobalBuilder.obj.num_workgroups

def num_subgroups():
    return GlobalBuilder.obj.num_subgroups

def subgroup_id():
    return GlobalBuilder.obj.subgroup_id

def subgroup_size():
    return GlobalBuilder.obj.subgroup_size

def subgroup_invocation():
    return GlobalBuilder.obj.subgroup_invocation

def set_mapping_index(index: ShaderVariable):
    GlobalBuilder.obj.set_mapping_index(index)

def set_mapping_registers(registers: ShaderVariable):
    GlobalBuilder.obj.set_mapping_registers(registers)

def mapping_index():
    return GlobalBuilder.obj.mapping_index

def mapping_registers():
    return GlobalBuilder.obj.mapping_registers

def shared_buffer(var_type: vd.dtype, size: int, var_name: Optional[str] = None):
    return GlobalBuilder.obj.shared_buffer(var_type, size, var_name)

def abs(arg: ShaderVariable):
    return GlobalBuilder.obj.abs(arg)

def acos(arg: ShaderVariable):
    return GlobalBuilder.obj.acos(arg)

def acosh(arg: ShaderVariable):
    return GlobalBuilder.obj.acosh(arg)

def asin(arg: ShaderVariable):
    return GlobalBuilder.obj.asin(arg)

def asinh(arg: ShaderVariable):
    return GlobalBuilder.obj.asinh(arg)

def atan(arg: ShaderVariable):
    return GlobalBuilder.obj.atan(arg)

def atan2(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.atan2(arg1, arg2)

def atanh(arg: ShaderVariable):
    return GlobalBuilder.obj.atanh(arg)

def atomic_add(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.atomic_add(arg1, arg2)

def barrier():
    GlobalBuilder.obj.barrier()

def ceil(arg: ShaderVariable):
    return GlobalBuilder.obj.ceil(arg)

def clamp(arg: ShaderVariable, min_val: ShaderVariable, max_val: ShaderVariable):
    return GlobalBuilder.obj.clamp(arg, min_val, max_val)

def cos(arg: ShaderVariable):
    return GlobalBuilder.obj.cos(arg)

def cosh(arg: ShaderVariable):
    return GlobalBuilder.obj.cosh(arg)

def cross(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.cross(arg1, arg2)

def degrees(arg: ShaderVariable):
    return GlobalBuilder.obj.degrees(arg)

def determinant(arg: ShaderVariable):
    return GlobalBuilder.obj.determinant(arg)

def distance(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.distance(arg1, arg2)

def dot(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.dot(arg1, arg2)

def exp(arg: ShaderVariable):
    return GlobalBuilder.obj.exp(arg)

def exp2(arg: ShaderVariable):
    return GlobalBuilder.obj.exp2(arg)

def float_bits_to_int(arg: ShaderVariable):
    return GlobalBuilder.obj.float_bits_to_int(arg)

def float_bits_to_uint(arg: ShaderVariable):
    return GlobalBuilder.obj.float_bits_to_uint(arg)

def floor(arg: ShaderVariable):
    return GlobalBuilder.obj.floor(arg)

def fma(arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
    return GlobalBuilder.obj.fma(arg1, arg2, arg3)

def int_bits_to_float(arg: ShaderVariable):
    return GlobalBuilder.obj.int_bits_to_float(arg)

def inverse(arg: ShaderVariable):
    return GlobalBuilder.obj.inverse(arg)

def inverse_sqrt(arg: ShaderVariable):
    return GlobalBuilder.obj.inverse_sqrt(arg)

def isinf(arg: ShaderVariable):
    return GlobalBuilder.obj.isinf(arg)

def isnan(arg: ShaderVariable):
    return GlobalBuilder.obj.isnan(arg)

def length(arg: ShaderVariable):
    return GlobalBuilder.obj.length(arg)

def log(arg: ShaderVariable):
    return GlobalBuilder.obj.log(arg)

def log2(arg: ShaderVariable):
    return GlobalBuilder.obj.log2(arg)

def max(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.max(arg1, arg2)

def memory_barrier():
    GlobalBuilder.obj.memory_barrier()

def memory_barrier_shared():
    GlobalBuilder.obj.memory_barrier_shared()

def min(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.min(arg1, arg2)

def mix(arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
    return GlobalBuilder.obj.mix(arg1, arg2, arg3)

def mod(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.mod(arg1, arg2)

def normalize(arg: ShaderVariable):
    return GlobalBuilder.obj.normalize(arg)

def pow(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.pow(arg1, arg2)

def radians(arg: ShaderVariable):
    return GlobalBuilder.obj.radians(arg)

def round(arg: ShaderVariable):
    return GlobalBuilder.obj.round(arg)

def round_even(arg: ShaderVariable):
    return GlobalBuilder.obj.round_even(arg)

def sign(arg: ShaderVariable):
    return GlobalBuilder.obj.sign(arg)

def sin(arg: ShaderVariable):
    return GlobalBuilder.obj.sin(arg)

def sinh(arg: ShaderVariable):
    return GlobalBuilder.obj.sinh(arg)

def smoothstep(arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
    return GlobalBuilder.obj.smoothstep(arg1, arg2, arg3)

def sqrt(arg: ShaderVariable):
    return GlobalBuilder.obj.sqrt(arg)

def step(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.step(arg1, arg2)

def tan(arg: ShaderVariable):
    return GlobalBuilder.obj.tan(arg)

def tanh(arg: ShaderVariable):
    return GlobalBuilder.obj.tanh(arg)

def transpose(arg: ShaderVariable):
    return GlobalBuilder.obj.transpose(arg)

def trunc(arg: ShaderVariable):
    return GlobalBuilder.obj.trunc(arg)

def uint_bits_to_float(arg: ShaderVariable):
    return GlobalBuilder.obj.uint_bits_to_float(arg)

def mult_c64(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.mult_c64(arg1, arg2)

def mult_c64_by_const(arg1: ShaderVariable, number: complex):
    return GlobalBuilder.obj.mult_c64_by_const(arg1, number)

def mult_conj_c64(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.mult_conj_c64(arg1, arg2)

def if_statement(arg: ShaderVariable, command: Optional[str] = None):
    GlobalBuilder.obj.if_statement(arg, command=command)

def if_any(*args: List[ShaderVariable]):
    GlobalBuilder.obj.if_any(*args)

def if_all(*args: List[ShaderVariable]):
    GlobalBuilder.obj.if_all(*args)

def else_statement():
    GlobalBuilder.obj.else_statement()

def else_if_statement(arg: ShaderVariable):
    GlobalBuilder.obj.else_if_statement(arg)

def else_if_any(*args: List[ShaderVariable]):
    GlobalBuilder.obj.else_if_any(*args)

def else_if_all(*args: List[ShaderVariable]):
    GlobalBuilder.obj.else_if_all(*args)

def return_statement(arg=None):
    GlobalBuilder.obj.return_statement(arg)

def while_statement(arg: ShaderVariable):
    GlobalBuilder.obj.while_statement(arg)

def end():
    GlobalBuilder.obj.end()

def logical_and(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.logical_and(arg1, arg2)

def logical_or(arg1: ShaderVariable, arg2: ShaderVariable):
    return GlobalBuilder.obj.logical_or(arg1, arg2)

def subgroup_add(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_add(arg1)

def subgroup_mul(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_mul(arg1)

def subgroup_min(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_min(arg1)

def subgroup_max(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_max(arg1)

def subgroup_and(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_and(arg1)

def subgroup_or(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_or(arg1)

def subgroup_xor(arg1: ShaderVariable):
    return GlobalBuilder.obj.subgroup_xor(arg1)

def subgroup_elect():
    return GlobalBuilder.obj.subgroup_elect()

def subgroup_barrier():
    GlobalBuilder.obj.subgroup_barrier()

def new(var_type: vd.dtype, *args, var_name: Optional[str] = None):
    return GlobalBuilder.obj.new(var_type, *args, var_name=var_name)

def new_float(*args, var_name: Optional[str] = None):
    return new(vd.float32, *args, var_name=var_name)

def new_int(*args, var_name: Optional[str] = None):
    return new(vd.int32, *args, var_name=var_name)

def new_uint(*args, var_name: Optional[str] = None):
    return new(vd.uint32, *args, var_name=var_name)

def new_vec2(*args, var_name: Optional[str] = None):
    return new(vd.vec2, *args, var_name=var_name)

def new_vec3(*args, var_name: Optional[str] = None):
    return new(vd.vec3, *args, var_name=var_name)

def new_vec4(*args, var_name: Optional[str] = None):
    return new(vd.vec4, *args, var_name=var_name)

def new_uvec2(*args, var_name: Optional[str] = None):
    return new(vd.uvec2, *args, var_name=var_name)

def new_uvec3(*args, var_name: Optional[str] = None):
    return new(vd.uvec3, *args, var_name=var_name)

def new_uvec4(*args, var_name: Optional[str] = None):
    return new(vd.uvec4, *args, var_name=var_name)

def new_ivec2(*args, var_name: Optional[str] = None):
    return new(vd.ivec2, *args, var_name=var_name)

def new_ivec3(*args, var_name: Optional[str] = None):
    return new(vd.ivec3, *args, var_name=var_name)

def new_ivec4(*args, var_name: Optional[str] = None):
    return new(vd.ivec4, *args, var_name=var_name)

def printf(format: str, *args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.printf(format, *args, seperator=seperator)

def print_vars(*args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.print_vars(*args, seperator=seperator)

def unravel_index(index: ShaderVariable, shape: ShaderVariable):
    return GlobalBuilder.obj.unravel_index(index, shape)

def complex_from_euler_angle(angle: ShaderVariable):
    return GlobalBuilder.obj.complex_from_euler_angle(angle)