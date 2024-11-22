import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

__builder_obj = vc.ShaderBuilder()
builder_obj = __builder_obj

def set_global_builder(builder: vc.ShaderBuilder):
    global builder_obj
    old_value = builder_obj
    builder_obj = builder
    return old_value


global_invocation = builder_obj.global_invocation
local_invocation = builder_obj.local_invocation
workgroup = builder_obj.workgroup
workgroup_size = builder_obj.workgroup_size
num_workgroups = builder_obj.num_workgroups

num_subgroups = builder_obj.num_subgroups
subgroup_id = builder_obj.subgroup_id

subgroup_size = builder_obj.subgroup_size
subgroup_invocation = builder_obj.subgroup_invocation

def shared_buffer(var_type: vd.dtype, size: int, var_name: Optional[str] = None):
    return builder_obj.shared_buffer(var_type, size, var_name)

def memory_barrier():
    builder_obj.memory_barrier()

def memory_barrier_shared():
    builder_obj.memory_barrier_shared()

def barrier():
    builder_obj.barrier()

def if_statement(arg: vc.ShaderVariable):
    builder_obj.if_statement(arg)

def if_any(*args: List[vc.ShaderVariable]):
    builder_obj.if_any(*args)

def if_all(*args: List[vc.ShaderVariable]):
    builder_obj.if_all(*args)

def else_statement():
    builder_obj.else_statement()

def return_statement(arg=None):
    builder_obj.return_statement(arg)

def while_statement(arg: vc.ShaderVariable):
    builder_obj.while_statement(arg)

def length(arg: vc.ShaderVariable):
    return builder_obj.length(arg)

def end():
    builder_obj.end()

def logical_and(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.logical_and(arg1, arg2)

def logical_or(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.logical_or(arg1, arg2)

def ceil(arg: vc.ShaderVariable):
    return builder_obj.ceil(arg)

def floor(arg: vc.ShaderVariable):
    return builder_obj.floor(arg)

def exp(arg: vc.ShaderVariable):
    return builder_obj.exp(arg)

def sin(arg: vc.ShaderVariable):
    return builder_obj.sin(arg)

def cos(arg: vc.ShaderVariable):
    return builder_obj.cos(arg)

def tan(arg: vc.ShaderVariable):
    return builder_obj.tan(arg)

def arctan2(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.arctan2(arg1, arg2)

def sqrt(arg: vc.ShaderVariable):
    return builder_obj.sqrt(arg)

def mod(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.mod(arg1, arg2)

def max(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.max(arg1, arg2)

def min(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.min(arg1, arg2)

def atomic_add(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return builder_obj.atomic_add(arg1, arg2)

def subgroup_add(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_add(arg1)

def subgroup_mul(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_mul(arg1)

def subgroup_min(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_min(arg1)

def subgroup_max(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_max(arg1)

def subgroup_and(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_and(arg1)

def subgroup_or(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_or(arg1)

def subgroup_xor(arg1: vc.ShaderVariable):
    return builder_obj.subgroup_xor(arg1)

def subgroup_elect():
    return builder_obj.subgroup_elect()

def subgroup_barrier():
    builder_obj.subgroup_barrier()

def new(var_type: vd.dtype, *args, var_name: Optional[str] = None):
    return builder_obj.new(var_type, *args, var_name=var_name)

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

def float_bits_to_int(arg: vc.ShaderVariable):
    return builder_obj.float_bits_to_int(arg)

def int_bits_to_float(arg: vc.ShaderVariable):
    return builder_obj.int_bits_to_float(arg)

def printf(format: str, *args: Union[vc.ShaderVariable, str], seperator=" "):
    builder_obj.printf(format, *args, seperator=seperator)

def print_vars(*args: Union[vc.ShaderVariable, str], seperator=" "):
    builder_obj.print_vars(*args, seperator=seperator)

def unravel_index(index: vc.ShaderVariable, shape: vc.ShaderVariable):
    return builder_obj.unravel_index(index, shape)