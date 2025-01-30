import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

class GlobalBuilder:
    obj = vc.ShaderBuilder()

def set_global_builder(builder: vc.ShaderBuilder):
    old_value = GlobalBuilder.obj
    GlobalBuilder.obj = builder  # Update the global reference.
    return old_value

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

def shared_buffer(var_type: vd.dtype, size: int, var_name: Optional[str] = None):
    return GlobalBuilder.obj.shared_buffer(var_type, size, var_name)

def memory_barrier():
    GlobalBuilder.obj.memory_barrier()

def memory_barrier_shared():
    GlobalBuilder.obj.memory_barrier_shared()

def barrier():
    GlobalBuilder.obj.barrier()

def if_statement(arg: vc.ShaderVariable):
    GlobalBuilder.obj.if_statement(arg)

def if_any(*args: List[vc.ShaderVariable]):
    GlobalBuilder.obj.if_any(*args)

def if_all(*args: List[vc.ShaderVariable]):
    GlobalBuilder.obj.if_all(*args)

def else_statement():
    GlobalBuilder.obj.else_statement()

def return_statement(arg=None):
    GlobalBuilder.obj.return_statement(arg)

def while_statement(arg: vc.ShaderVariable):
    GlobalBuilder.obj.while_statement(arg)

def length(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.length(arg)

def end():
    GlobalBuilder.obj.end()

def logical_and(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.logical_and(arg1, arg2)

def logical_or(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.logical_or(arg1, arg2)

def ceil(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.ceil(arg)

def floor(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.floor(arg)

def abs(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.abs(arg)

def exp(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.exp(arg)

def sin(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.sin(arg)

def cos(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.cos(arg)

def tan(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.tan(arg)

def arctan2(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.arctan2(arg1, arg2)

def sqrt(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.sqrt(arg)

def mod(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.mod(arg1, arg2)

def max(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.max(arg1, arg2)

def min(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.min(arg1, arg2)

def log(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.log(arg)

def log2(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.log2(arg)

def atomic_add(arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
    return GlobalBuilder.obj.atomic_add(arg1, arg2)

def subgroup_add(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_add(arg1)

def subgroup_mul(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_mul(arg1)

def subgroup_min(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_min(arg1)

def subgroup_max(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_max(arg1)

def subgroup_and(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_and(arg1)

def subgroup_or(arg1: vc.ShaderVariable):
    return GlobalBuilder.obj.subgroup_or(arg1)

def subgroup_xor(arg1: vc.ShaderVariable):
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

def float_bits_to_int(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.float_bits_to_int(arg)

def int_bits_to_float(arg: vc.ShaderVariable):
    return GlobalBuilder.obj.int_bits_to_float(arg)

def printf(format: str, *args: Union[vc.ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.printf(format, *args, seperator=seperator)

def print_vars(*args: Union[vc.ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.print_vars(*args, seperator=seperator)

def unravel_index(index: vc.ShaderVariable, shape: vc.ShaderVariable):
    return GlobalBuilder.obj.unravel_index(index, shape)