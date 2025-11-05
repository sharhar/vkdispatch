import vkdispatch.base.dtype as dtypes

from .global_codegen_callbacks import set_global_codegen_callbacks

from .builder import ShaderBuilder, ShaderVariable
#from .variables.variables import check_is_int

from typing import List, Union, Optional, Tuple

inf_f32 = "uintBitsToFloat(0x7F800000)"
ninf_f32 = "uintBitsToFloat(0xFF800000)"

class GlobalBuilder:
    obj = ShaderBuilder()

def set_global_builder(builder: ShaderBuilder):
    old_value = GlobalBuilder.obj
    GlobalBuilder.obj = builder  # Update the global reference.

    set_global_codegen_callbacks(
        append_contents=builder.append_contents,
        new_name=builder.new_name,
        new_var=builder.new_var,
        new_scaled_var=builder.new_scaled_var,
    )

    return old_value

def get_global_builder() -> ShaderBuilder:
    return GlobalBuilder.obj

def make_var(var_type: dtypes.dtype,
             var_name: Optional[str],
             parents: List[ShaderVariable],
             lexical_unit: bool = False,
             settable: bool = False) -> ShaderVariable:
    return GlobalBuilder.obj.make_var(var_type, var_name, parents, lexical_unit=lexical_unit, settable=settable)

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

def set_kernel_index(index: ShaderVariable):
    GlobalBuilder.obj.set_kernel_index(index)

def set_mapping_registers(registers: ShaderVariable):
    GlobalBuilder.obj.set_mapping_registers(registers)

def mapping_index():
    return GlobalBuilder.obj.mapping_index

def kernel_index():
    return GlobalBuilder.obj.kernel_index

def mapping_registers():
    return GlobalBuilder.obj.mapping_registers

def shared_buffer(var_type: dtypes.dtype, size: int, var_name: Optional[str] = None):
    return GlobalBuilder.obj.shared_buffer(var_type, size, var_name)

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

def new_scope(indent: bool = True, comment: str = None):
    GlobalBuilder.obj.new_scope(indent=indent, comment=comment)

def end(indent: bool = True):
    GlobalBuilder.obj.end(indent=indent)

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

def printf(format: str, *args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.printf(format, *args, seperator=seperator)

def print_vars(*args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.print_vars(*args, seperator=seperator)


def complex_from_euler_angle(angle: ShaderVariable):
    return GlobalBuilder.obj.complex_from_euler_angle(angle)
