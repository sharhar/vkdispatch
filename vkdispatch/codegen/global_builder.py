import vkdispatch.base.dtype as dtypes

from .shader_writer import set_global_shader_writer

from .functions.type_casting import to_dtype, str_to_dtype

from .builder import ShaderBuilder, ShaderVariable

from typing import List, Union, Optional, Tuple

class GlobalBuilder:
    obj = ShaderBuilder()

def set_global_builder(builder: ShaderBuilder):
    old_value = GlobalBuilder.obj
    GlobalBuilder.obj = builder  # Update the global reference.
    set_global_shader_writer(builder)
    return old_value

def get_global_builder() -> ShaderBuilder:
    return GlobalBuilder.obj

def make_var(var_type: dtypes.dtype,
             var_name: Optional[str],
             parents: List[ShaderVariable],
             lexical_unit: bool = False,
             settable: bool = False) -> ShaderVariable:
    return GlobalBuilder.obj.make_var(var_type, var_name, parents, lexical_unit=lexical_unit, settable=settable)

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

def printf(format: str, *args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.printf(format, *args, seperator=seperator)

def print_vars(*args: Union[ShaderVariable, str], seperator=" "):
    GlobalBuilder.obj.print_vars(*args, seperator=seperator)
