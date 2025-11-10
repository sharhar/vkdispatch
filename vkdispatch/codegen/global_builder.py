import vkdispatch.base.dtype as dtypes
from .shader_writer import set_global_shader_writer
from .builder import ShaderBuilder
from typing import Optional

class GlobalBuilder:
    obj = ShaderBuilder()

def set_global_builder(builder: ShaderBuilder):
    old_value = GlobalBuilder.obj
    GlobalBuilder.obj = builder  # Update the global reference.
    set_global_shader_writer(builder)
    return old_value

def get_global_builder() -> ShaderBuilder:
    return GlobalBuilder.obj

def shared_buffer(var_type: dtypes.dtype, size: int, var_name: Optional[str] = None):
    return GlobalBuilder.obj.shared_buffer(var_type, size, var_name)

