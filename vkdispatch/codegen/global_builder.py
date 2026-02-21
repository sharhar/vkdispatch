import threading
import vkdispatch.base.dtype as dtypes
from .shader_writer import set_shader_writer
from .builder import ShaderBuilder
from typing import Optional

_builder_context = threading.local()

def _get_builder() -> Optional['ShaderBuilder']:
    return getattr(_builder_context, 'active_builder', None)

def set_builder(builder: ShaderBuilder):
    if builder is None:
        _builder_context.active_builder = None
        set_shader_writer(None)
        return

    assert _get_builder() is None, "A global ShaderBuilder is already set for the current thread!"
    set_shader_writer(builder)
    _builder_context.active_builder = builder

def get_builder() -> ShaderBuilder:
    builder = _get_builder()
    assert builder is not None, "No global ShaderBuilder is set for the current thread!"
    return builder

def shared_buffer(var_type: dtypes.dtype, size: int, var_name: Optional[str] = None):
    return get_builder().shared_buffer(var_type, size, var_name)

