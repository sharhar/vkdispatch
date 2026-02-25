import threading
import vkdispatch.base.dtype as dtypes
from .shader_writer import set_shader_writer
from .backends import CodeGenBackend, GLSLBackend, CUDABackend, OpenCLBackend
from vkdispatch.base.init import is_cuda
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .builder import ShaderBuilder

_builder_context = threading.local()
_shader_print_line_numbers = threading.local()
_codegen_backend = threading.local()

def _make_runtime_default_codegen_backend() -> CodeGenBackend:
    if is_cuda():
        return CUDABackend()

    return GLSLBackend()

def get_shader_print_line_numbers() -> bool:
    return getattr(_shader_print_line_numbers, 'value', False)

def set_shader_print_line_numbers(value: bool):
    _shader_print_line_numbers.value = value

def _get_builder() -> Optional['ShaderBuilder']:
    return getattr(_builder_context, 'active_builder', None)

def _get_codegen_backend() -> Optional[CodeGenBackend]:
    return getattr(_codegen_backend, 'active_backend', None)

def set_codegen_backend(backend: Optional[Union[CodeGenBackend, str]]):
    if backend is None:
        _codegen_backend.active_backend = None
        return

    if isinstance(backend, str):
        backend_name = backend.lower()

        if backend_name == "glsl":
            _codegen_backend.active_backend = GLSLBackend()
            return

        if backend_name == "cuda":
            _codegen_backend.active_backend = CUDABackend()
            return

        if backend_name == "opencl":
            _codegen_backend.active_backend = OpenCLBackend()
            return

        raise ValueError(f"Unknown codegen backend '{backend}'")

    _codegen_backend.active_backend = backend

def get_codegen_backend() -> CodeGenBackend:
    builder = _get_builder()

    if builder is not None:
        return builder.backend

    backend = _get_codegen_backend()

    if backend is None:
        backend = _make_runtime_default_codegen_backend()
        _codegen_backend.active_backend = backend

    return backend

def set_builder(builder: 'ShaderBuilder'):
    if builder is None:
        _builder_context.active_builder = None
        set_shader_writer(None)
        return

    assert _get_builder() is None, "A global ShaderBuilder is already set for the current thread!"
    set_shader_writer(builder)
    _builder_context.active_builder = builder

def get_builder() -> 'ShaderBuilder':
    builder = _get_builder()
    assert builder is not None, "No global ShaderBuilder is set for the current thread!"
    return builder

def shared_buffer(var_type: dtypes.dtype, size: int, var_name: Optional[str] = None):
    return get_builder().shared_buffer(var_type, size, var_name)
