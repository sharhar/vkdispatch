import threading
import vkdispatch.base.dtype as dtypes
from .variables.base_variable import BaseVariable
from typing import Optional

_thread_context = threading.local()

def _get_shader_writer() -> Optional['ShaderWriter']:
    return getattr(_thread_context, 'writer', None)

def get_shader_writer() -> 'ShaderWriter':
    writer = _get_shader_writer()
    assert writer is not None, "No global ShaderWriter is set for the current thread!"
    return writer

def set_shader_writer(writer: 'ShaderWriter'):
    if writer is None:
        _thread_context.writer = None
        return

    assert _get_shader_writer() is None, "A global ShaderWriter is already set for the current thread!"
    _thread_context.writer = writer

class ShaderWriter:
    var_count: int
    contents: str
    scope_num: int

    def __init__(self):
        self.var_count = 0
        self.scope_num = 1
        self.contents = ""

    def append_contents(self, contents: str) -> None:
        self.contents += ("    " * self.scope_num) + contents

    def new_name(self) -> str:
        new_var = f"var{self.var_count}"
        self.var_count += 1
        return new_var
    
    def scope_increment(self):
        self.scope_num += 1
    
    def scope_decrement(self):
        self.scope_num -= 1

    def new_var(self,
                var_type: dtypes.dtype,
                var_name: str,
                parents: list,
                lexical_unit: bool = False,
                settable: bool = False,
                register: bool = False) -> BaseVariable:
        raise NotImplementedError
    
    def new_scaled_var(self,
                        var_type: dtypes.dtype,
                        name: str,
                        scale: int = 1,
                        offset: int = 0,
                        parents: list = None):
        raise NotImplementedError

def append_contents(contents: str):
    get_shader_writer().append_contents(contents)

def new_name() -> str:
    return get_shader_writer().new_name()

def scope_increment():
    get_shader_writer().scope_increment()

def scope_decrement():
    get_shader_writer().scope_decrement()

def new_var(var_type: dtypes.dtype,
            var_name: Optional[str],
            parents: list,
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> BaseVariable:
    return get_shader_writer().new_var(var_type, var_name, parents, lexical_unit, settable, register)

def new_scaled_var(var_type: dtypes.dtype,
                     name: str,
                     scale: int = 1,
                     offset: int = 0,
                     parents: list = None):
     return get_shader_writer().new_scaled_var(var_type, name, scale, offset, parents)
