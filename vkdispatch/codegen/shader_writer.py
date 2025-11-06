import vkdispatch.base.dtype as dtypes
from .variables.base_variable import BaseVariable

from typing import Optional

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

__global_shader_writer: ShaderWriter = None

def set_global_shader_writer(writer: ShaderWriter):
    global __global_shader_writer
    __global_shader_writer = writer

def append_contents(contents: str):
    global __global_shader_writer
    __global_shader_writer.append_contents(contents)

def new_name() -> str:
    global __global_shader_writer
    return __global_shader_writer.new_name()

def scope_increment():
    global __global_shader_writer
    __global_shader_writer.scope_increment()

def scope_decrement():
    global __global_shader_writer
    __global_shader_writer.scope_decrement()

def new_var(var_type: dtypes.dtype,
            var_name: Optional[str],
            parents: list,
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> BaseVariable:
    global __global_shader_writer
    return __global_shader_writer.new_var(var_type, var_name, parents, lexical_unit, settable, register)

def new_scaled_var(var_type: dtypes.dtype,
                     name: str,
                     scale: int = 1,
                     offset: int = 0,
                     parents: list = None):
     global __global_shader_writer
     return __global_shader_writer.new_scaled_var(var_type, name, scale, offset, parents)
