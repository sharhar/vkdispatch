import vkdispatch.base.dtype as dtypes

from .variables.base_variable import BaseVariable

from typing import Callable, List

__append_contents: Callable[[str], None] = None
__new_name: Callable[[], str] = None
__new_var: Callable[[dtypes.dtype, str, List, bool, bool, bool], BaseVariable] = None
__new_scaled_var: Callable[[dtypes.dtype, str, int, int, List], BaseVariable] = None

def set_global_codegen_callbacks(append_contents: Callable[[str], None],
                                 new_name: Callable[[], str],
                                 new_var: Callable[[dtypes.dtype, str, List, bool, bool, bool], BaseVariable],
                                 new_scaled_var: Callable[[dtypes.dtype, str, int, int, List], BaseVariable]):
    global __append_contents, __new_name
    global __new_var, __new_scaled_var
    __append_contents = append_contents
    __new_name = new_name
    __new_var = new_var
    __new_scaled_var = new_scaled_var

def append_contents(contents: str):
    global __append_contents
    __append_contents(contents)

def new_name() -> str:
    global __new_name
    return __new_name()

def new_var(var_type: dtypes.dtype,
            var_name: str,
            parents: List[BaseVariable],
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> BaseVariable:
    global __new_var
    return __new_var(var_type, var_name, parents, lexical_unit, settable, register)

def new_scaled_var(var_type: dtypes.dtype,
                   name: str,
                   scale: int = 1,
                   offset: int = 0,
                   parents: List[BaseVariable] = None):
    global __new_scaled_var
    return __new_scaled_var(var_type, name, scale, offset, parents)