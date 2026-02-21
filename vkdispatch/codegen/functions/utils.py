import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import List

from .base_functions.base_utils import *
from ..global_builder import get_codegen_backend

from ..shader_writer import scope_increment, scope_decrement

def new_var(var_type: dtypes.dtype,
            var_name: Optional[str],
            parents: list,
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> ShaderVariable:
    return new_base_var(var_type, var_name, parents, lexical_unit, settable, register)

def codegen_backend():
    return get_codegen_backend()

def backend_type_name(var_type: dtypes.dtype) -> str:
    return codegen_backend().type_name(var_type)

def backend_constructor(var_type: dtypes.dtype, *args) -> str:
    return codegen_backend().constructor(
        var_type,
        [resolve_input(elem) for elem in args]
    )

def backend_constructor_from_resolved(var_type: dtypes.dtype, args: List[str]) -> str:
    return codegen_backend().constructor(var_type, args)
