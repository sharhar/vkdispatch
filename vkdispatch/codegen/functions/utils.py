import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import List, Optional

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

def mark_backend_feature(feature_name: str) -> None:
    codegen_backend().mark_feature_usage(feature_name)

def backend_type_name(var_type: dtypes.dtype) -> str:
    return codegen_backend().type_name(var_type)

def _resolve_arg_types(args: tuple) -> List[Optional[dtypes.dtype]]:
    resolved_types: List[Optional[dtypes.dtype]] = []

    for elem in args:
        if isinstance(elem, ShaderVariable):
            resolved_types.append(elem.var_type)
            continue

        if is_number(elem):
            resolved_types.append(number_to_dtype(elem))
            continue

        resolved_types.append(None)

    return resolved_types

def backend_constructor(var_type: dtypes.dtype, *args) -> str:
    resolved_types = _resolve_arg_types(args)
    return codegen_backend().constructor(
        var_type,
        [resolve_input(elem) for elem in args],
        arg_types=resolved_types,
    )

def backend_constructor_from_resolved(
    var_type: dtypes.dtype,
    args: List[str],
    arg_types: Optional[List[Optional[dtypes.dtype]]] = None,
) -> str:
    return codegen_backend().constructor(var_type, args, arg_types=arg_types)
