import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from .arithmetic import is_number
from typing import Any, Union, Tuple

from ..global_codegen_callbacks import new_var

import numpy as np

from .common_builtins import dtype_to_floating, resolve_input

def length(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.abs(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return new_var(
        dtype_to_floating(var.var_type),
        f"length({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def distance(x: Any, y: Any) -> Union[BaseVariable, float]:
    if is_number(y) and is_number(x):
        return float(np.abs(y - x))
    
    base_var = None

    if isinstance(y, BaseVariable):
        base_var = y
    elif isinstance(x, BaseVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return new_var(
        dtype_to_floating(base_var.var_type),
        f"distance({resolve_input(x)}, {resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def dot(x: Any, y: Any) -> Union[BaseVariable, float]:
    if is_number(y) and is_number(x):
        return float(np.dot(x, y))
    
    base_var = None

    if isinstance(y, BaseVariable):
        base_var = y
    elif isinstance(x, BaseVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return new_var(
        dtype_to_floating(base_var.var_type),
        f"dot({resolve_input(x)}, {resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def cross(x: BaseVariable, y: BaseVariable) -> BaseVariable:
    assert isinstance(x, BaseVariable), "Argument x must be a ShaderVariable"
    assert isinstance(y, BaseVariable), "Argument y must be a ShaderVariable"

    assert x.var_type == dtypes.vec3, "Argument x must be of type vec3 or dvec3"
    assert y.var_type == dtypes.vec3, "Argument y must be of type vec3 or dvec3"

    return new_var(
        dtypes.vec3,
        f"cross({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def normalize(var: BaseVariable) -> BaseVariable:
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable"

    return new_var(
        var.var_type,
        f"normalize({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )