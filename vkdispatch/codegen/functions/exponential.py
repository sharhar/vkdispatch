import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from .arithmetic import is_number
from typing import Any, Union

import numpy as np

from .trigonometry import dtype_to_floating

def pow(x: Any, y: Any) -> Union[BaseVariable, float]:
    if is_number(y) and is_number(x):
        return float(np.power(x, y))
    
    if is_number(x) and isinstance(y, BaseVariable):
        return y.new_var(
            dtype_to_floating(y.var_type),
            f"pow({x}, {y.resolve()})",
            parents=[y]
        )
    
    if is_number(y) and isinstance(x, BaseVariable):
        return x.new_var(
            dtype_to_floating(x.var_type),
            f"pow({x.resolve()}, {y})",
            parents=[x]
        )

    assert isinstance(y, BaseVariable), "First argument must be a ShaderVariable or number"
    assert isinstance(x, BaseVariable), "Second argument must be a ShaderVariable or number"

    return y.new_var(
        dtype_to_floating(y.var_type),
        f"pow({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def exp(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.exp(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"exp({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def exp2(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.exp2(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"exp2({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def log(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.log(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"log({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def log2(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.log2(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"log2({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def sqrt(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(np.sqrt(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"sqrt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def inversesqrt(var: Any) -> Union[BaseVariable, float]:
    if is_number(var):
        return float(1.0 / np.sqrt(var))

    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"

    return var.new_var(
        dtype_to_floating(var.var_type),
        f"inversesqrt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )