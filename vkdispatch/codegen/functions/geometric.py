import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from . import utils
from . import scalar_eval as se

def length(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.abs_value(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"length({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def distance(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.abs_value(y - x)
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"distance({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def dot(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.dot(x, y)
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"dot({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def cross(x: ShaderVariable, y: ShaderVariable) -> ShaderVariable:
    if not isinstance(x, ShaderVariable) or not isinstance(y, ShaderVariable):
        raise ValueError("Both arguments must be ShaderVariables")
    
    if x.var_type != dtypes.vec3 or y.var_type != dtypes.vec3:
        raise ValueError("Both arguments must be of type vec3 or dvec3")

    return utils.new_var(
        dtypes.vec3,
        f"cross({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def normalize(var: ShaderVariable) -> ShaderVariable:
    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable")

    return utils.new_var(
        var.var_type,
        f"normalize({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )
