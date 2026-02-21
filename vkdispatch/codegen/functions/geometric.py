import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from . import utils
from ..._compat import numpy_compat as npc

def length(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.abs_value(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"length({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def distance(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return npc.abs_value(y - x)
    
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
        return npc.dot(x, y)
    
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
    assert isinstance(x, ShaderVariable), "Argument x must be a ShaderVariable"
    assert isinstance(y, ShaderVariable), "Argument y must be a ShaderVariable"

    assert x.var_type == dtypes.vec3, "Argument x must be of type vec3 or dvec3"
    assert y.var_type == dtypes.vec3, "Argument y must be of type vec3 or dvec3"

    return utils.new_var(
        dtypes.vec3,
        f"cross({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def normalize(var: ShaderVariable) -> ShaderVariable:
    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable"

    return utils.new_var(
        var.var_type,
        f"normalize({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )
