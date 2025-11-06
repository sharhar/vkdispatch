import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union
import numpy as np

from . import utils

def dtype_to_floating(var_type: dtypes.dtype) -> dtypes.dtype:
    if var_type == dtypes.int32 or var_type == dtypes.uint32:
        return dtypes.float32

    if var_type == dtypes.ivec2 or var_type == dtypes.uvec2:
        return dtypes.vec2

    if var_type == dtypes.ivec3 or var_type == dtypes.uvec3:
        return dtypes.vec3
    
    if var_type == dtypes.ivec4 or var_type == dtypes.uvec4:
        return dtypes.vec4
    
    return var_type

def radians(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return var * (3.141592653589793 / 180.0)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"radians({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def degrees(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return var * (180.0 / 3.141592653589793)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"degrees({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def sin(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.sin(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"sin({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def cos(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.cos(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"cos({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def tan(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.tan(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"tan({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def asin(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arcsin(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"asin({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def acos(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arccos(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"acos({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def atan(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arctan(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"atan({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def atan2(y: Any, x: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return float(np.arctan2(y, x))
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        return utils.new_var(
            dtype_to_floating(y.var_type),
            f"atan({y.resolve()}, {x})",
            parents=[y]
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        return utils.new_var(
            dtype_to_floating(x.var_type),
            f"atan({y}, {x.resolve()})",
            parents=[x]
        )

    assert isinstance(y, ShaderVariable), "First argument must be a ShaderVariable or number"
    assert isinstance(x, ShaderVariable), "Second argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(y.var_type),
        f"atan({y.resolve()}, {x.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def sinh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.sinh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"sinh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def cosh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.cosh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"cosh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def tanh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.tanh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"tanh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def asinh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arcsinh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"asinh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def acosh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arccosh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"acosh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def atanh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.arctanh(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtype_to_floating(var.var_type),
        f"atanh({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )