from ..variables.variables import ShaderVariable
from typing import Any, Union

from . import utils
from ..._compat import numpy_compat as npc

def pow(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return npc.power(x, y)
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        return utils.new_var(
            utils.dtype_to_floating(y.var_type),
            f"pow({x}, {y.resolve()})",
            parents=[y]
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        return utils.new_var(
            utils.dtype_to_floating(x.var_type),
            f"pow({x.resolve()}, {y})",
            parents=[x]
        )

    assert isinstance(y, ShaderVariable), "First argument must be a ShaderVariable or number"
    assert isinstance(x, ShaderVariable), "Second argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(y.var_type),
        f"pow({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def exp(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.exp(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"exp({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def exp2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.exp2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"exp2({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def log(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.log(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"log({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def log2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.log2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"log2({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def sqrt(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.sqrt(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"sqrt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def inversesqrt(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(1.0 / npc.sqrt(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    utils.mark_backend_feature("inversesqrt")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"inversesqrt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )
