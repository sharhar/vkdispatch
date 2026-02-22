import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from . import utils
from ..._compat import numpy_compat as npc

def _unary_math_var(func_name: str, var: ShaderVariable) -> ShaderVariable:
    result_type = utils.dtype_to_floating(var.var_type)
    return utils.new_var(
        result_type,
        utils.codegen_backend().unary_math_expr(func_name, result_type, var.resolve()),
        parents=[var],
        lexical_unit=True
    )

def pow(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return npc.power(x, y)
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        result_type = utils.dtype_to_floating(y.var_type)
        return utils.new_var(
            result_type,
            utils.codegen_backend().binary_math_expr(
                "pow",
                dtypes.float32,
                utils.resolve_input(x),
                result_type,
                y.resolve(),
            ),
            parents=[y]
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        result_type = utils.dtype_to_floating(x.var_type)
        return utils.new_var(
            result_type,
            utils.codegen_backend().binary_math_expr(
                "pow",
                result_type,
                x.resolve(),
                dtypes.float32,
                utils.resolve_input(y),
            ),
            parents=[x]
        )

    assert isinstance(y, ShaderVariable), "First argument must be a ShaderVariable or number"
    assert isinstance(x, ShaderVariable), "Second argument must be a ShaderVariable or number"

    result_type = utils.dtype_to_floating(dtypes.cross_type(x.var_type, y.var_type))
    return utils.new_var(
        result_type,
        utils.codegen_backend().binary_math_expr(
            "pow",
            utils.dtype_to_floating(x.var_type),
            x.resolve(),
            utils.dtype_to_floating(y.var_type),
            y.resolve(),
        ),
        parents=[y, x],
        lexical_unit=True
    )

def exp(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.exp(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("exp", var)

def exp2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.exp2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("exp2", var)

def log(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.log(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("log", var)

def log2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.log2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("log2", var)

def sqrt(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return npc.sqrt(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("sqrt", var)

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
