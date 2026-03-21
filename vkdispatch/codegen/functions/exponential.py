import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from . import utils
from . import scalar_eval as se

def _is_glsl_backend() -> bool:
    return utils.codegen_backend().name == "glsl"

def _is_float64_dtype(var_type: dtypes.dtype) -> bool:
    if dtypes.is_scalar(var_type):
        return var_type == dtypes.float64

    if dtypes.is_vector(var_type):
        return var_type.scalar == dtypes.float64

    return False

def _float64_to_float32_dtype(var_type: dtypes.dtype) -> dtypes.dtype:
    if var_type == dtypes.float64:
        return dtypes.float32

    if dtypes.is_vector(var_type) and var_type.scalar == dtypes.float64:
        return dtypes.to_vector(dtypes.float32, var_type.child_count)

    raise TypeError(f"Unsupported fp64 fallback dtype: {var_type}")

def _needs_glsl_float64_trig_fallback(var_type: dtypes.dtype) -> bool:
    return _is_glsl_backend() and _is_float64_dtype(var_type)

def process_float_var(var: ShaderVariable) -> bool:
    pass

def _unary_math_var(func_name: str, var: ShaderVariable) -> ShaderVariable:
    result_type = utils.dtype_to_floating(var.var_type)
    expr_arg_type = result_type
    expr_arg = var.resolve()
    expr_result_type = result_type

    if _needs_glsl_float64_trig_fallback(result_type) and func_name in {"exp", "exp2", "log", "log2"}:
        expr_arg_type = _float64_to_float32_dtype(result_type)
        expr_result_type = expr_arg_type
        expr_arg = utils.backend_constructor_from_resolved(expr_arg_type, [expr_arg])

    expr = utils.codegen_backend().unary_math_expr(func_name, expr_result_type, expr_arg)

    if expr_result_type != result_type:
        expr = utils.backend_constructor_from_resolved(result_type, [expr])

    return utils.new_var(
        result_type,
        expr,
        parents=[var],
        lexical_unit=True
    )

def pow(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.power(x, y)
    
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
        return se.exp(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("exp", var)

def exp2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.exp2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("exp2", var)

def log(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.log(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("log", var)

def log2(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.log2(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("log2", var)

# has double
def sqrt(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.sqrt(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    return _unary_math_var("sqrt", var)

# has double
def inversesqrt(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(1.0 / se.sqrt(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"
    utils.mark_backend_feature("inversesqrt")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"inversesqrt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )
