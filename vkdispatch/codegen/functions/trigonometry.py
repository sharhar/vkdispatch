import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, List, Union

from . import utils
from . import scalar_eval as se

def dtype_to_floating(var_type: dtypes.dtype) -> dtypes.dtype:
    return dtypes.make_floating_dtype(var_type)

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

def _cast_expr(var_type: dtypes.dtype, expr: str) -> str:
    return utils.backend_constructor_from_resolved(var_type, [expr])

def _unary_math_var(func_name: str, var: ShaderVariable) -> ShaderVariable:
    result_type = dtype_to_floating(var.var_type)
    expr_arg_type = result_type
    expr_arg = var.resolve()
    expr_result_type = result_type

    if _needs_glsl_float64_trig_fallback(result_type):
        expr_arg_type = _float64_to_float32_dtype(result_type)
        expr_result_type = expr_arg_type
        expr_arg = _cast_expr(expr_arg_type, expr_arg)

    expr = utils.codegen_backend().unary_math_expr(func_name, expr_result_type, expr_arg)

    if expr_result_type != result_type:
        expr = _cast_expr(result_type, expr)

    return utils.new_var(
        result_type,
        expr,
        parents=[var],
        lexical_unit=True
    )

def _binary_math_var(
    func_name: str,
    result_type: dtypes.dtype,
    lhs_type: dtypes.dtype,
    lhs_expr: str,
    rhs_type: dtypes.dtype,
    rhs_expr: str,
    parents: List[ShaderVariable],
    *,
    lexical_unit: bool = False,
) -> ShaderVariable:
    expr_result_type = result_type
    expr_lhs_type = lhs_type
    expr_rhs_type = rhs_type
    expr_lhs = lhs_expr
    expr_rhs = rhs_expr

    if _needs_glsl_float64_trig_fallback(result_type):
        expr_result_type = _float64_to_float32_dtype(result_type)

        if _is_float64_dtype(lhs_type):
            expr_lhs_type = _float64_to_float32_dtype(lhs_type)
            expr_lhs = _cast_expr(expr_lhs_type, lhs_expr)

        if _is_float64_dtype(rhs_type):
            expr_rhs_type = _float64_to_float32_dtype(rhs_type)
            expr_rhs = _cast_expr(expr_rhs_type, rhs_expr)

    expr = utils.codegen_backend().binary_math_expr(
        func_name,
        expr_lhs_type,
        expr_lhs,
        expr_rhs_type,
        expr_rhs,
    )

    if expr_result_type != result_type:
        expr = _cast_expr(result_type, expr)

    return utils.new_var(
        result_type,
        expr,
        parents=parents,
        lexical_unit=lexical_unit,
    )

def radians(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return var * (3.141592653589793 / 180.0)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    utils.mark_backend_feature("radians")
    return _unary_math_var("radians", var)

def degrees(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return var * (180.0 / 3.141592653589793)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    utils.mark_backend_feature("degrees")
    return _unary_math_var("degrees", var)

def sin(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.sin(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("sin", var)

def cos(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.cos(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")

    return _unary_math_var("cos", var)

def tan(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.tan(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("tan", var)

def asin(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arcsin(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("asin", var)

def acos(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arccos(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("acos", var)

def atan(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arctan(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("atan", var)

def atan2(y: Any, x: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.arctan2(y, x)
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        result_type = dtype_to_floating(y.var_type)
        scalar_result_type = result_type.scalar if dtypes.is_vector(result_type) else result_type
        return _binary_math_var(
            "atan2",
            result_type,
            result_type,
            y.resolve(),
            scalar_result_type,
            utils.resolve_input(x),
            [y],
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        result_type = dtype_to_floating(x.var_type)
        scalar_result_type = result_type.scalar if dtypes.is_vector(result_type) else result_type
        return _binary_math_var(
            "atan2",
            result_type,
            scalar_result_type,
            utils.resolve_input(y),
            result_type,
            x.resolve(),
            [x],
        )

    if not isinstance(y, ShaderVariable) or not isinstance(x, ShaderVariable):
        raise ValueError("Both arguments must be ShaderVariables or numbers")

    result_type = dtype_to_floating(dtypes.cross_type(y.var_type, x.var_type))
    return _binary_math_var(
        "atan2",
        result_type,
        result_type,
        y.resolve(),
        dtype_to_floating(x.var_type),
        x.resolve(),
        [y, x],
        lexical_unit=True,
    )

def sinh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.sinh(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("sinh", var)

def cosh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.cosh(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")

    return _unary_math_var("cosh", var)

def tanh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.tanh(var)
    
    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("tanh", var)

def asinh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arcsinh(var)
    
    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("asinh", var)

def acosh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arccosh(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("acosh", var)

def atanh(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.arctanh(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError("Argument must be a ShaderVariable or number")
    
    return _unary_math_var("atanh", var)
