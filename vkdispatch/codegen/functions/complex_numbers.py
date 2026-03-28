import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from .common_builtins import fma

from .type_casting import to_complex, to_dtype
from . import utils

from .trigonometry import cos, sin

def complex_from_euler_angle(angle: ShaderVariable):
    if not isinstance(angle, ShaderVariable):
        raise TypeError("complex_from_euler_angle expects a ShaderVariable angle")

    target_complex_type = dtypes.complex_from_float(dtypes.make_floating_dtype(angle.var_type))
    return to_dtype(target_complex_type, cos(angle), sin(angle))

def validate_complex_number(arg1: Any) -> Union[ShaderVariable, complex]:
    if isinstance(arg1, ShaderVariable):
        if not dtypes.is_complex(arg1.var_type):
            raise TypeError(f"Input variable '{arg1.resolve()}' of type '{arg1.var_type.name}' is not a complex type!")
        
        return arg1
    
    if not utils.is_number(arg1):
        raise TypeError(f"Argument must be a ShaderVariable or a number, got {type(arg1)}!")

    return complex(arg1)
    
def _new_big_complex(var_type: dtypes.dtype, arg1: Any, arg2: Any):
    var_str = utils.backend_constructor(var_type, arg1, arg2)

    return utils.new_var(
        var_type,
        var_str, 
        [utils.resolve_input(arg1), utils.resolve_input(arg2)],
        lexical_unit=True
    )

def mult_complex(arg1: ShaderVariable, arg2: ShaderVariable):
    a1 = validate_complex_number(arg1)
    a2 = validate_complex_number(arg2)

    fallback_type = dtypes.complex64
    for normalized_arg in (a1, a2):
        if isinstance(normalized_arg, ShaderVariable):
            fallback_type = normalized_arg.var_type
            break

    result_type = None
    for normalized_arg in (a1, a2):
        arg_type = normalized_arg.var_type if isinstance(normalized_arg, ShaderVariable) else fallback_type
        result_type = arg_type if result_type is None else dtypes.cross_type(result_type, arg_type)

    return _new_big_complex(
        result_type, # type: ignore[arg-type]
        fma(a1.real, a2.real, -a1.imag * a2.imag),
        fma(a1.real, a2.imag, a1.imag * a2.real),
    )
