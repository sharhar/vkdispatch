import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union

from .common_builtins import fma

from .type_casting import to_complex
from . import utils

from .trigonometry import cos, sin

from ..shader_writer import scope_indentation

def complex_from_euler_angle(angle: ShaderVariable):
    return to_complex(cos(angle), sin(angle))

def validate_complex_number(arg1: Any) -> Union[ShaderVariable, complex]:
    if isinstance(arg1, ShaderVariable):
        assert arg1.var_type == dtypes.complex64, "Input variables to complex multiplication must be complex"
        return arg1
    
    assert utils.is_number(arg1), "Argument must be ShaderVariable or number"
    
    return complex(arg1)
    
def _new_big_complex(arg1: Any, arg2: Any):
    var_str = f"""{dtypes.complex64.glsl_type}(
{scope_indentation()}    {utils.resolve_input(arg1)},
{scope_indentation()}    {utils.resolve_input(arg2)})"""

    return utils.new_var(
        dtypes.complex64,
        var_str, 
        [utils.resolve_input(arg1), utils.resolve_input(arg2)],
        lexical_unit=True
    )

def mult_complex(arg1: ShaderVariable, arg2: ShaderVariable):
    a1 = validate_complex_number(arg1)
    a2 = validate_complex_number(arg2)

    return _new_big_complex(fma(a1.real, a2.real, -a1.imag * a2.imag), fma(a1.real, a2.imag, a1.imag * a2.real))
