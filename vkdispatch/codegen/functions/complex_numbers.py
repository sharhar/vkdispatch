import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union
import numpy as np

from .common_builtins import fma

from .type_casting import to_complex
from . import utils

from .trigonometry import cos, sin

def complex_from_euler_angle(angle: ShaderVariable):
    return to_complex(cos(angle), sin(angle))

def validate_complex_number(arg1: Any) -> Union[ShaderVariable, complex]:
    if isinstance(arg1, ShaderVariable):
        assert arg1.var_type == dtypes.complex64, "Input variables to complex multiplication must be complex"
        return arg1
    
    assert utils.is_number(arg1), "Argument must be ShaderVariable or number"
    
    return complex(arg1)

def mult_complex(arg1: ShaderVariable, arg2: ShaderVariable):
    a1 = validate_complex_number(arg1)
    a2 = validate_complex_number(arg2)

    return to_complex(fma(a1.real, a2.real, -a1.imag * a2.imag), fma(a1.real, a2.imag, a1.imag * a2.real))