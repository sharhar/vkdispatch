import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from typing import Any, Union
import numpy as np

from .common_builtins import fma

from .type_casting import to_complex
from . import utils

from .trigonometry import cos, sin

def complex_from_euler_angle(angle: BaseVariable):
    return to_complex(cos(angle), sin(angle))

def validate_complex_number(arg1: Any) -> Union[BaseVariable, complex]:
    if isinstance(arg1, BaseVariable):
        assert arg1.var_type == dtypes.complex64, "Input variables to complex multiplication must be complex"
        return arg1
    
    assert utils.is_number(arg1), "Argument must be BaseVariable or number"
    
    return complex(arg1)

def complex_conjugate(arg: BaseVariable):
    a = validate_complex_number(arg)
    return to_complex(a.real, -a.imag)

def mult_complex(arg1: BaseVariable, arg2: BaseVariable):
    a1 = validate_complex_number(arg1)
    a2 = validate_complex_number(arg2)

    return to_complex(a1.real * a2.real - a1.imag * a2.imag, a1.real * a2.imag + a1.imag * a2.real)

def mult_complex_conj(arg1: BaseVariable, arg2: BaseVariable):
    a1 = validate_complex_number(arg1)
    a2 = validate_complex_number(arg2)

    return to_complex(a1.real * a2.real + a1.imag * a2.imag, a1.real * a2.imag - a1.imag * a2.real)


def mult_complex_fma(register_out: BaseVariable, register_a: BaseVariable, register_b: complex):
    r_out = validate_complex_number(register_out)
    r_a = validate_complex_number(register_a)
    r_b = validate_complex_number(register_b)

    r_out.real = r_a.imag * -r_b.imag
    r_out.real = fma(r_a.real, r_b.real, r_out.real)

    r_out.imag = r_a.imag * r_b.real
    r_out.imag = fma(r_a.real, r_b.imag, r_out.imag)

def mult_complex_conj_fma(register_out: BaseVariable, register_a: BaseVariable, register_b: complex):
    r_out = validate_complex_number(register_out)
    r_a = validate_complex_number(register_a)
    r_b = validate_complex_number(register_b)

    assert isinstance(register_out, BaseVariable), "Out register must be a BaseVariable"
    assert register_out.is_register(), "Our register must be a register"

    r_out.real = r_a.imag * r_b.imag
    r_out.real = fma(r_a.real, r_b.real, r_out.real)

    r_out.imag = r_a.imag * -r_b.real
    r_out.imag = fma(r_a.real, r_b.imag, r_out.imag)