import vkdispatch.base.dtype as dtypes
from vkdispatch.codegen.variables.base_variable import BaseVariable
import numpy as np
from typing import Any, Optional

import numbers

from vkdispatch.codegen.shader_writer import new_scaled_var, append_contents, new_name

from vkdispatch.codegen.shader_writer import new_var as new_var_impl

def new_base_var(var_type: dtypes.dtype,
            var_name: Optional[str],
            parents: list,
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> BaseVariable:
    return new_var_impl(var_type, var_name, parents, lexical_unit, settable, register)

def is_number(x) -> bool:
    return isinstance(x, numbers.Number) and not isinstance(x, bool)

def is_int_number(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)

def is_float_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral) and not isinstance(x, bool) \
           and (isinstance(x, float) or isinstance(x, np.floating))

def is_complex_number(x) -> bool:
    return isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real)

def is_scalar_number(x) -> bool:
    return is_number(x) and (is_int_number(x) or is_float_number(x)) and not is_complex_number(x)

def is_int_power_of_2(n: int) -> bool:
    """Check if an integer is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def number_to_dtype(number: numbers.Number):
    if is_int_number(number):
        if number >= 0:
            return dtypes.uint32

        return dtypes.int32
    elif is_float_number(number):
        return dtypes.float32
    elif is_complex_number(number):
        return dtypes.complex64
    else:
        raise TypeError(f"Unsupported number type: {type(number)}")

def check_is_int(variable):
    return isinstance(variable, int) or np.issubdtype(type(variable), np.integer)

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

def resolve_input(var: Any) -> str:
    #print("Resolving input:", var)

    if is_number(var):
        return str(var)
    
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"
    return var.resolve()


def to_dtype_base(var_type: dtypes.dtype, *args):
    return new_base_var(
        var_type,
        f"{var_type.glsl_type}({', '.join([resolve_input(elem) for elem in args])})", 
        args,
        lexical_unit=True
    )
