import vkdispatch.base.dtype as dtypes
from vkdispatch.codegen.variables.base_variable import BaseVariable

from typing import Any, Optional

import numbers
import math

from ...._compat import numpy_compat as npc
from vkdispatch.codegen.shader_writer import new_scaled_var, append_contents, new_name
from vkdispatch.codegen.global_builder import get_codegen_backend

from vkdispatch.codegen.shader_writer import new_var as new_var_impl

_I32_MIN = -(2 ** 31)
_I32_MAX = 2 ** 31 - 1
_U32_MAX = 2 ** 32 - 1

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

def _is_numpy_float(x) -> bool:
    return npc.is_numpy_floating_instance(x)

def is_float_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral) and not isinstance(x, bool) \
           and (isinstance(x, float) or _is_numpy_float(x))

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
            if number <= _U32_MAX:
                return dtypes.uint32
            return dtypes.uint64

        if number >= _I32_MIN and number <= _I32_MAX:
            return dtypes.int32
        return dtypes.int64
    elif is_float_number(number):
        return dtypes.float32
    elif is_complex_number(number):
        return dtypes.complex64
    else:
        raise TypeError(f"Unsupported number type: {type(number)}")

def _check_is_int_numpy(x) -> bool:
    return npc.is_numpy_integer_scalar(x)

def check_is_int(variable):
    return npc.is_integer_scalar(variable)

def dtype_to_floating(var_type: dtypes.dtype) -> dtypes.dtype:
    return dtypes.make_floating_dtype(var_type)

def format_number_literal(var: numbers.Number, *, force_float32: bool = False) -> str:
    if is_complex_number(var):
        return str(var)

    if is_float_number(var) or (force_float32 and is_int_number(var)):
        value = float(var)

        if math.isinf(value):
            if value > 0:
                return get_codegen_backend().inf_f32_expr()
            return get_codegen_backend().ninf_f32_expr()

        if math.isnan(value):
            return "(0.0f / 0.0f)"

        literal = repr(value)
        if "e" not in literal and "E" not in literal and "." not in literal:
            literal += ".0"
        return literal + "f"

    return str(var)

def resolve_input(var: Any) -> str:
    #print("Resolving input:", var)

    if is_number(var):
        return format_number_literal(var)
    
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable or number"
    return var.resolve()

def backend_constructor(var_type: dtypes.dtype, *args) -> str:
    return get_codegen_backend().constructor(
        var_type,
        [resolve_input(elem) for elem in args]
    )

def to_dtype_base(var_type: dtypes.dtype, *args):
    return new_base_var(
        var_type,
        backend_constructor(var_type, *args),
        args,
        lexical_unit=True
    )
