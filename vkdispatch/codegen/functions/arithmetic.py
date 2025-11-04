import vkdispatch.base.dtype as dtypes

from ..global_codegen_callbacks import append_contents
from ..variables.base_variable import BaseVariable

from typing import Any

import numpy as np
import numbers

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
    return is_number() and (is_int_number(x) or is_float_number(x)) and not is_complex_number(x)

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
    # elif is_complex_number(number):
    #     return dtypes.complex64
    else:
        raise TypeError(f"Unsupported number type: {type(number)}")

def arithmetic_op_common(var: BaseVariable,
                         other: Any,
                         reverse: bool = False,
                         inplace: bool = False) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    result_type = None

    if is_scalar_number(other):
        result_type = dtypes.cross_type(var.var_type, number_to_dtype(other))
    elif isinstance(other, BaseVariable):
        result_type = dtypes.cross_type(var.var_type, other.var_type)
    elif is_complex_number(other):
        raise TypeError("Python built-in complex numbers are not supported in arithmetic operations yet!")
    else:
        raise TypeError(f"Unsupported type for arithmetic op: ShaderVariable and {type(other)}")

    if inplace:
        assert var.is_setable(), "Inplace arithmetic requires the variable to be settable."
        assert not reverse, "Inplace arithmetic does not support reverse operations."
        var.read_callback()
        var.write_callback()
        assert result_type == var.var_type, "Inplace arithmetic requires the result type to match the variable type."

    if is_scalar_number(other):
        return result_type

    if inplace:
        other.read_callback()
    
    return dtypes.cross_type(var.var_type, other.var_type)

def add(var: BaseVariable, other: Any, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, inplace=inplace)

    if is_scalar_number(other):
        if not inplace:
            return var.new_scaled_var(
                return_type,
                var.resolve(),
                offset=other,
                parents=[var])

        append_contents(f"{var.resolve()} += {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return var.new_var(
            return_type,
            f"{var.resolve()} + {other.resolve()}",
            parents=[var, other])
    
    append_contents(f"{var.resolve()} += {other.resolve()};\n")
    return var

def sub(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)

    if is_scalar_number(other):
        if not inplace:
            return var.new_scaled_var(
                return_type,
                f"(-{var.resolve()})" if reverse else var.resolve(),
                offset=other,
                parents=[var])

        append_contents(f"{var.resolve()} -= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return var.new_var(
            return_type,
            (
                f"{var.resolve()} - {other.resolve()}"
                if not reverse else
                f"{other.resolve()} - {var.resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} -= {other.resolve()};\n")
    return var

def mul(var: BaseVariable, other: Any, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, inplace=inplace)

    if is_scalar_number(other):
        if not inplace:
            if other == 1:
                return var

            if dtypes.is_integer_dtype(var.var_type) and is_int_number(other) and is_int_power_of_2(other):
                power = int(np.round(np.log2(other)))
                return var.new_var(var.var_type, f"{var.resolve()} << {power}", [var])

            return var.new_scaled_var(
                return_type,
                var.resolve(),
                scale=other,
                parents=[var])

        append_contents(f"{var.resolve()} *= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if dtypes.is_complex(var.var_type) and dtypes.is_complex(other.var_type):
        raise ValueError("Complex multiplication is not supported via the `*` operator.")

    if dtypes.is_matrix(var.var_type) and dtypes.is_matrix(other.var_type):
        raise ValueError("Matrix multiplication is not supported via the `*` operator. Use `@` operator instead.")

    if not inplace:
        return var.new_var(
            var.var_type,
            f"{var.resolve()} * {other.resolve()}",
            parents=[var, other])
    
    append_contents(f"{var.resolve()} *= {other.resolve()};\n")
    return var

def truediv(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    if dtypes.is_integer_dtype(var.var_type) and inplace:
        raise ValueError("Inplace true division is not supported for integer types.")
    
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    return_type = dtypes.make_floating_dtype(return_type)

    if is_scalar_number(other):
        if not inplace:
            return var.new_var(
                return_type,
                (
                    f"{var.cast_to(return_type).resolve()} / {float(other)}"
                    if not reverse else
                    f"{float(other)} / {var.cast_to(return_type).resolve()}"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} /= {float(other)};\n")
        return var

    assert isinstance(other, BaseVariable)

    if dtypes.is_complex(var.var_type) and dtypes.is_complex(other.var_type):
        raise ValueError("Complex division is not supported.")

    if dtypes.is_matrix(var.var_type) and dtypes.is_matrix(other.var_type):
        raise ValueError("Matrix division is not supported.")

    if not inplace:
        return var.new_var(
            return_type,
            (
                f"{var.cast_to(return_type).resolve()} / {other.cast_to(return_type).resolve()}"
                if not reverse else
                f"{other.cast_to(return_type).resolve()} / {var.cast_to(return_type).resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} /= {other.cast_to(return_type).resolve()};\n")
    return var

def floordiv(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    assert dtypes.is_integer_dtype(var.var_type), "Floor division is only supported for integer types."
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    assert dtypes.is_integer_dtype(return_type), "Floor division is only supported for integer types."

    if is_scalar_number(other):
        assert is_int_number(other), "Floor division only supports integer scalar values."

        if not inplace:
            if other == 1:
                return var

            if is_int_power_of_2(other):
                power = int(np.round(np.log2(other)))
                return var.new_var(var.var_type, f"{var.resolve()} >> {power}", [var])

            return var.new_var(
                return_type,
                (
                    f"{var.resolve()} / {other}"
                    if not reverse else
                    f"{other} / {var.resolve()}"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} /= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return var.new_var(
            return_type,
            (
                f"{var.resolve()} / {other.resolve()}"
                if not reverse else
                f"{other.resolve()} / {var.resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} /= {other.resolve()};\n")
    return var

def mod(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    assert dtypes.is_integer_dtype(var.var_type), "Modulus is only supported for integer types."
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    assert dtypes.is_integer_dtype(return_type), "Modulus is only supported for integer types."

    if is_scalar_number(other):
        if not inplace:
            return var.new_var(
                return_type,
                (
                    f"{var.resolve()} % {other}"
                    if not reverse else
                    f"{other} % {var.resolve()}"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} %= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return var.new_var(
            return_type,
            (
                f"{var.resolve()} % {other.resolve()}"
                if not reverse else
                f"{other.resolve()} % {var.resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} %= {other.resolve()};\n")
    return var

def pow(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)

    if is_scalar_number(other):
        if not inplace:
            return var.new_var(
                return_type,
                (
                    f"pow({var.resolve()}, {other})"
                    if not reverse else
                    f"pow({other}, {var.resolve()})"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} = pow({var.resolve()}, {other});\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return var.new_var(
            return_type,
            (
                f"pow({var.resolve()}, {other.resolve()})"
                if not reverse else
                f"pow({other.resolve()}, {var.resolve()})"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} = pow({var.resolve()}, {other.resolve()});\n")
    return var

def neg(var: BaseVariable) -> BaseVariable:
    return var.new_var(
        var.var_type,
        f"-{var.resolve()}",
        parents=[var])

def absolute(var: BaseVariable) -> BaseVariable:
    return var.new_var(
        var.var_type,
        f"abs({var.resolve()})",
        parents=[var],
        lexical_unit=True)