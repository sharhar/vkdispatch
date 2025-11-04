import vkdispatch.base.dtype as dtypes

from ..global_codegen_callbacks import append_contents
from ..variables.base_variable import BaseVariable

from .arithmetic import number_to_dtype, is_int_number

from ..global_codegen_callbacks import new_var

from typing import Any

def bitwise_op_common(var: BaseVariable,
                         other: Any,
                         reverse: bool = False,
                         inplace: bool = False) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"
    assert dtypes.is_integer_dtype(var.var_type), "Bitwise operations only supported on integer types."

    result_type = None

    if is_int_number(other):
        result_type = dtypes.cross_type(var.var_type, number_to_dtype(other))
    elif isinstance(other, BaseVariable):
        result_type = dtypes.cross_type(var.var_type, other.var_type)
    else:
        raise TypeError(f"Unsupported type for bitwise op: ShaderVariable and {type(other)}")

    if inplace:
        assert var.is_setable(), "Inplace bitwise requires the variable to be settable."
        assert not reverse, "Inplace bitwise does not support reverse operations."
        var.read_callback()
        var.write_callback()
        assert result_type == var.var_type, "Inplace bitwise requires the result type to match the variable type."

    if is_int_number(other):
        return result_type

    assert dtypes.is_integer_dtype(other.var_type), "Bitwise operations only supported on integer types."

    if inplace:
        other.read_callback()
    
    return dtypes.cross_type(var.var_type, other.var_type)

def lshift(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False):
    return_type = bitwise_op_common(var, other, reverse=reverse, inplace=inplace)

    if is_int_number(other):
        if not inplace:
            return new_var(
                return_type,
                (
                    f"{var.resolve()} << {other}"
                    if not reverse else
                    f"{other} << {var.resolve()}"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} <<= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return new_var(
            return_type,
            (
                f"{var.resolve()} << {other.resolve()}"
                if not reverse else
                f"{other.resolve()} << {var.resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} <<= {other.resolve()};\n")
    return var

def rshift(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False):
    return_type = bitwise_op_common(var, other, reverse=reverse, inplace=inplace)

    if is_int_number(other):
        if not inplace:
            return new_var(
                return_type,
                (
                    f"{var.resolve()} >> {other}"
                    if not reverse else
                    f"{other} >> {var.resolve()}"
                ),
                parents=[var])

        append_contents(f"{var.resolve()} >>= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return new_var(
            return_type,
            (
                f"{var.resolve()} >> {other.resolve()}"
                if not reverse else
                f"{other.resolve()} >> {var.resolve()}"
            ),
            parents=[var, other])
    
    append_contents(f"{var.resolve()} >>= {other.resolve()};\n")
    return var

def and_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if is_int_number(other):
        if not inplace:
            return new_var(return_type, f"{var.resolve()} & {other}",parents=[var])

        append_contents(f"{var.resolve()} &= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return new_var(return_type, f"{var.resolve()} & {other.resolve()}",parents=[var, other])
    
    append_contents(f"{var.resolve()} &= {other.resolve()};\n")
    return var

def xor_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if is_int_number(other):
        if not inplace:
            return new_var(return_type, f"{var.resolve()} ^ {other}",parents=[var])

        append_contents(f"{var.resolve()} ^= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return new_var(return_type, f"{var.resolve()} ^ {other.resolve()}",parents=[var, other])
    
    append_contents(f"{var.resolve()} ^= {other.resolve()};\n")
    return var

def or_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if is_int_number(other):
        if not inplace:
            return new_var(return_type, f"{var.resolve()} | {other}",parents=[var])

        append_contents(f"{var.resolve()} |= {other};\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return new_var(return_type, f"{var.resolve()} | {other.resolve()}",parents=[var, other])
    
    append_contents(f"{var.resolve()} |= {other.resolve()};\n")
    return var

def invert(var: BaseVariable):
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"
    assert dtypes.is_integer_dtype(var.var_type), "Bitwise operations only supported on integer types."

    return new_var(
        var.var_type,
        f"~{var.resolve()}",
        parents=[var]
    )