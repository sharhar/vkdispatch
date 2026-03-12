import vkdispatch.base.dtype as dtypes
from  vkdispatch.codegen.variables.base_variable import BaseVariable
from typing import Any

from . import base_utils


def _mark_bit_unary(var: BaseVariable, op: str) -> None:
    base_utils.get_codegen_backend().mark_composite_unary_op(var.var_type, op)


def _mark_bit_binary(lhs_type: dtypes.dtype, rhs_type: dtypes.dtype, op: str, *, inplace: bool = False) -> None:
    base_utils.get_codegen_backend().mark_composite_binary_op(lhs_type, rhs_type, op, inplace=inplace)

def bitwise_op_common(var: BaseVariable,
                         other: Any,
                         reverse: bool = False,
                         inplace: bool = False) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"
    assert dtypes.is_integer_dtype(var.var_type), "Bitwise operations only supported on integer types."

    result_type = None

    if base_utils.is_int_number(other):
        result_type = dtypes.cross_type(var.var_type, base_utils.number_to_dtype(other))
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

    if base_utils.is_int_number(other):
        return result_type

    assert dtypes.is_integer_dtype(other.var_type), "Bitwise operations only supported on integer types."

    if inplace:
        other.read_callback()
    
    return dtypes.cross_type(var.var_type, other.var_type)

def lshift(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False):
    return_type = bitwise_op_common(var, other, reverse=reverse, inplace=inplace)

    if base_utils.is_int_number(other):
        _mark_bit_binary(var.var_type if not reverse else base_utils.number_to_dtype(other), base_utils.number_to_dtype(other) if not reverse else var.var_type, "<<", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(
                return_type,
                (
                    f"{var.resolve()} << {other}"
                    if not reverse else
                    f"{other} << {var.resolve()}"
                ),
                parents=[var])

        base_utils.append_contents(f"{var.resolve()} <<= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_bit_binary(var.var_type if not reverse else other.var_type, other.var_type if not reverse else var.var_type, "<<", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            (
                f"{var.resolve()} << {other.resolve()}"
                if not reverse else
                f"{other.resolve()} << {var.resolve()}"
            ),
            parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} <<= {other.resolve()};\n")
    return var

def rshift(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False):
    return_type = bitwise_op_common(var, other, reverse=reverse, inplace=inplace)

    if base_utils.is_int_number(other):
        _mark_bit_binary(var.var_type if not reverse else base_utils.number_to_dtype(other), base_utils.number_to_dtype(other) if not reverse else var.var_type, ">>", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(
                return_type,
                (
                    f"{var.resolve()} >> {other}"
                    if not reverse else
                    f"{other} >> {var.resolve()}"
                ),
                parents=[var])

        base_utils.append_contents(f"{var.resolve()} >>= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_bit_binary(var.var_type if not reverse else other.var_type, other.var_type if not reverse else var.var_type, ">>", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            (
                f"{var.resolve()} >> {other.resolve()}"
                if not reverse else
                f"{other.resolve()} >> {var.resolve()}"
            ),
            parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} >>= {other.resolve()};\n")
    return var

def and_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if base_utils.is_int_number(other):
        _mark_bit_binary(var.var_type, base_utils.number_to_dtype(other), "&", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(return_type, f"{var.resolve()} & {other}",parents=[var])

        base_utils.append_contents(f"{var.resolve()} &= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_bit_binary(var.var_type, other.var_type, "&", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(return_type, f"{var.resolve()} & {other.resolve()}",parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} &= {other.resolve()};\n")
    return var

def xor_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if base_utils.is_int_number(other):
        _mark_bit_binary(var.var_type, base_utils.number_to_dtype(other), "^", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(return_type, f"{var.resolve()} ^ {other}",parents=[var])

        base_utils.append_contents(f"{var.resolve()} ^= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_bit_binary(var.var_type, other.var_type, "^", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(return_type, f"{var.resolve()} ^ {other.resolve()}",parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} ^= {other.resolve()};\n")
    return var

def or_bits(var: BaseVariable, other: Any, inplace: bool = False):
    return_type = bitwise_op_common(var, other, inplace=inplace)

    if base_utils.is_int_number(other):
        _mark_bit_binary(var.var_type, base_utils.number_to_dtype(other), "|", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(return_type, f"{var.resolve()} | {other}",parents=[var])

        base_utils.append_contents(f"{var.resolve()} |= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_bit_binary(var.var_type, other.var_type, "|", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(return_type, f"{var.resolve()} | {other.resolve()}",parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} |= {other.resolve()};\n")
    return var

def invert(var: BaseVariable):
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"
    assert dtypes.is_integer_dtype(var.var_type), "Bitwise operations only supported on integer types."
    _mark_bit_unary(var, "~")

    return base_utils.new_base_var(
        var.var_type,
        f"~{var.resolve()}",
        parents=[var]
    )
