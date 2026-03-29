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
    if not isinstance(var, BaseVariable):
        raise TypeError(f"First argument must be a ShaderVariable, but got {type(var)}")
    
    if not dtypes.is_integer_dtype(var.var_type):
        raise TypeError(f"Bitwise operations only supported on integer types, but got '{var.var_type.name}'")

    result_type = None

    if base_utils.is_int_number(other):
        result_type = dtypes.cross_type(var.var_type, base_utils.number_to_dtype(other))
    elif isinstance(other, BaseVariable):
        result_type = dtypes.cross_type(var.var_type, other.var_type)
    else:
        raise TypeError(f"Unsupported type for bitwise op: ShaderVariable and {type(other)}")

    if inplace:
        if not var.is_setable():
            raise ValueError("Inplace bitwise requires the variable to be settable.")
        
        if reverse:
            raise ValueError("Inplace bitwise does not support reverse operations.")
        
        var.read_callback()
        var.write_callback()

        if result_type != var.var_type:
            raise TypeError(f"Inplace bitwise requires the result type to match the variable type, but got '{result_type.name}' and '{var.var_type.name}' respectively.")

    if base_utils.is_int_number(other):
        return result_type

    if not dtypes.is_integer_dtype(other.var_type):
        raise TypeError(f"Bitwise operations only supported on integer types, but got '{other.var_type.name}'")

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

    if not isinstance(other, BaseVariable):
        raise TypeError(f"Unsupported type for left shift: ShaderVariable and {type(other)}")
    
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

    if not isinstance(other, BaseVariable):
        raise TypeError(f"Unsupported type for right shift: ShaderVariable and {type(other)}")
    
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
    
    if not isinstance(other, BaseVariable):
        raise TypeError(f"Unsupported type for bitwise AND: ShaderVariable and {type(other)}")
    
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
    
    if not isinstance(other, BaseVariable):
        raise TypeError(f"Unsupported type for bitwise XOR: ShaderVariable and {type(other)}")
    
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

    if not isinstance(other, BaseVariable):
        raise TypeError(f"Unsupported type for bitwise OR: ShaderVariable and {type(other)}")
    
    _mark_bit_binary(var.var_type, other.var_type, "|", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(return_type, f"{var.resolve()} | {other.resolve()}",parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} |= {other.resolve()};\n")
    return var

def invert(var: BaseVariable):
    if not isinstance(var, BaseVariable):
        raise TypeError(f"Argument must be a ShaderVariable, but got {type(var)}")
    
    if not dtypes.is_integer_dtype(var.var_type):
        raise TypeError(f"Bitwise operations only supported on integer types, but got '{var.var_type.name}'")
    
    _mark_bit_unary(var, "~")

    return base_utils.new_base_var(
        var.var_type,
        f"~{var.resolve()}",
        parents=[var]
    )
