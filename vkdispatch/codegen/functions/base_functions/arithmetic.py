import vkdispatch.base.dtype as dtypes
from  vkdispatch.codegen.variables.base_variable import BaseVariable
from typing import Any, Tuple

from .. import scalar_eval as se

def my_log2_int(x: int) -> int:
    return int(se.round(se.log2(x)))


from . import base_utils


def _mark_arith_unary(var: BaseVariable, op: str) -> None:
    base_utils.get_codegen_backend().mark_composite_unary_op(var.var_type, op)


def _mark_arith_binary(lhs_type: dtypes.dtype, rhs_type: dtypes.dtype, op: str, *, inplace: bool = False) -> None:
    base_utils.get_codegen_backend().mark_composite_binary_op(lhs_type, rhs_type, op, inplace=inplace)

def _resolve_arithmetic_binary_expr(
    op: str,
    lhs_type: dtypes.dtype,
    lhs_expr: str,
    rhs_type: dtypes.dtype,
    rhs_expr: str,
) -> Tuple[str, bool]:
    override_expr = base_utils.get_codegen_backend().arithmetic_binary_expr(
        op, lhs_type, lhs_expr, rhs_type, rhs_expr
    )
    if override_expr is not None:
        return override_expr, True
    return f"{lhs_expr} {op} {rhs_expr}", False

def _resolve_arithmetic_unary_expr(op: str, var_type: dtypes.dtype, var_expr: str) -> Tuple[str, bool]:
    override_expr = base_utils.get_codegen_backend().arithmetic_unary_expr(op, var_type, var_expr)
    if override_expr is not None:
        return override_expr, True
    return f"{op}{var_expr}", False

def arithmetic_op_common(var: BaseVariable,
                         other: Any,
                         reverse: bool = False,
                         inplace: bool = False) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    result_type = None

    if base_utils.is_scalar_number(other):
        result_type = dtypes.cross_type(var.var_type, base_utils.number_to_dtype(other))
    elif isinstance(other, BaseVariable):
        result_type = dtypes.cross_type(var.var_type, other.var_type)
    elif base_utils.is_complex_number(other):
        raise TypeError("Python built-in complex numbers are not supported in arithmetic operations yet!")
    else:
        raise TypeError(f"Unsupported type for arithmetic op: ShaderVariable and {type(other)}")

    if inplace:
        assert var.is_setable(), "Inplace arithmetic requires the variable to be settable."
        assert not reverse, "Inplace arithmetic does not support reverse operations."
        var.read_callback()
        var.write_callback()
        assert result_type == var.var_type, "Inplace arithmetic requires the result type to match the variable type."

    if base_utils.is_scalar_number(other):
        return result_type

    if inplace:
        other.read_callback()
    
    return dtypes.cross_type(var.var_type, other.var_type)

def add(var: BaseVariable, other: Any, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, inplace=inplace)

    if base_utils.is_scalar_number(other):
        scalar_type = base_utils.number_to_dtype(other)
        scalar_expr = base_utils.format_number_literal(other)
        _mark_arith_binary(var.var_type, scalar_type, "+", inplace=inplace)
        expr, use_assignment = _resolve_arithmetic_binary_expr(
            "+",
            var.var_type,
            var.resolve(),
            scalar_type,
            scalar_expr,
        )
        if not inplace:
            if use_assignment:
                return base_utils.new_base_var(
                    return_type,
                    expr,
                    parents=[var],
                )
            return base_utils.new_scaled_var(
                return_type,
                var.resolve(),
                offset=other,
                parents=[var])

        if use_assignment:
            base_utils.append_contents(f"{var.resolve()} = {expr};\n")
        else:
            base_utils.append_contents(f"{var.resolve()} += {scalar_expr};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_arith_binary(var.var_type, other.var_type, "+", inplace=inplace)
    expr, use_assignment = _resolve_arithmetic_binary_expr(
        "+",
        var.var_type,
        var.resolve(),
        other.var_type,
        other.resolve(),
    )

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            expr,
            parents=[var, other])
    
    if use_assignment:
        base_utils.append_contents(f"{var.resolve()} = {expr};\n")
    else:
        base_utils.append_contents(f"{var.resolve()} += {other.resolve()};\n")
    return var

def sub(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)

    if base_utils.is_scalar_number(other):
        scalar_type = base_utils.number_to_dtype(other)
        scalar_expr = base_utils.format_number_literal(other)
        if reverse and not inplace:
            _mark_arith_unary(var, "-")
            _mark_arith_binary(var.var_type, scalar_type, "+", inplace=False)
        else:
            # Non-reverse scalar subtraction is emitted as `+ (-scalar)` via scaled-var optimization.
            _mark_arith_binary(var.var_type, scalar_type, "+" if not inplace else "-", inplace=inplace)
        expr, use_assignment = _resolve_arithmetic_binary_expr(
            "-",
            scalar_type if reverse else var.var_type,
            scalar_expr if reverse else var.resolve(),
            var.var_type if reverse else scalar_type,
            var.resolve() if reverse else scalar_expr,
        )
        if not inplace:
            if use_assignment:
                return base_utils.new_base_var(
                    return_type,
                    expr,
                    parents=[var],
                )
            return base_utils.new_scaled_var(
                return_type,
                f"(-{var.resolve()})" if reverse else var.resolve(),
                offset=other,
                parents=[var])

        if use_assignment:
            base_utils.append_contents(f"{var.resolve()} = {expr};\n")
        else:
            base_utils.append_contents(f"{var.resolve()} -= {scalar_expr};\n")
        return var

    assert isinstance(other, BaseVariable)
    lhs_type = var.var_type if not reverse else other.var_type
    rhs_type = other.var_type if not reverse else var.var_type
    _mark_arith_binary(lhs_type, rhs_type, "-", inplace=inplace)
    expr, use_assignment = _resolve_arithmetic_binary_expr(
        "-",
        lhs_type,
        var.resolve() if not reverse else other.resolve(),
        rhs_type,
        other.resolve() if not reverse else var.resolve(),
    )

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            expr,
            parents=[var, other])
    
    if use_assignment:
        base_utils.append_contents(f"{var.resolve()} = {expr};\n")
    else:
        base_utils.append_contents(f"{var.resolve()} -= {other.resolve()};\n")
    return var

def mul(var: BaseVariable, other: Any, inplace: bool = False) -> BaseVariable:
    if base_utils.is_scalar_number(other):
        return_type = arithmetic_op_common(var, other, inplace=inplace)
        scalar_type = base_utils.number_to_dtype(other)
        scalar_expr = base_utils.format_number_literal(other)
        expr, use_assignment = _resolve_arithmetic_binary_expr(
            "*",
            var.var_type,
            var.resolve(),
            scalar_type,
            scalar_expr,
        )
        if not inplace:
            if other == 1:
                return var

            if (
                not use_assignment
                and dtypes.is_integer_dtype(var.var_type)
                and base_utils.is_int_number(other)
                and base_utils.is_int_power_of_2(other)
            ):
                power = my_log2_int(other)
                _mark_arith_binary(var.var_type, scalar_type, "<<", inplace=False)
                return base_utils.new_base_var(var.var_type, f"{var.resolve()} << {power}", [var])

            _mark_arith_binary(var.var_type, scalar_type, "*", inplace=False)
            if use_assignment:
                return base_utils.new_base_var(
                    return_type,
                    expr,
                    parents=[var],
                )
            return base_utils.new_scaled_var(return_type, var.resolve(), scale=other, parents=[var])

        _mark_arith_binary(var.var_type, scalar_type, "*", inplace=True)
        if use_assignment:
            base_utils.append_contents(f"{var.resolve()} = {expr};\n")
        else:
            base_utils.append_contents(f"{var.resolve()} *= {scalar_expr};\n")
        return var

    assert isinstance(other, BaseVariable)

    if dtypes.is_complex(var.var_type) and dtypes.is_complex(other.var_type):
        raise ValueError("Complex multiplication is not supported via the `*` operator.")

    if dtypes.is_matrix(var.var_type) and dtypes.is_matrix(other.var_type):
        raise ValueError("Matrix multiplication is not supported via the `*` operator. Use `@` operator instead.")

    return_type = dtypes.cross_multiply_type(var.var_type, other.var_type)
    if inplace:
        assert var.is_setable(), "Inplace arithmetic requires the variable to be settable."
        var.read_callback()
        var.write_callback()
        other.read_callback()
        assert return_type == var.var_type, "Inplace arithmetic requires the result type to match the variable type."

    _mark_arith_binary(var.var_type, other.var_type, "*", inplace=inplace)
    expr, use_assignment = _resolve_arithmetic_binary_expr(
        "*",
        var.var_type,
        var.resolve(),
        other.var_type,
        other.resolve(),
    )
    if not inplace:
        return base_utils.new_base_var(
            return_type,
            expr,
            parents=[var, other])
    
    if use_assignment:
        base_utils.append_contents(f"{var.resolve()} = {expr};\n")
    else:
        base_utils.append_contents(f"{var.resolve()} *= {other.resolve()};\n")
    return var

def truediv(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    if dtypes.is_integer_dtype(var.var_type) and inplace:
        raise ValueError("Inplace true division is not supported for integer types.")
    
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    return_type = dtypes.make_floating_dtype(return_type)

    if base_utils.is_scalar_number(other):
        scalar_f_type = dtypes.float32
        other_expr = base_utils.format_number_literal(other, force_float32=True)
        if not reverse:
            _mark_arith_binary(return_type, scalar_f_type, "/", inplace=inplace)
        else:
            _mark_arith_binary(scalar_f_type, return_type, "/", inplace=inplace)
        lhs_expr = base_utils.to_dtype_base(return_type, var).resolve() if not reverse else other_expr
        rhs_expr = other_expr if not reverse else base_utils.to_dtype_base(return_type, var).resolve()
        lhs_type = return_type if not reverse else scalar_f_type
        rhs_type = scalar_f_type if not reverse else return_type
        expr, use_assignment = _resolve_arithmetic_binary_expr(
            "/",
            lhs_type,
            lhs_expr,
            rhs_type,
            rhs_expr,
        )
        if not inplace:
            return base_utils.new_base_var(
                return_type,
                expr,
                parents=[var])

        if use_assignment:
            inplace_expr, _ = _resolve_arithmetic_binary_expr(
                "/",
                var.var_type,
                var.resolve(),
                scalar_f_type,
                other_expr,
            )
            base_utils.append_contents(f"{var.resolve()} = {inplace_expr};\n")
        else:
            base_utils.append_contents(f"{var.resolve()} /= {other_expr};\n")
        return var

    assert isinstance(other, BaseVariable)

    if dtypes.is_complex(var.var_type) and dtypes.is_complex(other.var_type):
        raise ValueError("Complex division is not supported.")

    if dtypes.is_matrix(var.var_type) and dtypes.is_matrix(other.var_type):
        raise ValueError("Matrix division is not supported.")

    lhs_mark_type = return_type if not reverse else dtypes.make_floating_dtype(other.var_type)
    rhs_mark_type = dtypes.make_floating_dtype(other.var_type) if not reverse else return_type
    _mark_arith_binary(lhs_mark_type, rhs_mark_type, "/", inplace=inplace)

    lhs_expr = (
        base_utils.to_dtype_base(lhs_mark_type, var).resolve()
        if not reverse else
        base_utils.to_dtype_base(lhs_mark_type, other).resolve()
    )
    rhs_expr = (
        base_utils.to_dtype_base(rhs_mark_type, other).resolve()
        if not reverse else
        base_utils.to_dtype_base(rhs_mark_type, var).resolve()
    )
    expr, use_assignment = _resolve_arithmetic_binary_expr(
        "/",
        lhs_mark_type,
        lhs_expr,
        rhs_mark_type,
        rhs_expr,
    )

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            expr,
            parents=[var, other])
    
    if use_assignment:
        inplace_expr, _ = _resolve_arithmetic_binary_expr(
            "/",
            var.var_type,
            var.resolve(),
            rhs_mark_type,
            rhs_expr,
        )
        base_utils.append_contents(f"{var.resolve()} = {inplace_expr};\n")
    else:
        base_utils.append_contents(f"{var.resolve()} /= {rhs_expr};\n")
    return var

def floordiv(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    assert dtypes.is_integer_dtype(var.var_type), "Floor division is only supported for integer types."
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    assert dtypes.is_integer_dtype(return_type), "Floor division is only supported for integer types."

    if base_utils.is_scalar_number(other):
        assert base_utils.is_int_number(other), "Floor division only supports integer scalar values."

        if not inplace:
            if other == 1:
                return var

            if base_utils.is_int_power_of_2(other):
                power = my_log2_int(other)
                _mark_arith_binary(var.var_type, base_utils.number_to_dtype(other), ">>", inplace=False)
                return base_utils.new_base_var(var.var_type, f"{var.resolve()} >> {power}", [var])

            scalar_type = base_utils.number_to_dtype(other)
            _mark_arith_binary(var.var_type if not reverse else scalar_type, scalar_type if not reverse else var.var_type, "/", inplace=False)
            return base_utils.new_base_var(
                return_type,
                (
                    f"{var.resolve()} / {other}"
                    if not reverse else
                    f"{other} / {var.resolve()}"
                ),
                parents=[var])

        _mark_arith_binary(var.var_type, base_utils.number_to_dtype(other), "/", inplace=True)
        base_utils.append_contents(f"{var.resolve()} /= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_arith_binary(var.var_type if not reverse else other.var_type, other.var_type if not reverse else var.var_type, "/", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            (
                f"{var.resolve()} / {other.resolve()}"
                if not reverse else
                f"{other.resolve()} / {var.resolve()}"
            ),
            parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} /= {other.resolve()};\n")
    return var

def mod(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    assert dtypes.is_integer_dtype(var.var_type), "Modulus is only supported for integer types."
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)
    assert dtypes.is_integer_dtype(return_type), "Modulus is only supported for integer types."

    if base_utils.is_scalar_number(other):
        scalar_type = base_utils.number_to_dtype(other)
        _mark_arith_binary(var.var_type if not reverse else scalar_type, scalar_type if not reverse else var.var_type, "%", inplace=inplace)
        if not inplace:
            return base_utils.new_base_var(
                return_type,
                (
                    f"{var.resolve()} % {other}"
                    if not reverse else
                    f"{other} % {var.resolve()}"
                ),
                parents=[var])

        base_utils.append_contents(f"{var.resolve()} %= {other};\n")
        return var

    assert isinstance(other, BaseVariable)
    _mark_arith_binary(var.var_type if not reverse else other.var_type, other.var_type if not reverse else var.var_type, "%", inplace=inplace)

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            (
                f"{var.resolve()} % {other.resolve()}"
                if not reverse else
                f"{other.resolve()} % {var.resolve()}"
            ),
            parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} %= {other.resolve()};\n")
    return var

def pow(var: BaseVariable, other: Any, reverse: bool = False, inplace: bool = False) -> BaseVariable:
    return_type = arithmetic_op_common(var, other, reverse=reverse, inplace=inplace)

    if base_utils.is_scalar_number(other):
        other_expr = base_utils.format_number_literal(other)
        if not inplace:
            return base_utils.new_base_var(
                return_type,
                (
                    f"pow({var.resolve()}, {other_expr})"
                    if not reverse else
                    f"pow({other_expr}, {var.resolve()})"
                ),
                parents=[var])

        base_utils.append_contents(f"{var.resolve()} = pow({var.resolve()}, {other_expr});\n")
        return var

    assert isinstance(other, BaseVariable)

    if not inplace:
        return base_utils.new_base_var(
            return_type,
            (
                f"pow({var.resolve()}, {other.resolve()})"
                if not reverse else
                f"pow({other.resolve()}, {var.resolve()})"
            ),
            parents=[var, other])
    
    base_utils.append_contents(f"{var.resolve()} = pow({var.resolve()}, {other.resolve()});\n")
    return var

def neg(var: BaseVariable) -> BaseVariable:
    _mark_arith_unary(var, "-")
    expr, _ = _resolve_arithmetic_unary_expr("-", var.var_type, var.resolve())
    return base_utils.new_base_var(
        var.var_type,
        expr,
        parents=[var])

def absolute(var: BaseVariable) -> BaseVariable:
    return base_utils.new_base_var(
        var.var_type,
        f"abs({var.resolve()})",
        parents=[var],
        lexical_unit=True)
