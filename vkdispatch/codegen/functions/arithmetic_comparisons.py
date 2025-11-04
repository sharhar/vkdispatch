import vkdispatch.base.dtype as dtypes

from ..variables.base_variable import BaseVariable

from ..global_codegen_callbacks import new_var

from .arithmetic import is_number

from typing import Any

def less_than(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} < {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} < {other.resolve()}",
        parents=[var, other]
    )

def less_or_equal(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} <= {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} <= {other.resolve()}",
        parents=[var, other]
    )

def equal_to(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} == {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} == {other.resolve()}",
        parents=[var, other]
    )

def not_equal_to(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} != {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} != {other.resolve()}",
        parents=[var, other]
    )

def greater_than(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} > {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} > {other.resolve()}",
        parents=[var, other]
    )

def greater_or_equal(var: BaseVariable, other: Any) -> BaseVariable:
    assert isinstance(var, BaseVariable), "First argument must be a ShaderVariable"

    if is_number(other):
        return new_var(
            dtypes.int32,
            f"{var.resolve()} >= {other}",
            parents=[var]
        )

    assert isinstance(other, BaseVariable)

    return new_var(
        dtypes.int32,
        f"{var.resolve()} >= {other.resolve()}",
        parents=[var, other]
    )