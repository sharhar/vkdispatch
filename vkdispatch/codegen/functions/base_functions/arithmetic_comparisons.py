import vkdispatch.base.dtype as dtypes
from  vkdispatch.codegen.variables.base_variable import BaseVariable
from typing import Any

from . import base_utils

def less_than(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} < {base_utils.resolve_input(other)}",
        parents=[var, other]
    )

def less_or_equal(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} <= {base_utils.resolve_input(other)}",
        parents=[var, other]
    )

def equal_to(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} == {base_utils.resolve_input(other)}",
        parents=[var, other]
    )

def not_equal_to(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} != {base_utils.resolve_input(other)}",
        parents=[var, other]
    )

def greater_than(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} > {base_utils.resolve_input(other)}",
        parents=[var, other]
    )

def greater_or_equal(var: BaseVariable, other: Any) -> BaseVariable:
    return base_utils.new_base_var(
        dtypes.int32,
        f"{base_utils.resolve_input(var)} >= {base_utils.resolve_input(other)}",
        parents=[var, other]
    )
