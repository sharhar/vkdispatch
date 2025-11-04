import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from .arithmetic import is_number
from typing import Any, Union, Tuple

import numpy as np

from .common_builtins import dtype_to_floating, resolve_input

def matrix_comp_mult(x: BaseVariable, y: BaseVariable) -> BaseVariable:
    assert isinstance(y, BaseVariable), "Second argument must be a ShaderVariable"
    assert isinstance(x, BaseVariable), "First argument must be a ShaderVariable"

    assert dtypes.is_matrix(x.var_type), "First argument must be a matrix"
    assert dtypes.is_matrix(y.var_type), "Second argument must be a matrix"

    assert x.var_type == y.var_type, "Matrices must have the same shape"

    return x.new_var(
        x.var_type,
        f"matrixCompMult({resolve_input(x)}, {resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def outer_product(x: BaseVariable, y: BaseVariable) -> BaseVariable:
    assert isinstance(y, BaseVariable), "Second argument must be a ShaderVariable"
    assert isinstance(x, BaseVariable), "First argument must be a ShaderVariable"

    assert dtypes.is_vector(x.var_type), "First argument must be a matrix"
    assert dtypes.is_vector(y.var_type), "Second argument must be a matrix"

    assert x.var_type == y.var_type, "Matrices must have the same shape"

    out_type = None

    if x.var_type == dtypes.vec2:
        out_type = dtypes.mat2
    elif x.var_type == dtypes.vec3:
        out_type = dtypes.mat3
    elif x.var_type == dtypes.vec4:
        out_type = dtypes.mat4
    else:
        raise AssertionError("Unsupported vector type for outer product")

    return x.new_var(
        out_type,
        f"outerProduct({resolve_input(x)}, {resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def transpose(var: BaseVariable) ->BaseVariable:
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return var.new_var(
        var.var_type,
        f"transpose({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def determinant(var: BaseVariable) -> BaseVariable:
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return var.new_var(
        dtypes.float32,
        f"determinant({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def inverse(var: BaseVariable) -> BaseVariable:
    assert isinstance(var, BaseVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return var.new_var(
        var.var_type,
        f"inverse({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )