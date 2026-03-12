import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable

from . import utils

def matrix_comp_mult(x: ShaderVariable, y: ShaderVariable) -> ShaderVariable:
    assert isinstance(y, ShaderVariable), "Second argument must be a ShaderVariable"
    assert isinstance(x, ShaderVariable), "First argument must be a ShaderVariable"

    assert dtypes.is_matrix(x.var_type), "First argument must be a matrix"
    assert dtypes.is_matrix(y.var_type), "Second argument must be a matrix"

    assert x.var_type == y.var_type, "Matrices must have the same shape"

    return utils.new_var(
        x.var_type,
        f"matrixCompMult({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def outer_product(x: ShaderVariable, y: ShaderVariable) -> ShaderVariable:
    assert isinstance(y, ShaderVariable), "Second argument must be a ShaderVariable"
    assert isinstance(x, ShaderVariable), "First argument must be a ShaderVariable"

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

    return utils.new_var(
        out_type,
        f"outerProduct({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def transpose(var: ShaderVariable) ->ShaderVariable:
    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return utils.new_var(
        var.var_type,
        f"transpose({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def determinant(var: ShaderVariable) -> ShaderVariable:
    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return utils.new_var(
        dtypes.float32,
        f"determinant({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def inverse(var: ShaderVariable) -> ShaderVariable:
    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable"

    assert dtypes.is_matrix(var.var_type), "Argument must be a matrix"

    return utils.new_var(
        var.var_type,
        f"inverse({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )