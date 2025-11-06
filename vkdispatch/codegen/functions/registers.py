import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Optional

from . import utils

from .type_casting import to_dtype, to_complex

def new_register(var_type: dtypes.dtype, *args, var_name: Optional[str] = None):
    new_var = utils.new_var(
        var_type,
        var_name,
        [],
        lexical_unit=True,
        settable=True,
        register=True
    )

    for arg in args:
        if isinstance(arg, ShaderVariable):
            arg.read_callback()

    if len(args) == 0:
        args = (0,)

    decleration = to_dtype(var_type, *args).resolve()

    utils.append_contents(f"{new_var.var_type.glsl_type} {new_var.name} = {decleration};\n")

    return new_var

def new_float_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.float32, *args, var_name=var_name)

def new_int_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.int32, *args, var_name=var_name)

def new_uint_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uint32, *args, var_name=var_name)

def new_complex_register(*args, var_name: Optional[str] = None):
    if len(args) > 0:
        true_args = to_complex(*args)
    else:
        true_args = (0,)

    return new_register(dtypes.complex64, *true_args, var_name=var_name)

def new_vec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec2, *args, var_name=var_name)

def new_vec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec3, *args, var_name=var_name)

def new_vec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec4, *args, var_name=var_name)

def new_uvec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uvec2, *args, var_name=var_name)

def new_uvec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uvec3, *args, var_name=var_name)

def new_uvec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uvec4, *args, var_name=var_name)

def new_ivec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ivec2, *args, var_name=var_name)

def new_ivec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ivec3, *args, var_name=var_name)

def new_ivec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ivec4, *args, var_name=var_name)

def new_mat2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.mat2, *args, var_name=var_name)

def new_mat3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.mat3, *args, var_name=var_name)

def new_mat4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.mat4, *args, var_name=var_name)