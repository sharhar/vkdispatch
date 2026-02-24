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

    utils.append_contents(f"{utils.backend_type_name(new_var.var_type)} {new_var.name} = {decleration};\n")

    return new_var

def new_float16_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.float16, *args, var_name=var_name)

def new_float_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.float32, *args, var_name=var_name)

def new_float64_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.float64, *args, var_name=var_name)

def new_int16_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.int16, *args, var_name=var_name)

def new_int_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.int32, *args, var_name=var_name)

def new_uint16_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uint16, *args, var_name=var_name)

def new_uint_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uint32, *args, var_name=var_name)

def new_complex_register(*args, var_name: Optional[str] = None):
    if len(args) > 0:
        true_args = (to_complex(*args),)
    else:
        true_args = (0,)

    return new_register(dtypes.complex64, *true_args, var_name=var_name)

def new_hvec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.hvec2, *args, var_name=var_name)

def new_hvec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.hvec3, *args, var_name=var_name)

def new_hvec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.hvec4, *args, var_name=var_name)

def new_vec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec2, *args, var_name=var_name)

def new_vec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec3, *args, var_name=var_name)

def new_vec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.vec4, *args, var_name=var_name)

def new_dvec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.dvec2, *args, var_name=var_name)

def new_dvec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.dvec3, *args, var_name=var_name)

def new_dvec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.dvec4, *args, var_name=var_name)

def new_ihvec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ihvec2, *args, var_name=var_name)

def new_ihvec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ihvec3, *args, var_name=var_name)

def new_ihvec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.ihvec4, *args, var_name=var_name)

def new_uhvec2_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uhvec2, *args, var_name=var_name)

def new_uhvec3_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uhvec3, *args, var_name=var_name)

def new_uhvec4_register(*args, var_name: Optional[str] = None):
    return new_register(dtypes.uhvec4, *args, var_name=var_name)

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
