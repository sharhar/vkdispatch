import vkdispatch.base.dtype as dtypes
from typing import Optional

from . import utils

def to_dtype(var_type: dtypes.dtype, *args):
    return utils.new_var(
        var_type,
        utils.backend_constructor(var_type, *args),
        args,
        lexical_unit=True
    )

def str_to_dtype(var_type: dtypes.dtype,
                 value: str,
                 parents: Optional[list] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 register: bool = False):
    return utils.new_var(
        var_type,
        value,
        parents=parents if parents is not None else [],
        lexical_unit=lexical_unit,
        settable=settable,
        register=register
    )

def to_float16(*args):
    return to_dtype(dtypes.float16, *args)

def to_float(*args):
    return to_dtype(dtypes.float32, *args)

def to_float64(*args):
    return to_dtype(dtypes.float64, *args)

def to_int16(*args):
    return to_dtype(dtypes.int16, *args)

def to_int(*args):
    return to_dtype(dtypes.int32, *args)

def to_uint16(*args):
    return to_dtype(dtypes.uint16, *args)

def to_uint(*args):
    return to_dtype(dtypes.uint32, *args)

def to_complex(*args):
    assert len(args) == 1 or len(args) == 2, "Must give one of two arguments for complex init"

    if len(args) == 1:
        return to_dtype(dtypes.complex64, args[0], 0)

    return to_dtype(dtypes.complex64, *args)

def to_hvec2(*args):
    return to_dtype(dtypes.hvec2, *args)

def to_hvec3(*args):
    return to_dtype(dtypes.hvec3, *args)

def to_hvec4(*args):
    return to_dtype(dtypes.hvec4, *args)

def to_vec2(*args):
    return to_dtype(dtypes.vec2, *args)

def to_vec3(*args):
    return to_dtype(dtypes.vec3, *args)

def to_vec4(*args):
    return to_dtype(dtypes.vec4, *args)

def to_dvec2(*args):
    return to_dtype(dtypes.dvec2, *args)

def to_dvec3(*args):
    return to_dtype(dtypes.dvec3, *args)

def to_dvec4(*args):
    return to_dtype(dtypes.dvec4, *args)

def to_ihvec2(*args):
    return to_dtype(dtypes.ihvec2, *args)

def to_ihvec3(*args):
    return to_dtype(dtypes.ihvec3, *args)

def to_ihvec4(*args):
    return to_dtype(dtypes.ihvec4, *args)

def to_ivec2(*args):
    return to_dtype(dtypes.ivec2, *args)

def to_ivec3(*args):
    return to_dtype(dtypes.ivec3, *args)

def to_ivec4(*args):
    return to_dtype(dtypes.ivec4, *args)

def to_uhvec2(*args):
    return to_dtype(dtypes.uhvec2, *args)

def to_uhvec3(*args):
    return to_dtype(dtypes.uhvec3, *args)

def to_uhvec4(*args):
    return to_dtype(dtypes.uhvec4, *args)

def to_uvec2(*args):
    return to_dtype(dtypes.uvec2, *args)

def to_uvec3(*args):
    return to_dtype(dtypes.uvec3, *args)

def to_uvec4(*args):
    return to_dtype(dtypes.uvec4, *args)

def to_mat2(*args):
    return to_dtype(dtypes.mat2, *args)

def to_mat3(*args):
    return to_dtype(dtypes.mat3, *args)

def to_mat4(*args):
    return to_dtype(dtypes.mat4, *args)
