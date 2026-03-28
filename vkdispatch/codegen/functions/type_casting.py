import vkdispatch.base.dtype as dtypes
from typing import Optional

from . import utils
from ..variables.variables import ShaderVariable

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

def to_int64(*args):
    return to_dtype(dtypes.int64, *args)

def to_uint16(*args):
    return to_dtype(dtypes.uint16, *args)

def to_uint(*args):
    return to_dtype(dtypes.uint32, *args)

def to_uint64(*args):
    return to_dtype(dtypes.uint64, *args)

def _complex_from_real_arg(arg) -> dtypes.dtype:
    if isinstance(arg, ShaderVariable):
        if dtypes.is_complex(arg.var_type):
            return arg.var_type
        if dtypes.is_scalar(arg.var_type):
            return dtypes.complex_from_float(dtypes.make_floating_dtype(arg.var_type))
        raise TypeError(f"Unsupported variable type for complex conversion: {arg.var_type}")

    if utils.is_number(arg):
        base_type = utils.number_to_dtype(arg)
        if dtypes.is_complex(base_type):
            return base_type
        return dtypes.complex_from_float(dtypes.make_floating_dtype(base_type))

    raise TypeError(f"Unsupported argument type for complex conversion: {type(arg)}")

def _infer_complex_dtype(*args) -> dtypes.dtype:
    complex_type = _complex_from_real_arg(args[0])

    for arg in args[1:]:
        complex_type = dtypes.cross_type(complex_type, _complex_from_real_arg(arg))

    return complex_type

def _to_complex_dtype(var_type: dtypes.dtype, *args):
    if len(args) != 1 and len(args) != 2:
        raise ValueError("Must give one of two arguments for complex init")

    if len(args) == 1 and isinstance(args[0], ShaderVariable) and dtypes.is_complex(args[0].var_type):
        return to_dtype(var_type, args[0])

    if len(args) == 1:
        return to_dtype(var_type, args[0], 0)

    return to_dtype(var_type, *args)

def to_complex32(*args):
    return _to_complex_dtype(dtypes.complex32, *args)

def to_complex(*args):
    return _to_complex_dtype(_infer_complex_dtype(*args), *args)

def to_complex64(*args):
    return _to_complex_dtype(dtypes.complex64, *args)

def to_complex128(*args):
    return _to_complex_dtype(dtypes.complex128, *args)

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
