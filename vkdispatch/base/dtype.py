import numpy as np

from typing import Optional

class dtype:
    name: str
    item_size: int
    glsl_type: str
    glsl_type_extern: Optional[str] = None
    dimentions: int
    format_str: str
    child_type: "dtype"
    child_count: int
    scalar: "Optional[dtype]"
    shape: tuple
    numpy_shape: tuple
    true_numpy_shape: tuple

class _Scalar(dtype):
    dimentions = 0
    child_count = 0
    shape = (1,)
    numpy_shape = (1,)
    true_numpy_shape = ()
    scalar = None

class _I32(_Scalar):
    name = "int32"
    item_size = 4
    glsl_type = "int"
    format_str = "%d"

class _U32(_Scalar):
    name = "uint32"
    item_size = 4
    glsl_type = "uint"
    format_str = "%u"

class _F32(_Scalar):
    name = "float32"
    item_size = 4
    glsl_type = "float"
    format_str = "%f"

int32 = _I32 # type: ignore
uint32 = _U32 # type: ignore
float32 = _F32 # type: ignore

class _Complex(dtype):
    dimentions = 0
    child_count = 2

class _CF64(_Complex):
    name = "complex64"
    item_size = 8
    glsl_type = "vec2"
    format_str = "(%f, %f)"
    child_type = float32
    shape = (2,)
    numpy_shape = (1,)
    true_numpy_shape = ()
    scalar = None

complex64 = _CF64 # type: ignore

class _Vector(dtype):
    dimentions = 1

class _V2F32(_Vector):
    name = "vec2"
    item_size = 8
    glsl_type = "vec2"
    format_str = "(%f, %f)"
    child_type = float32
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = float32

class _V3F32(_Vector):
    name = "vec3"
    item_size = 16
    glsl_type = "vec3"
    glsl_type_extern = "vec4"
    format_str = "(%f, %f, %f)"
    child_type = float32
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = float32

class _V4F32(_Vector):
    name = "vec4"
    item_size = 16
    glsl_type = "vec4"
    format_str = "(%f, %f, %f, %f)"
    child_type = float32
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = float32

class _V2I32(_Vector):
    name = "ivec2"
    item_size = 8
    glsl_type = "ivec2"
    format_str = "(%d, %d)"
    child_type = int32
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = int32

class _V3I32(_Vector):
    name = "ivec3"
    item_size = 16
    glsl_type = "ivec3"
    glsl_type_extern = "ivec4"
    format_str = "(%d, %d, %d)"
    child_type = int32
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = int32

class _V4I32(_Vector):
    name = "ivec4"
    item_size = 16
    glsl_type = "ivec4"
    format_str = "(%d, %d, %d, %d)"
    child_type = int32
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = int32

class _V2U32(_Vector):
    name = "uvec2"
    item_size = 8
    glsl_type = "uvec2"
    format_str = "(%u, %u)"
    child_type = uint32
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = uint32

class _V3U32(_Vector):
    name = "uvec3"
    item_size = 16
    glsl_type = "uvec3"
    glsl_type_extern = "uvec4"
    format_str = "(%u, %u, %u)"
    child_type = uint32
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = uint32

class _V4U32(_Vector):
    name = "uvec4"
    item_size = 16
    glsl_type = "uvec4"
    format_str = "(%u, %u, %u, %u)"
    child_type = uint32
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = uint32

vec2 = _V2F32 # type: ignore
vec3 = _V3F32 # type: ignore
vec4 = _V4F32 # type: ignore
ivec2 = _V2I32 # type: ignore
ivec3 = _V3I32 # type: ignore
ivec4 = _V4I32 # type: ignore
uvec2 = _V2U32 # type: ignore
uvec3 = _V3U32 # type: ignore
uvec4 = _V4U32 # type: ignore

class _Matrix(dtype):
    dimentions = 2

class _M2F32(_Matrix):
    name = "mat2"
    item_size = 16
    glsl_type = "mat2"
    format_str = "\\\\n[%f, %f]\\\\n[%f, %f]\\\\n"
    child_type = vec2
    child_count = 2
    shape = (2, 2)
    numpy_shape = (2, 2)
    true_numpy_shape = (2, 2)
    scalar = float32

class _M4F32(_Matrix):
    name = "mat4"
    item_size = 64
    glsl_type = "mat4"
    format_str = "\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n"
    child_type = vec4
    child_count = 4
    shape = (4, 4)
    numpy_shape = (4, 4)
    true_numpy_shape = (4, 4)
    scalar = float32

mat2 = _M2F32
mat4 = _M4F32

def to_vector(dtype: dtype, count: int) -> dtype: # type: ignore
    if count < 2 or count > 4:
        raise ValueError(f"Unsupported count ({count})!")

    if dtype == int32:
        if count == 2:
            return ivec2
        elif count == 3:
            return ivec3
        elif count == 4:
            return ivec4
    elif dtype == uint32:
        if count == 2:
            return uvec2
        elif count == 3:
            return uvec3
        elif count == 4:
            return uvec4
    elif dtype == float32:
        if count == 2:
            return vec2
        elif count == 3:
            return vec3
        elif count == 4:
            return vec4
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")

def is_dtype(in_type: dtype) -> bool:
    return issubclass(in_type, dtype) # type: ignore

def is_scalar(dtype: dtype) -> bool:
    return issubclass(dtype, _Scalar) # type: ignore

def is_complex(dtype: dtype) -> bool:
    return issubclass(dtype, _Complex) # type: ignore

def is_vector(dtype: dtype) -> bool:
    return issubclass(dtype, _Vector) # type: ignore

def is_matrix(dtype: dtype) -> bool:
    return issubclass(dtype, _Matrix) # type: ignore

def from_numpy_dtype(dtype: type) -> dtype:
    if dtype == np.int32:
        return int32
    elif dtype == np.uint32:
        return uint32
    elif dtype == np.float32:
        return float32
    elif dtype == np.complex64:
        return complex64
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")

def to_numpy_dtype(shader_type: dtype) -> np.dtype:
    if shader_type == int32:
        return np.int32
    elif shader_type == uint32:
        return np.uint32
    elif shader_type == float32:
        return np.float32
    elif shader_type == complex64:
        return np.complex64
    else:
        raise ValueError(f"Unsupported shader_type ({shader_type})!")
