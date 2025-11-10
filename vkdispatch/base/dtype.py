import numpy as np

from typing import Optional

class dtype:
    name: str
    item_size: int
    glsl_type: str
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
    child_type = None
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
    item_size = 12
    glsl_type = "vec3"
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
    item_size = 12
    glsl_type = "ivec3"
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
    item_size = 12
    glsl_type = "uvec3"
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

class _M3F32(_Matrix):
    name = "mat3"
    item_size = 36
    glsl_type = "mat3"
    format_str = "\\\\n[%f, %f, %f]\\\\n[%f, %f, %f]\\\\n[%f, %f, %f]\\\\n"
    child_type = vec3
    child_count = 3
    shape = (3, 3)
    numpy_shape = (3, 3)
    true_numpy_shape = (3, 3)
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
mat3 = _M3F32
mat4 = _M4F32

def to_vector(dtype: dtype, count: int) -> dtype: # type: ignore
    if count < 1 or count > 4:
        raise ValueError(f"Unsupported count ({count})!")

    if dtype == int32:
        if count == 1:
            return int32
        elif count == 2:
            return ivec2
        elif count == 3:
            return ivec3
        elif count == 4:
            return ivec4
    elif dtype == uint32:
        if count == 1:
            return uint32
        elif count == 2:
            return uvec2
        elif count == 3:
            return uvec3
        elif count == 4:
            return uvec4
    elif dtype == float32:
        if count == 1:
            return float32
        elif count == 2:
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

def is_float_dtype(dtype: dtype) -> bool:
    if not is_scalar(dtype):
        dtype = dtype.scalar

    return dtype == float32 # or dtype == complex64

def is_integer_dtype(dtype: dtype) -> bool:
    if not is_scalar(dtype):
        dtype = dtype.scalar

    return dtype == int32 or dtype == uint32

def make_floating_dtype(dtype: dtype) -> dtype:
    if is_scalar(dtype):
        return float32
    elif is_vector(dtype):
        return to_vector(float32, dtype.child_count)
    elif is_matrix(dtype):
        return dtype
    elif is_complex(dtype):
        return complex64
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")

def vector_size(dtype: dtype) -> int:
    if not is_vector(dtype):
        raise ValueError(f"Type ({dtype}) is not a vector!")

    return dtype.child_count

def cross_scalar_scalar(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_scalar(dtype1) and is_scalar(dtype2), "Both types must be scalar types!"
    
    if dtype1 == float32 or dtype2 == float32:
        return float32
    
    if dtype1 == int32 or dtype2 == int32:
        return int32
    
    return uint32

def cross_vector_scalar(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_vector(dtype1) and is_scalar(dtype2), "First type must be vector type and second type must be scalar type!"

    return to_vector(cross_scalar_scalar(dtype1.scalar, dtype2), dtype1.child_count)

def cross_vector_vector(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_vector(dtype1) and is_vector(dtype2), "Both types must be vector types!"

    if dtype1.child_count != dtype2.child_count:
        raise ValueError(f"Cannot cross types of vectors of two sizes! ({dtype1.child_count} != {dtype2.child_count})")

    return to_vector(cross_scalar_scalar(dtype1.scalar, dtype2.scalar), dtype1.child_count)

def cross_vector(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_vector(dtype1), "First type must be vector type!"

    if is_vector(dtype2):
        return cross_vector_vector(dtype1, dtype2)
    elif is_scalar(dtype2):
        return cross_vector_scalar(dtype1, dtype2)
    elif is_complex(dtype2):
        raise ValueError("Cannot cross vector and complex types!")
    else:
        raise ValueError("Second type must be vector or scalar type!")

def cross_matrix(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_matrix(dtype1), "Both types must be matrix types!"

    if is_matrix(dtype2):
        if dtype1.shape != dtype2.shape:
            raise ValueError(
                f"Cannot cross types of matrices with incompatible shapes! ({dtype1.shape} and {dtype2.shape})")

        return dtype1

    if is_vector(dtype2) or is_complex(dtype2):
        raise ValueError("Cannot cross matrix and vector/complex types!")
    
    if is_scalar(dtype2):
        return dtype1
    
    raise ValueError("Second type must be matrix or scalar type!")

def cross_type(dtype1: dtype, dtype2: dtype) -> dtype:
    if is_matrix(dtype1):
        return cross_matrix(dtype1, dtype2)
    elif is_matrix(dtype2):
        return cross_matrix(dtype2, dtype1)

    if is_vector(dtype1):
        return cross_vector(dtype1, dtype2)
    elif is_vector(dtype2):
        return cross_vector(dtype2, dtype1)
    
    if is_complex(dtype1):
        return complex64
    elif is_complex(dtype2):
        return complex64
    
    if is_scalar(dtype1) and is_scalar(dtype2):
        return cross_scalar_scalar(dtype1, dtype2)

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
