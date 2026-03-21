from typing import Any, Optional

from ..compat import numpy_compat as npc

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

class _I16(_Scalar):
    name = "int16"
    item_size = 2
    glsl_type = "int16_t"
    format_str = "%d"

class _U16(_Scalar):
    name = "uint16"
    item_size = 2
    glsl_type = "uint16_t"
    format_str = "%u"

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

class _I64(_Scalar):
    name = "int64"
    item_size = 8
    glsl_type = "int64_t"
    format_str = "%lld"

class _U64(_Scalar):
    name = "uint64"
    item_size = 8
    glsl_type = "uint64_t"
    format_str = "%llu"

class _F16(_Scalar):
    name = "float16"
    item_size = 2
    glsl_type = "float16_t"
    format_str = "%f"

class _F32(_Scalar):
    name = "float32"
    item_size = 4
    glsl_type = "float"
    format_str = "%f"

class _F64(_Scalar):
    name = "float64"
    item_size = 8
    glsl_type = "double"
    format_str = "%lf"

int16 = _I16 # type: ignore
uint16 = _U16 # type: ignore
int32 = _I32 # type: ignore
uint32 = _U32 # type: ignore
int64 = _I64 # type: ignore
uint64 = _U64 # type: ignore
float16 = _F16 # type: ignore
float32 = _F32 # type: ignore
float64 = _F64 # type: ignore

class _Complex(dtype):
    dimentions = 0
    child_count = 2

class _CF32(_Complex):
    name = "complex32"
    item_size = 4
    glsl_type = "f16vec2"
    format_str = "(%f, %f)"
    child_type = float16
    shape = (2,)
    numpy_shape = (1,)
    true_numpy_shape = ()
    scalar = None

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

class _CF128(_Complex):
    name = "complex128"
    item_size = 16
    glsl_type = "dvec2"
    format_str = "(%lf, %lf)"
    child_type = float64
    shape = (2,)
    numpy_shape = (1,)
    true_numpy_shape = ()
    scalar = None

complex32 = _CF32 # type: ignore
complex64 = _CF64 # type: ignore
complex128 = _CF128 # type: ignore

class _Vector(dtype):
    dimentions = 1

# --- float16 vectors ---

class _V2F16(_Vector):
    name = "hvec2"
    item_size = 4
    glsl_type = "f16vec2"
    format_str = "(%f, %f)"
    child_type = float16
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = float16

class _V3F16(_Vector):
    name = "hvec3"
    item_size = 6
    glsl_type = "f16vec3"
    format_str = "(%f, %f, %f)"
    child_type = float16
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = float16

class _V4F16(_Vector):
    name = "hvec4"
    item_size = 8
    glsl_type = "f16vec4"
    format_str = "(%f, %f, %f, %f)"
    child_type = float16
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = float16

# --- float32 vectors ---

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

# --- float64 vectors ---

class _V2F64(_Vector):
    name = "dvec2"
    item_size = 16
    glsl_type = "dvec2"
    format_str = "(%lf, %lf)"
    child_type = float64
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = float64

class _V3F64(_Vector):
    name = "dvec3"
    item_size = 24
    glsl_type = "dvec3"
    format_str = "(%lf, %lf, %lf)"
    child_type = float64
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = float64

class _V4F64(_Vector):
    name = "dvec4"
    item_size = 32
    glsl_type = "dvec4"
    format_str = "(%lf, %lf, %lf, %lf)"
    child_type = float64
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = float64

# --- int16 vectors ---

class _V2I16(_Vector):
    name = "ihvec2"
    item_size = 4
    glsl_type = "i16vec2"
    format_str = "(%d, %d)"
    child_type = int16
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = int16

class _V3I16(_Vector):
    name = "ihvec3"
    item_size = 6
    glsl_type = "i16vec3"
    format_str = "(%d, %d, %d)"
    child_type = int16
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = int16

class _V4I16(_Vector):
    name = "ihvec4"
    item_size = 8
    glsl_type = "i16vec4"
    format_str = "(%d, %d, %d, %d)"
    child_type = int16
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = int16

# --- int32 vectors ---

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

# --- uint16 vectors ---

class _V2U16(_Vector):
    name = "uhvec2"
    item_size = 4
    glsl_type = "u16vec2"
    format_str = "(%u, %u)"
    child_type = uint16
    child_count = 2
    shape = (2,)
    numpy_shape = (2,)
    true_numpy_shape = (2,)
    scalar = uint16

class _V3U16(_Vector):
    name = "uhvec3"
    item_size = 6
    glsl_type = "u16vec3"
    format_str = "(%u, %u, %u)"
    child_type = uint16
    child_count = 3
    shape = (3,)
    numpy_shape = (3,)
    true_numpy_shape = (3,)
    scalar = uint16

class _V4U16(_Vector):
    name = "uhvec4"
    item_size = 8
    glsl_type = "u16vec4"
    format_str = "(%u, %u, %u, %u)"
    child_type = uint16
    child_count = 4
    shape = (4,)
    numpy_shape = (4,)
    true_numpy_shape = (4,)
    scalar = uint16

# --- uint32 vectors ---

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

hvec2 = _V2F16 # type: ignore
hvec3 = _V3F16 # type: ignore
hvec4 = _V4F16 # type: ignore
vec2 = _V2F32 # type: ignore
vec3 = _V3F32 # type: ignore
vec4 = _V4F32 # type: ignore
dvec2 = _V2F64 # type: ignore
dvec3 = _V3F64 # type: ignore
dvec4 = _V4F64 # type: ignore
ihvec2 = _V2I16 # type: ignore
ihvec3 = _V3I16 # type: ignore
ihvec4 = _V4I16 # type: ignore
ivec2 = _V2I32 # type: ignore
ivec3 = _V3I32 # type: ignore
ivec4 = _V4I32 # type: ignore
uhvec2 = _V2U16 # type: ignore
uhvec3 = _V3U16 # type: ignore
uhvec4 = _V4U16 # type: ignore
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

# Maps scalar dtype -> {count: vector_dtype}
_VECTOR_TABLE = {
    int16: {1: int16, 2: ihvec2, 3: ihvec3, 4: ihvec4},
    uint16: {1: uint16, 2: uhvec2, 3: uhvec3, 4: uhvec4},
    int32: {1: int32, 2: ivec2, 3: ivec3, 4: ivec4},
    uint32: {1: uint32, 2: uvec2, 3: uvec3, 4: uvec4},
    float16: {1: float16, 2: hvec2, 3: hvec3, 4: hvec4},
    float32: {1: float32, 2: vec2, 3: vec3, 4: vec4},
    float64: {1: float64, 2: dvec2, 3: dvec3, 4: dvec4},
}

def to_vector(dtype: dtype, count: int) -> dtype: # type: ignore
    if count < 1 or count > 4:
        raise ValueError(f"Unsupported count ({count})!")

    table = _VECTOR_TABLE.get(dtype)
    if table is None:
        raise ValueError(f"Unsupported dtype ({dtype})!")
    return table[count]

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

    return dtype == float16 or dtype == float32 or dtype == float64

def is_integer_dtype(dtype: dtype) -> bool:
    if not is_scalar(dtype):
        dtype = dtype.scalar

    return dtype in (int16, uint16, int32, uint32, int64, uint64)

# Promotion precedence: float64 > float32 > float16 > int64 > int32 > int16 > uint64 > uint32 > uint16
_SCALAR_RANK = {
    uint16: 0,
    int16: 1,
    uint32: 2,
    int32: 3,
    uint64: 4,
    int64: 5,
    float16: 6,
    float32: 7,
    float64: 8,
}

_COMPLEX_FROM_FLOAT = {
    float16: complex32,
    float32: complex64,
    float64: complex128,
}

def complex_from_float(dtype: dtype) -> dtype:
    if not is_scalar(dtype):
        raise ValueError(f"Unsupported dtype ({dtype})!")

    result = _COMPLEX_FROM_FLOAT.get(dtype)
    if result is None:
        raise ValueError(f"Unsupported complex base dtype ({dtype})!")
    return result

def _promote_scalar(dtype: dtype) -> dtype:
    """Return the floating-point type that matches the width of *dtype*.

    Used by make_floating_dtype to convert integer scalars to their natural
    floating counterpart.
    """
    if dtype == int16 or dtype == uint16:
        return float16
    if dtype == int32 or dtype == uint32:
        return float32
    if dtype == int64 or dtype == uint64:
        return float64
    return dtype

def make_floating_dtype(dtype: dtype) -> dtype:
    if is_scalar(dtype):
        return _promote_scalar(dtype)
    elif is_vector(dtype):
        return to_vector(_promote_scalar(dtype.scalar), dtype.child_count)
    elif is_matrix(dtype):
        return dtype
    elif is_complex(dtype):
        return dtype
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")

def vector_size(dtype: dtype) -> int:
    if not is_vector(dtype):
        raise ValueError(f"Type ({dtype}) is not a vector!")

    return dtype.child_count

def cross_scalar_scalar(dtype1: dtype, dtype2: dtype) -> dtype:
    assert is_scalar(dtype1) and is_scalar(dtype2), "Both types must be scalar types!"

    r1 = _SCALAR_RANK[dtype1]
    r2 = _SCALAR_RANK[dtype2]
    return dtype1 if r1 >= r2 else dtype2

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
        if is_complex(dtype2):
            return complex_from_float(cross_scalar_scalar(dtype1.child_type, dtype2.child_type))
        if is_scalar(dtype2):
            return complex_from_float(cross_scalar_scalar(dtype1.child_type, _promote_scalar(dtype2)))
        raise ValueError("Cannot cross complex and non-scalar types!")
    elif is_complex(dtype2):
        if is_scalar(dtype1):
            return complex_from_float(cross_scalar_scalar(dtype2.child_type, _promote_scalar(dtype1)))
        raise ValueError("Cannot cross complex and non-scalar types!")

    if is_scalar(dtype1) and is_scalar(dtype2):
        return cross_scalar_scalar(dtype1, dtype2)

def cross_multiply_type(dtype1: dtype, dtype2: dtype) -> dtype:
    """Resolve result type for multiplication.

    Unlike ``cross_type``, multiplication is order-sensitive for matrix/vector
    combinations and supports ``matN * vecN`` and ``vecN * matN``.
    """
    if is_matrix(dtype1) and is_vector(dtype2):
        if dtype1.child_count != dtype2.child_count:
            raise ValueError(
                f"Cannot multiply matrix '{dtype1.name}' and vector '{dtype2.name}' with incompatible dimensions!"
            )
        if dtype1.scalar != float32 or dtype2.scalar != float32:
            raise ValueError("Matrix/vector multiplication only supports float32 matrix and vector types.")
        return dtype2

    if is_vector(dtype1) and is_matrix(dtype2):
        if dtype1.child_count != dtype2.child_count:
            raise ValueError(
                f"Cannot multiply vector '{dtype1.name}' and matrix '{dtype2.name}' with incompatible dimensions!"
            )
        if dtype1.scalar != float32 or dtype2.scalar != float32:
            raise ValueError("Matrix/vector multiplication only supports float32 matrix and vector types.")
        return dtype1

    return cross_type(dtype1, dtype2)

def from_numpy_dtype(dtype: Any) -> dtype:
    dtype_name = npc.host_dtype_name(dtype)

    _NAME_MAP = {
        "int16": int16,
        "uint16": uint16,
        "int32": int32,
        "uint32": uint32,
        "int64": int64,
        "uint64": uint64,
        "float16": float16,
        "float32": float32,
        "float64": float64,
        "complex32": complex32,
        "complex64": complex64,
        "complex128": complex128,
    }

    result = _NAME_MAP.get(dtype_name)
    if result is None:
        raise ValueError(f"Unsupported dtype ({dtype})!")
    return result


def to_numpy_dtype(shader_type: dtype) -> Any:
    _TYPE_MAP = {
        int16: "int16",
        uint16: "uint16",
        int32: "int32",
        uint32: "uint32",
        int64: "int64",
        uint64: "uint64",
        float16: "float16",
        float32: "float32",
        float64: "float64",
        complex32: "complex32",
        complex64: "complex64",
        complex128: "complex128",
    }

    name = _TYPE_MAP.get(shader_type)
    if name is None:
        raise ValueError(f"Unsupported shader_type ({shader_type})!")

    if npc.HAS_NUMPY and hasattr(npc.numpy_module(), name):
        return getattr(npc.numpy_module(), name)
    return npc.host_dtype(name)
