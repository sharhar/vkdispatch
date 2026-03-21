from .buffer import Buffer
from . import dtype as dt
from typing import Tuple

def buffer_u32(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 32-bit integers with the specified shape."""
    return Buffer(shape, dt.uint32)

def buffer_uv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 32-bit integer vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.uvec2)

def buffer_uv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 32-bit integer vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.uvec3)

def buffer_uv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 32-bit integer vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.uvec4)

def buffer_i32(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 32-bit integers with the specified shape."""
    return Buffer(shape, dt.int32)

def buffer_iv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 32-bit integer vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.ivec2)

def buffer_iv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 32-bit integer vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.ivec3)

def buffer_iv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 32-bit integer vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.ivec4)

def buffer_f32(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 32-bit floating-point numbers with the specified shape."""
    return Buffer(shape, dt.float32)

def buffer_v2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 32-bit floating-point vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.vec2)

def buffer_v3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 32-bit floating-point vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.vec3)

def buffer_v4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 32-bit floating-point vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.vec4)

def buffer_c64(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 64-bit complex numbers with the specified shape."""
    return Buffer(shape, dt.complex64)

def buffer_u16(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 16-bit integers with the specified shape."""
    return Buffer(shape, dt.uint16)

def buffer_uhv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 16-bit integer vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.uhvec2)

def buffer_uhv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 16-bit integer vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.uhvec3)

def buffer_uhv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of unsigned 16-bit integer vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.uhvec4)

def buffer_i16(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 16-bit integers with the specified shape."""
    return Buffer(shape, dt.int16)

def buffer_ihv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 16-bit integer vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.ihvec2)

def buffer_ihv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 16-bit integer vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.ihvec3)

def buffer_ihv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of signed 16-bit integer vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.ihvec4)

def buffer_f16(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 16-bit floating-point numbers with the specified shape."""
    return Buffer(shape, dt.float16)

def buffer_hv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 16-bit floating-point vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.hvec2)

def buffer_hv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 16-bit floating-point vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.hvec3)

def buffer_hv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 16-bit floating-point vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.hvec4)

def buffer_f64(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 64-bit floating-point numbers with the specified shape."""
    return Buffer(shape, dt.float64)

def buffer_dv2(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 64-bit floating-point vectors of size 2 with the specified shape."""
    return Buffer(shape, dt.dvec2)

def buffer_dv3(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 64-bit floating-point vectors of size 3 with the specified shape."""
    return Buffer(shape, dt.dvec3)

def buffer_dv4(shape: Tuple[int, ...]) -> Buffer:
    """Create a buffer of 64-bit floating-point vectors of size 4 with the specified shape."""
    return Buffer(shape, dt.dvec4)