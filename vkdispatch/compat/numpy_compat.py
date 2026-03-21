from __future__ import annotations

import builtins
import cmath
import math
import struct

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

try:
    import numpy as _np
except Exception:  # pragma: no cover - intentionally broad for optional dependency import
    _np = None

HAS_NUMPY = _np is not None
pi = math.pi


def require_numpy(feature_name: str) -> None:
    if HAS_NUMPY:
        return

    raise RuntimeError(
        f"{feature_name} requires numpy, but numpy is not available. "
        "Install numpy or use the bytes-based API."
    )


def numpy_module():
    return _np


def prod(values: Iterable[int]) -> int:
    values_tuple = tuple(values)

    if HAS_NUMPY:
        return int(_np.prod(values_tuple))

    result = 1
    for value in values_tuple:
        result *= int(value)
    return result


def ceil(value: float) -> float:
    if HAS_NUMPY:
        return float(_np.ceil(value))
    return float(math.ceil(value))

def round(value: float) -> float:
    if HAS_NUMPY:
        return float(_np.round(value))
    return float(builtins.round(value))

def angle(value: complex) -> float:
    if HAS_NUMPY:
        return float(_np.angle(value))
    return float(cmath.phase(value))


def exp_complex(value: complex) -> complex:
    if HAS_NUMPY:
        return complex(_np.exp(value))
    return cmath.exp(value)


def is_numpy_integer_scalar(value: Any) -> bool:
    return bool(HAS_NUMPY and _np.issubdtype(type(value), _np.integer))


def is_integer_scalar(value: Any) -> bool:
    return isinstance(value, int) or is_numpy_integer_scalar(value)


def is_numpy_floating_instance(value: Any) -> bool:
    return bool(HAS_NUMPY and isinstance(value, _np.floating))


@dataclass(frozen=True)
class HostDType:
    name: str
    itemsize: int
    struct_format: str
    kind: str


INT16 = HostDType("int16", 2, "h", "int")
UINT16 = HostDType("uint16", 2, "H", "uint")
INT32 = HostDType("int32", 4, "i", "int")
UINT32 = HostDType("uint32", 4, "I", "uint")
INT64 = HostDType("int64", 8, "q", "int")
UINT64 = HostDType("uint64", 8, "Q", "uint")
FLOAT16 = HostDType("float16", 2, "e", "float")
FLOAT32 = HostDType("float32", 4, "f", "float")
FLOAT64 = HostDType("float64", 8, "d", "float")
COMPLEX32 = HostDType("complex32", 4, "ee", "complex")
COMPLEX64 = HostDType("complex64", 8, "ff", "complex")
COMPLEX128 = HostDType("complex128", 16, "dd", "complex")

_HOST_DTYPES = {
    "int16": INT16,
    "uint16": UINT16,
    "int32": INT32,
    "uint32": UINT32,
    "int64": INT64,
    "uint64": UINT64,
    "float16": FLOAT16,
    "float32": FLOAT32,
    "float64": FLOAT64,
    "complex32": COMPLEX32,
    "complex64": COMPLEX64,
    "complex128": COMPLEX128,
}


def host_dtype(name: str) -> HostDType:
    if name not in _HOST_DTYPES:
        raise ValueError(f"Unsupported dtype ({name})!")
    return _HOST_DTYPES[name]


def is_host_dtype(value: Any) -> bool:
    return isinstance(value, HostDType)


def host_dtype_name(dtype: Any) -> str:
    if isinstance(dtype, HostDType):
        return dtype.name

    if isinstance(dtype, str):
        return dtype

    if HAS_NUMPY:
        return str(_np.dtype(dtype).name)

    raise ValueError(f"Unsupported dtype ({dtype})!")


def _numpy_dtype_or_none(dtype_name: str):
    if not HAS_NUMPY:
        return None

    try:
        return _np.dtype(dtype_name)
    except TypeError:
        return None


def dtype_itemsize(dtype: Any) -> int:
    if isinstance(dtype, HostDType):
        return dtype.itemsize

    if HAS_NUMPY:
        return int(_np.dtype(dtype).itemsize)

    return host_dtype(host_dtype_name(dtype)).itemsize


def dtype_kind(dtype: Any) -> str:
    if isinstance(dtype, HostDType):
        return dtype.kind

    if HAS_NUMPY:
        dtype_obj = _np.dtype(dtype)
        if _np.issubdtype(dtype_obj, _np.complexfloating):
            return "complex"
        if _np.issubdtype(dtype_obj, _np.unsignedinteger):
            return "uint"
        if _np.issubdtype(dtype_obj, _np.integer):
            return "int"
        if _np.issubdtype(dtype_obj, _np.floating):
            return "float"

    return host_dtype(host_dtype_name(dtype)).kind


def dtype_struct_format(dtype: Any) -> str:
    if isinstance(dtype, HostDType):
        return dtype.struct_format
    return host_dtype(host_dtype_name(dtype)).struct_format


class CompatArray:
    def __init__(self, buffer: bytes, dtype: HostDType, shape: Tuple[int, ...]):
        self._buffer = bytes(buffer)
        self.dtype = dtype
        self.shape = tuple(shape)
        self.size = prod(self.shape)

    def reshape(self, shape: Tuple[int, ...]) -> "CompatArray":
        shape = tuple(shape)
        if prod(shape) != self.size:
            raise ValueError("Cannot reshape array with mismatched element count")
        return CompatArray(self._buffer, self.dtype, shape)

    def tobytes(self) -> bytes:
        return bytes(self._buffer)

    @property
    def nbytes(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"CompatArray(shape={self.shape}, dtype={self.dtype.name}, nbytes={len(self._buffer)})"


def is_array_like(value: Any) -> bool:
    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return True
    return isinstance(value, CompatArray)


def array_shape(value: Any) -> Tuple[int, ...]:
    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return tuple(value.shape)
    if isinstance(value, CompatArray):
        return tuple(value.shape)
    raise TypeError(f"Unsupported array-like value ({type(value)})")


def array_dtype(value: Any) -> Any:
    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return value.dtype
    if isinstance(value, CompatArray):
        return value.dtype
    raise TypeError(f"Unsupported array-like value ({type(value)})")


def array_nbytes(value: Any) -> int:
    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return int(value.size * value.dtype.itemsize)
    if isinstance(value, CompatArray):
        return value.nbytes
    raise TypeError(f"Unsupported array-like value ({type(value)})")


def as_contiguous_bytes(value: Any) -> bytes:
    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return _np.ascontiguousarray(value).tobytes()
    if isinstance(value, CompatArray):
        return value.tobytes()
    raise TypeError(f"Unsupported array-like value ({type(value)})")


def from_buffer(buffer: bytes, dtype: Any, shape: Tuple[int, ...]):
    dtype_name = host_dtype_name(dtype)

    if HAS_NUMPY:
        np_dtype = _numpy_dtype_or_none(dtype_name)
        if np_dtype is not None:
            return _np.frombuffer(buffer, dtype=np_dtype).reshape(shape)

        if dtype_name == "complex32":
            half_pairs = _np.frombuffer(buffer, dtype=_np.float16).reshape(*shape, 2)
            return half_pairs[..., 0].astype(_np.float32) + (1j * half_pairs[..., 1].astype(_np.float32))

    return CompatArray(buffer, host_dtype(dtype_name), tuple(shape))


def ensure_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"Unsupported bytes-like object ({type(value)})")


def is_bytes_like(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def flatten(value: Any) -> List[Any]:
    if isinstance(value, CompatArray):
        return unpack_values(value.tobytes(), value.dtype)

    if HAS_NUMPY and isinstance(value, _np.ndarray):
        return value.reshape(-1).tolist()

    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for element in value:
            out.extend(flatten(element))
        return out

    return [value]


def _coerce_scalar(value: Any, dtype: Any):
    kind = dtype_kind(dtype)

    if kind == "complex":
        if isinstance(value, complex):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("Complex values must be complex scalars or pairs")
            return complex(float(value[0]), float(value[1]))
        return complex(value)

    if kind == "float":
        return float(value)

    if kind in ("int", "uint"):
        return int(value)

    raise ValueError(f"Unsupported dtype kind ({kind})")


def pack_values(values: Sequence[Any], dtype: Any) -> bytes:
    values_list = list(values)
    dtype_name = host_dtype_name(dtype)

    if HAS_NUMPY:
        np_dtype = _numpy_dtype_or_none(dtype_name)
        if np_dtype is not None:
            array = _np.asarray(values_list, dtype=np_dtype)
            return array.tobytes()

    host = host_dtype(dtype_name)

    if host.kind == "complex":
        output = bytearray()
        pack_fmt = "=" + host.struct_format
        for value in values_list:
            coerced = _coerce_scalar(value, host)
            output.extend(struct.pack(pack_fmt, float(coerced.real), float(coerced.imag)))
        return bytes(output)

    pack_fmt = "=" + host.struct_format
    output = bytearray()
    for value in values_list:
        output.extend(struct.pack(pack_fmt, _coerce_scalar(value, host)))
    return bytes(output)


def unpack_values(data: bytes, dtype: Any) -> List[Any]:
    dtype_name = host_dtype_name(dtype)

    if HAS_NUMPY:
        np_dtype = _numpy_dtype_or_none(dtype_name)
        if np_dtype is not None:
            return _np.frombuffer(data, dtype=np_dtype).tolist()

    host = host_dtype(dtype_name)

    if host.kind == "complex":
        values: List[Any] = []
        unpack_fmt = "=" + host.struct_format
        for real, imag in struct.iter_unpack(unpack_fmt, data):
            values.append(complex(real, imag))
        return values

    unpack_fmt = "=" + host.struct_format
    stride = struct.calcsize(unpack_fmt)
    values = []

    for offset in range(0, len(data), stride):
        values.append(struct.unpack(unpack_fmt, data[offset: offset + stride])[0])

    return values

