from typing import Tuple
from typing import List
from typing import Union

from .dtype import dtype
from .context import Handle, Signal
from .errors import check_for_errors

from .dtype import complex64

from .._compat import numpy_compat as npc
from .dtype import to_numpy_dtype, from_numpy_dtype

from .backend import native

import typing

_ArgType = typing.TypeVar('_ArgType', bound=dtype)

import dataclasses

@dataclasses.dataclass
class ExternalBufferInfo:
    writable: bool
    iface: dict
    keepalive: bool
    cuda_ptr: int

class Buffer(Handle, typing.Generic[_ArgType]):
    """
    Represents a contiguous block of memory on the GPU (or shared across multiple devices).

    Buffers are the primary mechanism for transferring data between the host (CPU) 
    and the device (GPU). They are typed using ``vkdispatch.dtype`` and support 
    multi-dimensional shapes, similar to NumPy arrays.

    :param shape: The dimensions of the buffer. Must be a tuple of 1, 2, or 3 integers.
    :type shape: Tuple[int, ...]
    :param var_type: The data type of the elements stored in the buffer.
    :type var_type: vkdispatch.base.dtype.dtype
    :raises ValueError: If the shape has more than 3 dimensions or if the requested size exceeds 2^30 elements.
    """

    var_type: dtype
    shape: Tuple[int]
    size: int
    mem_size: int
    signals: List[Signal]
    is_external: bool
    owns_memory: bool
    is_writable: bool
    cuda_ptr: typing.Optional[int]
    cuda_source: typing.Any
    cuda_array_stream: typing.Optional[typing.Any]

    def __init__(self, shape: Tuple[int, ...], var_type: dtype, external_buffer: ExternalBufferInfo = None) -> None:
        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)

        if len(shape) > 3:
            raise ValueError("Buffer shape must be 1, 2, or 3 dimensions!")

        self.var_type: dtype = var_type
        self.shape: Tuple[int] = shape

        size = 1
        for dim in shape:
            size *= dim
        self.size = size

        self.mem_size: int = self.size * self.var_type.item_size

        if self.size > 2 ** 30:
            raise ValueError("Cannot allocate buffers larger than 2^30 elements!")

        shader_shape_internal = [1, 1, 1, 1]

        for ii, axis_size in enumerate(shape):
            shader_shape_internal[ii] = axis_size
        
        shader_shape_internal[3] = shader_shape_internal[0] * shader_shape_internal[1] * shader_shape_internal[2]

        self.shader_shape = tuple(shader_shape_internal)

        self.signals = []
        self.is_external = external_buffer is not None
        self.owns_memory = external_buffer is None
        self.is_writable = True if external_buffer is None else external_buffer.writable
        self.cuda_ptr = None if external_buffer is None else external_buffer.cuda_ptr
        self.cuda_source = None if external_buffer is None else (external_buffer.iface if external_buffer.keepalive else None)
        self.cuda_array_stream = None if external_buffer is None else external_buffer.iface.get("stream")

        if external_buffer is not None:
            handle = native.buffer_create_external(
                self.context._handle,
                self.mem_size,
                self.cuda_ptr,
            )
        else:
            handle = native.buffer_create(
                self.context._handle, self.mem_size, 0
            )
        check_for_errors()

        self.signals = [
            Signal(
                native.buffer_get_queue_signal(
                    handle, queue_index
                )
            )
            for queue_index in range(self.context.queue_count)
        ]

        self.register_handle(handle)

    def __repr__(self):
        return f"""Buffer {self._handle}:
    shape={self.shape}
    var_type={self.var_type.name}
    mem_size={self.mem_size} bytes
    is_external={self.is_external}
    writable={self.is_writable}
    cuda_ptr={self.cuda_ptr}
    cuda_iface={self.cuda_source}
"""

    def _destroy(self) -> None:
        """Destroy the buffer and all child handles."""

        for ii, signal in enumerate(self.signals):
            signal.wait(False, ii)

        native.buffer_destroy(self._handle)

    def __del__(self) -> None:
        self.destroy()

    def _wait_staging_idle(self, index: int):
        is_idle = native.buffer_wait_staging_idle(self._handle, index)
        check_for_errors()
        return is_idle

    def _do_writes(self, data: bytes, index: int = None):
        indicies = [index] if index is not None else range(self.context.queue_count)
        completed_stages = [0] * len(indicies)

        while not all(stage == 1 for stage in completed_stages):
            for i in range(len(indicies)):
                if completed_stages[i] == 1:
                    continue

                queue_index = indicies[i]

                if not self.signals[queue_index].try_wait(True, queue_index):
                    continue

                completed_stages[i] = 1

                native.buffer_write_staging(self._handle, queue_index, data, len(data))
                check_for_errors()

                native.buffer_write(self._handle, 0, len(data), queue_index)
                check_for_errors()

    def write(self, data: Union[bytes, bytearray, memoryview, typing.Any], index: int = None) -> None:
        """
        Uploads data from the host to the GPU buffer.

        If ``index`` is None, the data is broadcast to the memory of all active devices 
        in the context. Otherwise, it writes only to the device specified by the index.

        :param data: The source data. Can be a bytes-like object or an array-like object.
        :type data: Union[bytes, bytearray, memoryview, Any]
        :param index: The device index to write to. Defaults to -1 (all devices).
        :type index: int
        :raises ValueError: If the data size exceeds the buffer size or if the index is invalid.
        """
        if index is not None:
            assert isinstance(index, int), "Index must be an integer or None!"
            assert index >= 0 and index < self.context.queue_count, "Index must be valid!"

        if not getattr(self, "is_writable", True):
            raise ValueError("Cannot write to a read-only buffer alias.")

        true_data_object = None

        if npc.is_array_like(data):
            if npc.array_nbytes(data) != self.mem_size:
                raise ValueError("Numpy buffer sizes must match!")

            true_data_object = npc.as_contiguous_bytes(data)
        else:
            true_data_object = npc.ensure_bytes(data)

            if len(true_data_object) > self.mem_size:
                raise ValueError("Data Size must be less than buffer size")

        self._do_writes(true_data_object, index)

    def _do_reads(self, var_type: dtype, shape: List[int], index: int = None) -> bytes:
        assert index is None or (isinstance(index, int) and index >= 0), "Index must be None or a non-negative integer!"

        indicies = [index] if index is not None else range(self.context.queue_count)
        completed_stages = [0] * len(indicies)
        bytes_list: List[bytes] = [None] * len(indicies)

        mem_size = int(npc.prod(shape)) * var_type.item_size

        while not all(stage == 2 for stage in completed_stages):
            for i in range(len(indicies)):
                if completed_stages[i] == 2:
                    continue

                queue_index = indicies[i]

                if completed_stages[i] == 0:
                    if self.signals[queue_index].try_wait(False, queue_index):
                        completed_stages[i] = 1
                        native.buffer_read(self._handle, 0, mem_size, queue_index)
                        check_for_errors()
                    else:
                        continue

                if completed_stages[i] == 1:
                    if self.signals[queue_index].try_wait(True, queue_index):
                        completed_stages[i] = 2
                    else:
                        continue

                bytes_list[i] = native.buffer_read_staging(self._handle, queue_index, mem_size)
                check_for_errors()
        
        host_arrays = []

        for b in bytes_list:
            host_arrays.append(
                npc.from_buffer(b, dtype=to_numpy_dtype(var_type), shape=tuple(shape))
            )

        return host_arrays if index is None else host_arrays[0]

    def read(self, index: Union[int, None] = None):
        """
        Downloads data from the GPU buffer to the host.

        :param index: The device index to read from. If ``None``, reads from all devices 
                      and returns a stacked array with an extra dimension for the device index.
        :type index: Union[int, None]
        :return: A host array representation containing the buffer data.
        :raises ValueError: If the specified index is invalid.
        """

        true_scalar = self.var_type.scalar

        if true_scalar is None:
            true_scalar = self.var_type

        data_shape = list(self.shape) + list(self.var_type.true_numpy_shape)

        if index is not None:
            return self._do_reads(true_scalar, data_shape, index)
        
        results = self._do_reads(true_scalar, data_shape, None)

        if npc.HAS_NUMPY:
            return npc.numpy_module().array(results)

        return results

def asbuffer(array: typing.Any) -> Buffer:
    """Cast an array-like object to a buffer object."""

    if not npc.is_array_like(array):
        raise TypeError("Expected an array-like object")

    buffer = Buffer(npc.array_shape(array), from_numpy_dtype(npc.array_dtype(array)))
    buffer.write(array)

    return buffer


def from_cuda_array(
    obj: typing.Any,
    var_type: typing.Optional[dtype] = None,
    require_contiguous: bool = True,
    writable: typing.Optional[bool] = None,
    keepalive: bool = True,
) -> Buffer:
    from .init import get_backend
    from .backend import BACKEND_PYCUDA

    if get_backend() != BACKEND_PYCUDA:
        raise RuntimeError("from_cuda_array() is currently only supported with backend='pycuda'.")

    if not hasattr(obj, "__cuda_array_interface__"):
        raise TypeError("Expected an object with __cuda_array_interface__")

    npc.require_numpy("from_cuda_array")
    np = npc.numpy_module()

    iface = obj.__cuda_array_interface__
    if not isinstance(iface, dict):
        raise TypeError("__cuda_array_interface__ must be a dictionary")

    if "shape" not in iface or "typestr" not in iface or "data" not in iface:
        raise ValueError("__cuda_array_interface__ is missing required fields (shape/typestr/data)")

    shape = tuple(int(dim) for dim in iface["shape"])
    if len(shape) == 0:
        shape = (1,)

    data_entry = iface["data"]
    if not (isinstance(data_entry, tuple) and len(data_entry) >= 2):
        raise ValueError("__cuda_array_interface__['data'] must be a tuple (ptr, read_only)")

    ptr = int(data_entry[0])
    source_read_only = bool(data_entry[1])

    inferred_np_dtype = np.dtype(iface["typestr"])
    inferred_var_type = from_numpy_dtype(inferred_np_dtype)
    if var_type is None:
        var_type = inferred_var_type

    if not (var_type == inferred_var_type):
        raise ValueError(
            f"CAI dtype ({inferred_np_dtype}) does not match requested vd dtype ({var_type.name})."
        )

    if require_contiguous:
        strides = iface.get("strides")
        if strides is not None:
            expected_strides = []
            stride = int(inferred_np_dtype.itemsize)
            for dim in reversed(shape):
                expected_strides.insert(0, stride)
                stride *= int(dim)
            if tuple(int(x) for x in strides) != tuple(expected_strides):
                raise ValueError("Only contiguous C-order CUDA arrays are supported in from_cuda_array().")

    buffer_writable = (not source_read_only) if writable is None else bool(writable)
    if buffer_writable and source_read_only:
        raise ValueError("Requested writable=True for a read-only CUDA array.")
    
    external_buffer_info = ExternalBufferInfo(
        writable=buffer_writable,
        iface=iface,
        keepalive=keepalive,
        cuda_ptr=ptr
    )

    return Buffer(shape, var_type, external_buffer=external_buffer_info)

class RFFTBuffer(Buffer):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__(tuple(shape[:-1]) + (shape[-1] // 2 + 1,), complex64)

        self.real_shape = shape
        self.fourier_shape = self.shape
    
    def read_real(self, index: Union[int, None] = None):
        npc.require_numpy("RFFTBuffer.read_real")
        np = npc.numpy_module()
        return self.read(index).view(np.float32)[..., :self.real_shape[-1]]

    def read_fourier(self, index: Union[int, None] = None):
        return self.read(index)
    
    def write_real(self, data, index: int = None):
        npc.require_numpy("RFFTBuffer.write_real")
        np = npc.numpy_module()
        assert data.shape == self.real_shape, "Data shape must match real shape!"
        assert not np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be scalar!"

        true_data = np.zeros(self.shape[:-1] + (self.shape[-1] * 2,), dtype=np.float32)
        true_data[..., :self.real_shape[-1]] = data

        self.write(np.ascontiguousarray(true_data).view(np.complex64), index)

    def write_fourier(self, data, index: int = None):
        npc.require_numpy("RFFTBuffer.write_fourier")
        np = npc.numpy_module()
        assert data.shape == self.fourier_shape, f"Data shape {data.shape} must match fourier shape {self.fourier_shape}!"
        assert np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be complex!"

        self.write(np.ascontiguousarray(data.astype(np.complex64)).view(np.float32), index)

def asrfftbuffer(data) -> RFFTBuffer:
    npc.require_numpy("asrfftbuffer")
    np = npc.numpy_module()
    assert not np.issubdtype(data.dtype, np.complexfloating), "Data dtype must be scalar!"

    buffer = RFFTBuffer(data.shape)
    buffer.write_real(data)

    return buffer
