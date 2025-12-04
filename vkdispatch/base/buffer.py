from typing import Tuple
from typing import Dict
from typing import Union

import numpy as np

from .dtype import dtype
from .context import Handle
from .errors import check_for_errors

from .dtype import to_numpy_dtype, from_numpy_dtype, complex64

import vkdispatch_native

import typing

_ArgType = typing.TypeVar('_ArgType', bound=dtype)

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

    def __init__(self, shape: Tuple[int, ...], var_type: dtype) -> None:
        super().__init__()

        if len(shape) > 3:
            raise ValueError("Buffer shape must be 1, 2, or 3 dimensions!")

        self.var_type: dtype = var_type
        self.shape: Tuple[int] = shape
        self.size: int = int(np.prod(shape))
        self.mem_size: int = self.size * self.var_type.item_size

        if self.size > 2 ** 30:
            raise ValueError("Cannot allocate buffers larger than 2^30 elements!")

        shader_shape_internal = [1, 1, 1, 1]

        for ii, axis_size in enumerate(shape):
            shader_shape_internal[ii] = axis_size
        
        shader_shape_internal[3] = shader_shape_internal[0] * shader_shape_internal[1] * shader_shape_internal[2]

        self.shader_shape = tuple(shader_shape_internal)

        handle = vkdispatch_native.buffer_create(
            self.context._handle, self.mem_size, 0
        )
        check_for_errors()

        self.register_handle(handle)

    def _destroy(self) -> None:
        """Destroy the buffer and all child handles."""
        vkdispatch_native.buffer_destroy(self._handle)

    def __del__(self) -> None:
        self.destroy()

    def write(self, data: Union[bytes, np.ndarray], index: int = -1) -> None:
        """
        Uploads data from the host to the GPU buffer.

        If ``index`` is -1, the data is broadcast to the memory of all active devices 
        in the context. Otherwise, it writes only to the device specified by the index.

        :param data: The source data. Can be a raw ``bytes`` object or a ``numpy.ndarray``.
                     If a numpy array is provided, its size and dtype must match the buffer's capacity.
        :type data: Union[bytes, np.ndarray]
        :param index: The device index to write to. Defaults to -1 (all devices).
        :type index: int
        :raises ValueError: If the data size exceeds the buffer size or if the index is invalid.
        """
        if index < -1:
            raise ValueError(f"Invalid buffer index {index}!")

        true_data_object = None

        if isinstance(data, np.ndarray):
            if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
                raise ValueError("Numpy buffer sizes must match!")

            true_data_object = np.ascontiguousarray(data).tobytes()
        else:
            if len(data) > self.mem_size:
                raise ValueError("Data Size must be less than buffer size")

            true_data_object = data

        vkdispatch_native.buffer_write(
            self._handle, true_data_object, 0, len(true_data_object), index
        )
        check_for_errors()

    def read(self, index: Union[int, None] = None) -> np.ndarray:
        """
        Downloads data from the GPU buffer to the host.

        :param index: The device index to read from. If ``None``, reads from all devices 
                      and returns a stacked array with an extra dimension for the device index.
        :type index: Union[int, None]
        :return: A numpy array containing the buffer data.
        :rtype: np.ndarray
        :raises ValueError: If the specified index is invalid.
        """

        true_scalar = self.var_type.scalar

        if true_scalar is None:
            true_scalar = self.var_type

        if index is not None:
            if index < 0:
                raise ValueError(f"Invalid buffer index {index}!")
            
            result_bytes = vkdispatch_native.buffer_read(
                self._handle, 0, self.mem_size, index
            )

            result = np.frombuffer(result_bytes, dtype=to_numpy_dtype(true_scalar)).reshape(self.shape + self.var_type.true_numpy_shape)

            check_for_errors()
        else:
            result = np.zeros((self.context.queue_count,) + self.shape + self.var_type.true_numpy_shape, dtype=to_numpy_dtype(true_scalar))

            for i in range(self.context.queue_count):
                result[i] = self.read(i)

        return result


def asbuffer(array: np.ndarray) -> Buffer:
    """Cast a numpy array to a buffer object."""

    buffer = Buffer(array.shape, from_numpy_dtype(array.dtype))
    buffer.write(array)

    return buffer


class RFFTBuffer(Buffer):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__(tuple(shape[:-1]) + (shape[-1] // 2 + 1,), complex64)

        self.real_shape = shape
        self.fourier_shape = self.shape
    
    def read_real(self, index: Union[int, None] = None) -> np.ndarray:
        return self.read(index).view(np.float32)[..., :self.real_shape[-1]]

    def read_fourier(self, index: Union[int, None] = None) -> np.ndarray:
        return self.read(index)
    
    def write_real(self, data: np.ndarray, index: int = -1):
        assert data.shape == self.real_shape, "Data shape must match real shape!"
        assert not np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be scalar!"

        true_data = np.zeros(self.shape[:-1] + (self.shape[-1] * 2,), dtype=np.float32)
        true_data[..., :self.real_shape[-1]] = data

        self.write(np.ascontiguousarray(true_data).view(np.complex64), index)

    def write_fourier(self, data: np.ndarray, index: int = -1):
        assert data.shape == self.fourier_shape, f"Data shape {data.shape} must match fourier shape {self.fourier_shape}!"
        assert np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be complex!"

        self.write(np.ascontiguousarray(data.astype(np.complex64)).view(np.float32), index)

def asrfftbuffer(data: np.ndarray) -> RFFTBuffer:
    assert not np.issubdtype(data.dtype, np.complexfloating), "Data dtype must be scalar!"

    buffer = RFFTBuffer(data.shape)
    buffer.write_real(data)

    return buffer