from typing import Tuple
from typing import List
from typing import Union

import numpy as np

from .dtype import dtype
from .context import get_context, get_context_handle
from .errors import check_for_errors

from .dtype import to_numpy_dtype, from_numpy_dtype

import vkdispatch_native

class Buffer:
    """TODO: Docstring"""

    _handle: int
    var_type: dtype
    shape: Tuple[int]
    size: int
    mem_size: int

    def __init__(self, shape: Tuple[int, ...], var_type: dtype) -> None:
        if len(shape) > 3:
            raise ValueError("Buffer shape must be 1, 2, or 3 dimensions!")

        self.var_type: dtype = var_type
        self.shape: Tuple[int] = shape
        self.size: int = int(np.prod(shape))
        self.mem_size: int = self.size * self.var_type.item_size
        self.ctx = get_context()

        if self.size > 2 ** 30:
            raise ValueError("Cannot allocate buffers larger than 2^30 elements!")

        shader_shape_internal = [1, 1, 1, 1]

        for ii, axis_size in enumerate(shape):
            shader_shape_internal[ii] = axis_size
        
        shader_shape_internal[3] = shader_shape_internal[0] * shader_shape_internal[1] * shader_shape_internal[2]

        self.shader_shape = tuple(shader_shape_internal)

        self._handle: int = vkdispatch_native.buffer_create(
            get_context_handle(), self.mem_size, 0
        )
        check_for_errors()

    def __del__(self) -> None:
        pass  # vkdispatch_native.buffer_destroy(self._handle)

    def write(self, data: Union[bytes, np.ndarray], index: int = -1) -> None:
        """Given data in some numpy array, write that data to the buffer at the
        specified index. The default index of -1 will write to
        all buffers.

        Parameters:
        data (np.ndarray): The data to write to the buffer.
        index (int): The  index to write the data to. Default is -1 and
            will write to all buffers.

        Returns:
        None
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

    def read(self, index: Union[int, None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Read the data in the buffer at the specified device index and return it as a
        numpy array.

        Parameters:
        index (int): The index to read the data from. Default is 0.

        Returns:
        (np.ndarray): The data in the buffer as a numpy array.
        """

        if index is not None:
            if index < 0:
                raise ValueError(f"Invalid buffer index {index}!")
            
            true_scalar = self.var_type.scalar

            if true_scalar is None:
                true_scalar = self.var_type
            
            result_bytes = vkdispatch_native.buffer_read(
                self._handle, 0, self.mem_size, index
            )

            result = np.frombuffer(result_bytes, dtype=to_numpy_dtype(true_scalar)).reshape(self.shape + self.var_type.true_numpy_shape)

            check_for_errors()
        else:
            result = []

            for i in range(self.ctx.stream_count):
                result.append(self.read(i))

        return result

    @classmethod
    def __class_getitem__(cls, params):
       raise RuntimeError("Cannot index into vd.Buffer! Perhaps you meant to use vc.Buffer?")


def asbuffer(array: np.ndarray) -> Buffer:
    """Cast a numpy array to a buffer object."""

    buffer = Buffer(array.shape, from_numpy_dtype(array.dtype))
    buffer.write(array)

    return buffer
