from typing import Tuple

import numpy as np

import vkdispatch as vd
import vkdispatch_native
from vkdispatch.dtype import dtype


class Buffer:
    """TODO: Docstring"""

    _handle: int
    var_type: dtype
    shape: Tuple[int]
    size: int
    mem_size: int

    def __init__(self, shape: Tuple[int], var_type: dtype, per_device: bool = False) -> None:
        self.var_type: dtype = var_type
        self.shape: Tuple[int] = shape
        self.size: int = int(np.prod(shape))
        self.mem_size: int = self.size * self.var_type.item_size
        self.per_device: bool = per_device
        self.ctx = vd.get_context()

        self._handle: int = vkdispatch_native.buffer_create(
            vd.get_context_handle(), self.mem_size, 1 if self.per_device else 0
        )
        vd.check_for_errors()

    def __del__(self) -> None:
        pass  # vkdispatch_native.buffer_destroy(self._handle)

    def write(self, data: np.ndarray, index: int = -1) -> None:
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
        
        if self.per_device and index >= len(self.ctx.devices):
            raise ValueError(f"Invalid device index {index}!")
        elif not self.per_device and index >= self.ctx.stream_count:
            raise ValueError(f"Invalid stream index {index}!")


        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")

        vkdispatch_native.buffer_write(
            self._handle, np.ascontiguousarray(data), 0, self.mem_size, index
        )
        vd.check_for_errors()

    def read(self, index: int = 0) -> np.ndarray:
        """Read the data in the buffer at the specified device index and return it as a
        numpy array.

        Parameters:
        index (int): The index to read the data from. Default is 0.

        Returns:
        (np.ndarray): The data in the buffer as a numpy array.
        """
        if index < 0:
            raise ValueError(f"Invalid buffer index {index}!")
        
        if self.per_device and index >= len(self.ctx.devices):
            raise ValueError(f"Invalid device index {index}!")
        elif not self.per_device and index >= self.ctx.stream_count:
            raise ValueError(f"Invalid stream index {index}!")

        result = np.ndarray(
            shape=(self.shape + self.var_type._true_numpy_shape),
            dtype=vd.to_numpy_dtype(self.var_type.scalar),
        )
        vkdispatch_native.buffer_read(
            self._handle, result, 0, self.mem_size, index
        )
        vd.check_for_errors()

        return result


# TODO: Move this to a class method of Buffer
def asbuffer(array: np.ndarray) -> Buffer:
    """Cast a numpy array to a buffer object."""

    buffer = Buffer(array.shape, vd.from_numpy_dtype(array.dtype))
    buffer.write(array)

    return buffer
