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

    def __init__(self, shape: Tuple[int], var_type: dtype) -> None:
        self.var_type: dtype = var_type
        self.shape: Tuple[int] = shape
        self.size: int = np.prod(shape)
        self.mem_size: int = self.size * self.var_type.item_size

        # print(f"Buffer var_type: {var_type}")
        # print(f"Buffer shape: {shape}")
        # print(f"Buffer size: {self.size}")
        # print(f"Buffer mem_size: {self.mem_size}")
        print(vd.get_context_handle())

        self._handle: int = vkdispatch_native.buffer_create(
            vd.get_context_handle(), self.mem_size
        )

    def __del__(self) -> None:
        pass  # vkdispatch_native.buffer_destroy(self._handle)

    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        """Given data in some numpy array, write that data to the buffer at device
        specified by device_index. The default device index of -1 will write to the
        all devices.

        Parameters:
        data (np.ndarray): The data to write to the buffer.
        device_index (int): The device index to write the data to. Default is -1 and
            will write to all devices.

        Returns:
        None
        """
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")

        vkdispatch_native.buffer_write(
            self._handle, np.ascontiguousarray(data), 0, self.mem_size, device_index
        )

    def read(self, device_index: int = -1) -> np.ndarray:
        """Read the data in the buffer at the specified device index and return it as a
        numpy array.

        Parameters:
        device_index (int): The device index to read the data from. Default is -1 and
            will read from all devices.

        Returns:
        (np.ndarray): The data in the buffer as a numpy array.
        """
        result = np.ndarray(
            shape=(self.shape + self.var_type._true_numpy_shape),
            dtype=vd.to_numpy_dtype(self.var_type.scalar),
        )
        vkdispatch_native.buffer_read(
            self._handle, result, 0, self.mem_size, device_index
        )

        return result


# TODO: Move this to a class method of Buffer
def asbuffer(array: np.ndarray) -> Buffer:
    """Cast a numpy array to a buffer object."""

    print("asbuffer", array.shape, vd.from_numpy_dtype(array.dtype))

    buffer = Buffer(array.shape, vd.from_numpy_dtype(array.dtype))
    buffer.write(array)

    return buffer
