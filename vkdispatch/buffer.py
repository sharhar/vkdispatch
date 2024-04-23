import vkdispatch
import vkdispatch_native

import numpy as np

class buffer:
    def __init__(self, shape: tuple[int], dtype: np.dtype) -> None:
        self.dtype: np.dtype = np.dtype(dtype)
        self.shape: tuple[int] = shape
        self.size: int = np.prod(shape)
        self.mem_size: int = self.size * self.dtype.itemsize
        self._handle: int = vkdispatch_native.buffer_create(vkdispatch.get_context_handle(), self.mem_size)
    
    def __del__(self) -> None:
        pass #vkdispatch_native.buffer_destroy(self._handle)
    
    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")
        vkdispatch_native.buffer_write(self._handle, np.ascontiguousarray(data), 0, self.mem_size, device_index)
    
    def read(self, device_index: int = -1) -> np.ndarray:
        result = np.ndarray(shape=self.shape, dtype=self.dtype)
        vkdispatch_native.buffer_read(self._handle, result, 0, self.mem_size, device_index)
        return result

def asbuffer(array: np.ndarray) -> buffer:
    buffer = vkdispatch.buffer(array.shape, array.dtype)
    buffer.write(array)
    return buffer