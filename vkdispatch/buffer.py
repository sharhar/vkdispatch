import vkdispatch
import vkdispatch_native

import numpy as np

class Buffer:
    def __init__(self, ctx: vkdispatch.DeviceContext, shape: tuple[int], dtype: np.dtype) -> None:
        self.dtype: np.dtype = np.dtype(dtype)
        self.shape: tuple[int] = shape
        self.size: int = np.prod(shape)
        self.mem_size: int = self.size * self.dtype.itemsize
        self._handle: int = vkdispatch_native.create_buffer(ctx._handle, self.mem_size)
        self.ctx: vkdispatch.DeviceContext = ctx
    
    def __del__(self) -> None:
        pass #vkdispatch_native.destroy_buffer(self._handle)
    
    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")
        vkdispatch_native.write_buffer(self._handle, np.ascontiguousarray(data), 0, self.mem_size, device_index)
    
    def read(self, device_index: int = -1) -> np.ndarray:
        result = np.ndarray(shape=self.shape, dtype=self.dtype)
        vkdispatch_native.read_buffer(self._handle, result, 0, self.mem_size, device_index)
        return result
    
    def copy_to(self, other: 'Buffer', device_index: int = -1) -> None:
        if other.mem_size != self.mem_size:
            raise ValueError("Buffer memory sizes must match!")
        vkdispatch_native.copy_buffer(self._handle, other._handle, 0, 0, self.mem_size, device_index)