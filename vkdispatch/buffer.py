import vkdispatch as vd
import vkdispatch_native

import numpy as np

class buffer:
    def __init__(self, shape: tuple[int], var_type: 'vd.shader_type') -> None:
        self.var_type = var_type
        self.shape: tuple[int] = shape
        self.size: int = np.prod(shape)
        self.mem_size: int = self.size * self.var_type.item_size
        self._handle: int = vkdispatch_native.buffer_create(vd.get_context_handle(), self.mem_size)
    
    def __del__(self) -> None:
        pass #vkdispatch_native.buffer_destroy(self._handle)
    
    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")
        vkdispatch_native.buffer_write(self._handle, np.ascontiguousarray(data), 0, self.mem_size, device_index)
    
    def read(self, device_index: int = -1) -> np.ndarray:
        result = np.ndarray(shape=self.shape, dtype=vd.to_numpy_dtype(self.var_type))
        vkdispatch_native.buffer_read(self._handle, result, 0, self.mem_size, device_index)
        return result

def asbuffer(array: np.ndarray) -> buffer:
    buffer = vd.buffer(array.shape, vd.from_numpy_dtype(array.dtype))
    buffer.write(array)
    return buffer