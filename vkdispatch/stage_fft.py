import vkdispatch
import vkdispatch_native

import numpy as np

class fft_plan:
    def __init__(self, shape: tuple[int, ...]):
        assert len(shape) > 0 and len(shape) < 4, "shape must be 1D, 2D, or 3D"

        self.shape = shape
        self.mem_size = np.prod(shape) * np.dtype(np.complex64).itemsize # currently only support complex64

        self._handle = vkdispatch_native.stage_fft_plan_create(vkdispatch.get_context_handle(), list(self.shape), self.mem_size)
    
    def record(self, command_list: vkdispatch.command_list, buffer: vkdispatch.buffer, inverse: bool = False):
        assert buffer.dtype == np.complex64, "buffer must be of dtype complex64"
        assert buffer.mem_size == self.mem_size, "buffer size must match plan size"

        vkdispatch_native.stage_fft_record(command_list._handle, self._handle, buffer._handle, 1 if inverse else -1)

    def record_forward(self, command_list: vkdispatch.command_list, buffer: vkdispatch.buffer):
        self.record(command_list, buffer, False)
    
    def record_inverse(self, command_list: vkdispatch.command_list, buffer: vkdispatch.buffer):
        self.record(command_list, buffer, True)