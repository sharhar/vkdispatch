import vkdispatch
import vkdispatch_native

import numpy as np

"""


cpdef inline stage_fft_plan_create(unsigned long long context, list[int] dims, unsigned long long buffer_size):
    assert len(dims) > 0 and len(dims) < 4, "dims must be a list of length 1, 2, or 3"

    cdef Context* ctx = <Context*>context
    cdef unsigned long long dims_ = len(dims)

    cdef unsigned long long dims__[3] = [0, 0, 0]

    for i in range(dims_):
        dims__[i] = dims[i]    
    
    cdef FFTPlan* plan = stage_fft_plan_create_extern(ctx, dims_, dims__[0], dims__[1], dims__[2], buffer_size)

    return <unsigned long long>plan

cpdef inline stage_fft_record(unsigned long long command_list, unsigned long long plan, unsigned long long buffer, int inverse):
    cdef CommandList* cl = <CommandList*>command_list
    cdef FFTPlan* p = <FFTPlan*>plan
    cdef Buffer* b = <Buffer*>buffer

    stage_fft_record_extern(cl, p, b, inverse)

"""

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