# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/stage_fft.hh":
    struct Context
    struct Buffer
    struct CommandList
    struct FFTPlan


    FFTPlan* stage_fft_plan_create_extern(Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size, unsigned int do_r2c)
    void stage_fft_record_extern(CommandList* command_list, FFTPlan* plan, Buffer* buffer, int inverse)

cpdef inline stage_fft_plan_create(unsigned long long context, list[int] dims, unsigned long long buffer_size, unsigned int do_r2c):
    assert len(dims) > 0 and len(dims) < 4, "dims must be a list of length 1, 2, or 3"

    cdef Context* ctx = <Context*>context
    cdef unsigned long long dims_ = len(dims)

    cdef unsigned long long* dims__ = <unsigned long long*>malloc(3 * sizeof(unsigned long long))

    for i in range(3):
        dims__[i] = 0

    for i in range(dims_):
        dims__[i] = dims[i]    
    
    cdef FFTPlan* plan = stage_fft_plan_create_extern(ctx, dims_, dims__[0], dims__[1], dims__[2], buffer_size, do_r2c)

    free(dims__)

    return <unsigned long long>plan

cpdef inline stage_fft_record(unsigned long long command_list, unsigned long long plan, unsigned long long buffer, int inverse):
    cdef CommandList* cl = <CommandList*>command_list
    cdef FFTPlan* p = <FFTPlan*>plan
    cdef Buffer* b = <Buffer*>buffer

    stage_fft_record_extern(cl, p, b, inverse)