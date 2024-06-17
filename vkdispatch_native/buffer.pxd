# distutils: language=c++
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "buffer.h":
    struct Context
    struct Buffer

    Buffer* buffer_create_extern(Context* context, unsigned long long size, int per_device)
    void buffer_destroy_extern(Buffer* buffer)

    void buffer_write_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)
    void buffer_read_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)

    #void buffer_copy_extern(Buffer* src, Buffer* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index)

cpdef inline buffer_create(unsigned long long context, unsigned long long size, int per_device):
    return <unsigned long long>buffer_create_extern(<Context*>context, size, per_device)

cpdef inline buffer_destroy(unsigned long long buffer):
    buffer_destroy_extern(<Buffer*>buffer)

cpdef inline buffer_write(unsigned long long buffer, bytes data, unsigned long long offset, unsigned long long size, int device_index):
    cdef const char* data_view = data
    buffer_write_extern(<Buffer*>buffer, <void*>data_view, offset, size, device_index)

cpdef inline buffer_read(unsigned long long buffer, cnp.ndarray data, unsigned long long offset, unsigned long long size, int device_index):
    buffer_read_extern(<Buffer*>buffer, <void*>data.data, offset, size, device_index)

#cpdef inline buffer_copy(unsigned long long src, unsigned long long dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index):
#    buffer_copy_extern(<Buffer*>src, <Buffer*>dst, src_offset, dst_offset, size, device_index)