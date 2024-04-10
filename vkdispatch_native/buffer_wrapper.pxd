# distutils: language=c++
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "buffer.h":
    struct DeviceContext
    struct Buffer

    Buffer* create_buffer_extern(DeviceContext* device_context, unsigned long long size)
    void destroy_buffer_extern(Buffer* buffer)

    void buffer_write_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)
    void buffer_read_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)

    void buffer_copy_extern(Buffer* src, Buffer* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index)

cpdef inline create_buffer(unsigned long long device_context, unsigned long long size):
    return <unsigned long long>create_buffer_extern(<DeviceContext*>device_context, size)

cpdef inline destroy_buffer(unsigned long long buffer):
    destroy_buffer_extern(<Buffer*>buffer)

cpdef inline write_buffer(unsigned long long buffer, cnp.ndarray data, unsigned long long offset, unsigned long long size, int device_index):
    buffer_write_extern(<Buffer*>buffer, <void*>data.data, offset, size, device_index)

cpdef inline read_buffer(unsigned long long buffer, cnp.ndarray data, unsigned long long offset, unsigned long long size, int device_index):
    buffer_read_extern(<Buffer*>buffer, <void*>data.data, offset, size, device_index)

cpdef inline copy_buffer(unsigned long long src, unsigned long long dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index):
    buffer_copy_extern(<Buffer*>src, <Buffer*>dst, src_offset, dst_offset, size, device_index)