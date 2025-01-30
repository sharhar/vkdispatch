# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/buffer.hh":
    struct Context
    struct Buffer

    Buffer* buffer_create_extern(Context* context, unsigned long long size, int per_device)
    void buffer_destroy_extern(Buffer* buffer)

    void buffer_write_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)
    void buffer_read_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index)

cpdef inline buffer_create(unsigned long long context, unsigned long long size, int per_device):
    return <unsigned long long>buffer_create_extern(<Context*>context, size, per_device)

cpdef inline buffer_destroy(unsigned long long buffer):
    buffer_destroy_extern(<Buffer*>buffer)

cpdef inline buffer_write(unsigned long long buffer, bytes data, unsigned long long offset, unsigned long long size, int device_index):
    cdef const char* data_view = data
    buffer_write_extern(<Buffer*>buffer, <void*>data_view, offset, size, device_index)

cpdef inline buffer_read(unsigned long long buffer, unsigned long long offset, unsigned long long size, int device_index):
    cdef bytes data = bytes(size)
    cdef char* data_view = data

    buffer_read_extern(<Buffer*>buffer, <void*>data_view, offset, size, device_index)

    return data
