# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "buffer.h":
    struct DeviceContext
    struct Buffer

    Buffer* create_buffer_extern(DeviceContext* device_context, unsigned long long size)
    void destroy_buffer_extern(Buffer* buffer)

cpdef inline create_buffer(unsigned long long device_context, unsigned long long size):
    return <unsigned long long>create_buffer_extern(<DeviceContext*>device_context, size)

cpdef inline destroy_buffer(unsigned long long buffer):
    destroy_buffer_extern(<Buffer*>buffer)