# distutils: language=c++
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "errors.h":
    const char* get_error_string_extern()

cpdef inline get_error_string():
    cdef const char* error_string = get_error_string_extern()
    if error_string is NULL:
        return 0
    else:
        return error_string.decode('utf-8')

#cpdef inline buffer_create(unsigned long long context, unsigned long long size):
#    return <unsigned long long>buffer_create_extern(<Context*>context, size)

#cpdef inline buffer_destroy(unsigned long long buffer):
#    buffer_destroy_extern(<Buffer*>buffer)

#cpdef inline buffer_write(unsigned long long buffer, cnp.ndarray data, unsigned long long offset, unsigned long long size, int device_index):
#    buffer_write_extern(<Buffer*>buffer, <void*>data.data, offset, size, device_index)

#cpdef inline buffer_read(unsigned long long buffer, cnp.ndarray data, unsigned long long offset, unsigned long long size, int device_index):
#    buffer_read_extern(<Buffer*>buffer, <void*>data.data, offset, size, device_index)

#cpdef inline buffer_copy(unsigned long long src, unsigned long long dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index):
#    buffer_copy_extern(<Buffer*>src, <Buffer*>dst, src_offset, dst_offset, size, device_index)