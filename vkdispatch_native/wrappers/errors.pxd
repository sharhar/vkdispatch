# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/errors.hh":
    const char* get_error_string_extern()

cpdef inline get_error_string():
    cdef const char* error_string = get_error_string_extern()
    if error_string is NULL:
        return 0
    else:
        return error_string.decode('utf-8')
