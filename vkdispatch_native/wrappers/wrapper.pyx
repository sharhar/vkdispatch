# distutils: language=c++
from libcpp cimport bool
import sys

cimport init
cimport context
cimport buffer
cimport image
cimport command_list
cimport stage_transfer
cimport stage_fft
cimport stage_compute
cimport descriptor_set
cimport errors
cimport conditional

# print_callback.pyx
cimport cython
from libc.string cimport strlen

# We need to declare the Python C API functions we'll use
cdef extern from "Python.h":
    ctypedef struct PyObject
    void Py_INCREF(object)
    void Py_DECREF(object)
    ctypedef enum PyGILState_STATE:
        PyGILState_LOCKED
        PyGILState_UNLOCKED
    PyGILState_STATE PyGILState_Ensure()
    void PyGILState_Release(PyGILState_STATE)

# Global reference to print function
cdef object _print_func = None

cdef extern from "../include/base.hh":
    void init_print_system()
    void cleanup_print_system()
    void thread_safe_print(const char* message)

cdef void _impl_init_print_system() noexcept nogil:
    global _print_func
    with gil:
        _print_func = __builtins__.print

cdef void _impl_cleanup_print_system() noexcept nogil:
    global _print_func
    with gil:
        _print_func = None

cdef void _impl_thread_safe_print(const char* message) noexcept nogil:
    cdef:
        PyGILState_STATE gil_state
        bytes py_bytes
        str py_str
    
    gil_state = PyGILState_Ensure()
    try:
        py_bytes = message[:strlen(message)]
        py_str = py_bytes.decode('utf-8')
        _print_func(py_str, end='')
    finally:
        PyGILState_Release(gil_state)

# Export the C interface functions with exactly matching signatures
cdef extern void init_print_system():
    _impl_init_print_system()

cdef extern void cleanup_print_system():
    _impl_cleanup_print_system()

cdef extern void thread_safe_print(const char* message):
    _impl_thread_safe_print(message)

#cdef extern void cython_print_callback(const char* string) nogil:
#    with gil:
#        print(bytes(string).decode('utf-8'), end='')


#cdef extern void cython_acquire_gil_and_log(const char* message) nogil:
#    cdef PyGILState_STATE gil_state
#    gil_state = PyGILState_Ensure()
#    try:
#        cython_print_callback(message)
#    finally:
#        PyGILState_Release(gil_state)