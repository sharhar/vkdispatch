# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "device_context.h":
    struct DeviceContext

    DeviceContext* create_device_context_extern(int* device_indicies, int* submission_thread_couts, int device_count)
    void destroy_device_context_extern(DeviceContext* device_context);

cpdef inline create_device_context(list[int] device_indicies, list[int] submission_thread_counts):
    assert len(device_indicies) == len(submission_thread_counts)

    cdef int len_device_indicies = len(device_indicies)
    cdef int* device_indicies_c = <int*>malloc(len_device_indicies * sizeof(int))
    cdef int* submission_thread_counts_c = <int*>malloc(len_device_indicies * sizeof(int))

    for i in range(len_device_indicies):
        device_indicies_c[i] = device_indicies[i]
        submission_thread_counts_c[i] = submission_thread_counts[i]

    cdef unsigned long long result = <unsigned long long>create_device_context_extern(device_indicies_c, submission_thread_counts_c, len_device_indicies)

    free(device_indicies_c)
    free(submission_thread_counts_c)

    return result

cpdef inline destroy_device_context(unsigned long long device_context):
    destroy_device_context_extern(<DeviceContext*>device_context)