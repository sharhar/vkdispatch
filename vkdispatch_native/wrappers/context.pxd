# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/context.hh":
    struct Context

    Context* context_create_extern(int* device_indicies, int* queue_counts, int* queue_families, int device_count)
    void context_destroy_extern(Context* device_context);

cpdef inline context_create(list[int] device_indicies, list[list[int]] queue_families):
    assert len(device_indicies) == len(queue_families)

    cdef int len_device_indicies = len(device_indicies)
    cdef int* device_indicies_c = <int*>malloc(len_device_indicies * sizeof(int))
    cdef int* queue_counts_c    = <int*>malloc(len_device_indicies * sizeof(int))

    cdef int total_queue_count = 0

    for i in range(len_device_indicies):
        device_indicies_c[i] = device_indicies[i]
        queue_counts_c[i] = len(queue_families[i]) #submission_thread_counts[i]
        total_queue_count += queue_counts_c[i]

    cdef int* queue_families_c = <int*>malloc(total_queue_count * sizeof(int))

    cdef int current_index = 0
    
    for i in range(len_device_indicies):
        for j in range(queue_counts_c[i]):
            queue_families_c[current_index] = queue_families[i][j]
            current_index += 1
    
    assert current_index == total_queue_count

    cdef unsigned long long result = <unsigned long long>context_create_extern(device_indicies_c, queue_counts_c, queue_families_c, len_device_indicies)

    free(device_indicies_c)
    free(queue_counts_c)
    free(queue_families_c)

    return result

cpdef inline context_destroy(unsigned long long context):
    context_destroy_extern(<Context*>context)