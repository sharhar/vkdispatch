# distutils: language=c++
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "command_list.h":
    struct Context
    struct CommandList

    CommandList* command_list_create_extern(Context* context)
    void command_list_destroy_extern(CommandList* command_list)
    void command_list_get_instance_size_extern(CommandList* command_list, unsigned long long* instance_size)
    void command_list_reset_extern(CommandList* command_list)
    void command_list_submit_extern(CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int* indicies, int count, int per_device, void* signal)

cpdef inline command_list_create(unsigned long long context):
    return <unsigned long long>command_list_create_extern(<Context*>context)

cpdef inline command_list_destroy(unsigned long long command_list):
    command_list_destroy_extern(<CommandList*>command_list)

cpdef inline command_list_get_instance_size(unsigned long long command_list):
    cdef unsigned long long instance_size
    command_list_get_instance_size_extern(<CommandList*>command_list, &instance_size)
    return instance_size

cpdef inline command_list_reset(unsigned long long command_list):
    command_list_reset_extern(<CommandList*>command_list)

cpdef inline command_list_submit(unsigned long long command_list, bytes data, unsigned int instance_count, unsigned int instances_per_batch, list[int] indicies, int per_device):
    assert instance_count % instances_per_batch == 0, "instance_count must be a multiple of instances_per_batch"

    cdef int len_indicies = len(indicies)
    cdef int* indicies_c = <int*>malloc(len_indicies * sizeof(int))

    for i in range(len_indicies):
        indicies_c[i] = indicies[i]

    cdef const char* data_view = data

    command_list_submit_extern(<CommandList*>command_list, <void*>data_view, instance_count, indicies_c, len_indicies, per_device, <void*>0)
