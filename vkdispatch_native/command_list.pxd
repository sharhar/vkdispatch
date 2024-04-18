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
    void command_list_submit_extern(CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int* devices, int deviceCount, int* submission_thread_counts)

cpdef inline command_list_create(unsigned long long context):
    return <unsigned long long>command_list_create_extern(<Context*>context)

cpdef inline command_list_destroy(unsigned long long command_list):
    command_list_destroy_extern(<CommandList*>command_list)

cpdef inline command_list_get_instance_size(unsigned long long command_list):
    cdef unsigned long long instance_size
    command_list_get_instance_size_extern(<CommandList*>command_list, &instance_size)
    return instance_size

cpdef inline command_list_submit(unsigned long long command_list, cnp.ndarray data, unsigned int instance_count, int device):
    cdef int devices[1]
    devices[0] = device

    command_list_submit_extern(<CommandList*>command_list, <void*>data.data, instance_count, devices, 1, <int*>0)
