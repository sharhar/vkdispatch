# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/command_list.hh":
    struct Context
    struct CommandList

    CommandList* command_list_create_extern(Context* context)
    void command_list_destroy_extern(CommandList* command_list)
    unsigned long long command_list_get_instance_size_extern(CommandList* command_list) 
    void command_list_reset_extern(CommandList* command_list)
    void command_list_submit_extern(CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int* indicies, int count, void* signal)

cpdef inline command_list_create(unsigned long long context):
    return <unsigned long long>command_list_create_extern(<Context*>context)

cpdef inline command_list_destroy(unsigned long long command_list):
    command_list_destroy_extern(<CommandList*>command_list)

cpdef inline command_list_get_instance_size(unsigned long long command_list):
    return command_list_get_instance_size_extern(<CommandList*>command_list)

cpdef inline command_list_reset(unsigned long long command_list):
    command_list_reset_extern(<CommandList*>command_list)

cpdef inline command_list_submit(unsigned long long command_list, bytes data, unsigned int instance_count, list[int] indicies):
    cdef int len_indicies = len(indicies)
    cdef int* indicies_c = <int*>malloc(len_indicies * sizeof(int))

    for i in range(len_indicies):
        indicies_c[i] = indicies[i]

    cdef const char* data_view = NULL
    if data is not None:
        data_view = data

    command_list_submit_extern(<CommandList*>command_list, <void*>data_view, instance_count, indicies_c, len_indicies, <void*>0)
