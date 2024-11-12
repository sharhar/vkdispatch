# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/conditional.hh":
    struct CommandList

    int record_conditional_extern(CommandList* command_list)
    void record_conditional_end_extern(CommandList* command_list)

cpdef inline record_conditional(unsigned long long command_list):
    return record_conditional_extern(<CommandList*>command_list)

cpdef inline record_conditional_end(unsigned long long command_list):
    return record_conditional_end_extern(<CommandList*>command_list)