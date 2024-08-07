# distutils: language=c++
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "stage_compute.h":
    struct ComputePlan
    struct Context
    struct CommandList

    enum DescriptorType:
        DESCRIPTOR_TYPE_STORAGE_BUFFER = 1
        DESCRIPTOR_TYPE_STORAGE_IMAGE = 2
        DESCRIPTOR_TYPE_UNIFORM_BUFFER = 3
        DESCRIPTOR_TYPE_UNIFORM_IMAGE = 4
        DESCRIPTOR_TYPE_SAMPLER = 5
    
    struct ComputePlanCreateInfo:
        const char* shader_source
        DescriptorType* descriptorTypes
        unsigned int binding_count
        unsigned int pc_size

    ComputePlan* stage_compute_plan_create_extern(Context* ctx, ComputePlanCreateInfo* create_info)
    void stage_compute_bind_extern(ComputePlan* plan, unsigned int binding, void* object)
    void stage_compute_record_extern(CommandList* command_list, ComputePlan* plan, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z)

cpdef inline stage_compute_plan_create(unsigned long long context, bytes shader_source, unsigned int binding_count, unsigned int pc_size):
    cdef Context* ctx = <Context*>context

    cdef ComputePlanCreateInfo create_info
    create_info.shader_source = shader_source
    create_info.descriptorTypes = <DescriptorType*>malloc(binding_count * sizeof(DescriptorType))
    create_info.binding_count = binding_count
    create_info.pc_size = pc_size

    for i in range(binding_count):
        create_info.descriptorTypes[i] = DESCRIPTOR_TYPE_STORAGE_BUFFER

    cdef ComputePlan* plan = stage_compute_plan_create_extern(ctx, &create_info)

    free(create_info.descriptorTypes)

    return <unsigned long long>plan

cpdef inline stage_compute_bind(unsigned long long plan, unsigned int binding, unsigned long long object):
    cdef ComputePlan* p = <ComputePlan*>plan
    stage_compute_bind_extern(p, binding, <void*>object)

cpdef inline stage_compute_record(unsigned long long command_list, unsigned long long plan, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z):
    cdef CommandList* cl = <CommandList*>command_list
    cdef ComputePlan* p = <ComputePlan*>plan
    stage_compute_record_extern(cl, p, blocks_x, blocks_y, blocks_z)

