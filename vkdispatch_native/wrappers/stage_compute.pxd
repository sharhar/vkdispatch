# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/stage_compute.hh":
    struct ComputePlan
    struct Context
    struct CommandList
    struct DescriptorSet

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
        const char* shader_name

    ComputePlan* stage_compute_plan_create_extern(Context* ctx, ComputePlanCreateInfo* create_info)
    void stage_compute_record_extern(CommandList* command_list, ComputePlan* plan, DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z)

cpdef inline stage_compute_plan_create(unsigned long long context, bytes shader_source, list bindings, unsigned int pc_size, bytes shader_name):
    cdef Context* ctx = <Context*>context

    cdef ComputePlanCreateInfo create_info
    create_info.shader_source = shader_source
    create_info.descriptorTypes = <DescriptorType*>malloc(len(bindings) * sizeof(DescriptorType))
    create_info.binding_count = len(bindings)
    create_info.pc_size = pc_size
    create_info.shader_name = shader_name

    for i in range(len(bindings)):
        create_info.descriptorTypes[i] = bindings[i]

    cdef ComputePlan* plan = stage_compute_plan_create_extern(ctx, &create_info)

    free(create_info.descriptorTypes)

    return <unsigned long long>plan

cpdef inline stage_compute_record(unsigned long long command_list, unsigned long long plan, unsigned long long descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z):
    cdef CommandList* cl = <CommandList*>command_list
    cdef ComputePlan* p = <ComputePlan*>plan
    cdef DescriptorSet* ds = <DescriptorSet*>descriptor_set
    stage_compute_record_extern(cl, p, ds, blocks_x, blocks_y, blocks_z)

