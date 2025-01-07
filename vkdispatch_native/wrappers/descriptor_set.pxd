# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/descriptor_set.hh":
    struct ComputePlan
    struct DescriptorSet

    DescriptorSet* descriptor_set_create_extern(ComputePlan* plan)
    void descriptor_set_destroy_extern(DescriptorSet* descriptor_set)

    void descriptor_set_write_buffer_extern(DescriptorSet* descriptor_set, unsigned int binding, void* object, unsigned long long offset, unsigned long long range, int uniform)
    void descriptor_set_write_image_extern(DescriptorSet* descriptor_set, unsigned int binding, void* object, void* sampler_obj)

cpdef inline descriptor_set_create(unsigned long long plan):
    cdef ComputePlan* p = <ComputePlan*>plan
    return <unsigned long long>descriptor_set_create_extern(p)

cpdef inline descriptor_set_destroy(unsigned long long descriptor_set):
    descriptor_set_destroy_extern(<DescriptorSet*>descriptor_set)

cpdef inline descriptor_set_write_buffer(unsigned long long descriptor_set, unsigned int binding, unsigned long long object, unsigned long long offset, unsigned long long range, int uniform):
    cdef DescriptorSet* ds = <DescriptorSet*>descriptor_set
    descriptor_set_write_buffer_extern(ds, binding, <void*>object, offset, range, uniform)

cpdef inline descriptor_set_write_image(unsigned long long descriptor_set, unsigned int binding, unsigned long long object, unsigned long long sampler_obj):
    cdef DescriptorSet* ds = <DescriptorSet*>descriptor_set
    descriptor_set_write_image_extern(ds, binding, <void*>object, <void*>sampler_obj)

