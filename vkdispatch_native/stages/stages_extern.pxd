# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "stages/stages_extern.hh":
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

    struct Context
    struct Buffer
    struct CommandList
    struct FFTPlan

    FFTPlan* stage_fft_plan_create_extern(
        Context* ctx, 
        unsigned long long dims, 
        unsigned long long rows,
        unsigned long long cols, 
        unsigned long long depth, 
        unsigned long long buffer_size, 
        unsigned int do_r2c,
        int omit_rows,
        int omit_cols,
        int omit_depth,
        int normalize,
        unsigned long long pad_left_rows, unsigned long long pad_right_rows,
        unsigned long long pad_left_cols, unsigned long long pad_right_cols,
        unsigned long long pad_left_depth, unsigned long long pad_right_depth,
        int frequency_zeropadding,
        int kernel_num,
        int kernel_convolution,
        int conjugate_convolution,
        int convolution_features,
        unsigned long long input_buffer_size,
        int num_batches,
        int single_kernel_multiple_batches,
        int keep_shader_code)
    void stage_fft_record_extern(
        CommandList* command_list, 
        FFTPlan* plan,
        Buffer* buffer, int inverse,
        Buffer* kernel,
        Buffer* input_buffer)

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

cpdef inline stage_fft_plan_create(
    unsigned long long context, 
    list[int] dims, 
    list[int] axes, 
    unsigned long long buffer_size,
    bool do_r2c, 
    bool normalize,
    tuple[int, int, int] pad_left,
    tuple[int, int, int] pad_right,
    bool frequency_zeropadding,
    int kernel_num,
    bool kernel_convolution,
    bool conjugate_convolution,
    int convolution_features,
    unsigned long long input_buffer_size,
    int num_batches,
    bool single_kernel_multiple_batches,
    bool keep_shader_code):
    assert len(dims) > 0 and len(dims) < 4, "dims must be a list of length 1, 2, or 3"
    assert len(axes) <= 3, "axes must be a list of length less than or equal to 3"

    for ax in axes:
        assert ax < len(dims), "axes must be less than the length of dims"

    cdef Context* ctx = <Context*>context
    cdef unsigned long long dims_ = len(dims)

    cdef unsigned long long* dims__ = <unsigned long long*>malloc(3 * sizeof(unsigned long long))
    cdef int* omits__ = <int*>malloc(3 * sizeof(int))

    for i in range(3):
        dims__[i] = 1
        omits__[i] = 1

    for i in range(dims_):
        dims__[i] = dims[i]

    for i in range(len(axes)):
        if 0 <= axes[i] < 3:  # Ensure the index is within bounds
            omits__[axes[i]] = 0
        else:
            print("Invalid axis index: ", axes[i])
            sys.exit(1)
    
    cdef FFTPlan* plan = stage_fft_plan_create_extern(
        ctx, 
        dims_, dims__[0], dims__[1], dims__[2], 
        buffer_size,
        1 if do_r2c else 0,
        omits__[0], omits__[1], omits__[2], 
        1 if normalize else 0,
        pad_left[0], pad_right[0],
        pad_left[1], pad_right[1],
        pad_left[2], pad_right[2],
        1 if frequency_zeropadding else 0,
        kernel_num,
        1 if kernel_convolution else 0,
        1 if conjugate_convolution else 0,
        convolution_features,
        input_buffer_size,
        num_batches,
        1 if single_kernel_multiple_batches else 0,
        1 if keep_shader_code else 0)

    free(dims__)

    return <unsigned long long>plan

cpdef inline stage_fft_record(
    unsigned long long command_list, 
    unsigned long long plan, 
    unsigned long long buffer, int inverse,
    unsigned long long kernel,
    unsigned long long input_buffer):
    cdef CommandList* cl = <CommandList*>command_list
    cdef FFTPlan* p = <FFTPlan*>plan
    cdef Buffer* b = <Buffer*>buffer
    cdef Buffer* k = <Buffer*>kernel
    cdef Buffer* i = <Buffer*>input_buffer

    stage_fft_record_extern(cl, p, b, inverse, k, i)