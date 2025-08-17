#ifndef _STAGES_EXTERN_H_
#define _STAGES_EXTERN_H_

#include "../base.hh"

enum DescriptorType {
    DESCRIPTOR_TYPE_STORAGE_BUFFER = 1,
    DESCRIPTOR_TYPE_STORAGE_IMAGE = 2,
    DESCRIPTOR_TYPE_UNIFORM_BUFFER = 3,
    DESCRIPTOR_TYPE_UNIFORM_IMAGE = 4,
    DESCRIPTOR_TYPE_SAMPLER = 5,
};

struct ComputePlanCreateInfo {
    const char* shader_source;
    DescriptorType* descriptorTypes;
    unsigned int binding_count;
    unsigned int pc_size;
    const char* shader_name;
};

struct ComputeRecordInfo {
    struct ComputePlan* plan;
    struct DescriptorSet* descriptor_set;
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int pc_size;
};

struct FFTInitRecordInfo {
    struct FFTPlan* plan;
};

struct FFTExecRecordInfo {
    struct FFTPlan* plan;
    struct Buffer* buffer;
    int inverse;
};

struct FFTPlan* stage_fft_plan_create_extern(
    struct Context* ctx, 
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
    int keep_shader_code);

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info);
void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, struct DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z);

void stage_fft_record_extern(
    struct CommandList* command_list, 
    struct FFTPlan* plan, 
    struct Buffer* buffer, 
    int inverse, 
    struct Buffer* kernel,
    struct Buffer* input_buffer);

const char* stage_fft_axis_code(struct FFTPlan* plan, int axis);

#endif // _STAGES_EXTERN_H_