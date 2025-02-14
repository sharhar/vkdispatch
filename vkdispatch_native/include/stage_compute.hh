#ifndef _STAGE_COMPUTE_H_
#define _STAGE_COMPUTE_H_

#include "base.hh"

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

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info);
void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, struct DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z);

#endif // _STAGE_COMPUTE_H_