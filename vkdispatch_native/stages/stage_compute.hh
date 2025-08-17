#ifndef _STAGE_COMPUTE_H_
#define _STAGE_COMPUTE_H_

#include "../base.hh"

struct ComputePlan {
    struct Context* ctx;
    uint64_t descriptorSetLayouts_handle;
    uint64_t pipelineLayouts_handle;
    uint64_t pipelines_handle;
    
    VkDescriptorPoolSize* poolSizes;
    VkDescriptorSetLayoutBinding* bindings;

    unsigned int binding_count;
    unsigned int pc_size;

    uint32_t* code;
    size_t code_size;
};

#endif // _STAGE_COMPUTE_H_