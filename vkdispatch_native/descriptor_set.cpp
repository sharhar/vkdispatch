#include "internal.h"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;
    descriptor_set->descriptorSets = new VKLDescriptorSet*[plan->ctx->deviceCount];

    for (int i = 0; i < plan->ctx->deviceCount; i++) {
        descriptor_set->descriptorSets[i] = new VKLDescriptorSet(plan->pipelineLayouts[i], 0);
    }

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        delete descriptor_set->descriptorSets[i];
    }

    delete[] descriptor_set->descriptorSets;
    delete descriptor_set;
}

void descriptor_set_write_buffer_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object) {
    struct Buffer* buffer = (struct Buffer*)object;

    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        descriptor_set->descriptorSets[i]->writeBuffer(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer->buffers[i], 0, VK_WHOLE_SIZE);
    }
}