#include "internal.h"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;

    for (int i = 0; i < plan->ctx->deviceCount; i++) {
        descriptor_set->pools.push_back(plan->ctx->devices[i].createDescriptorPool(
            vk::DescriptorPoolCreateInfo()
            .setMaxSets(1)
            .setPoolSizes(plan->poolSizes[i])
        ));

        descriptor_set->sets.push_back(plan->ctx->devices[i].allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(descriptor_set->pools[i])
            .setSetLayouts(plan->descriptorSetLayouts[i])
            .setDescriptorSetCount(1)
        )[0]);
    }

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    /*

    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        delete descriptor_set->descriptorSets[i];
    }

    delete[] descriptor_set->descriptorSets;
    delete descriptor_set;
    */
}

void descriptor_set_write_buffer_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object) {
    struct Context* ctx = (struct Context*)descriptor_set->plan->ctx;
    struct Buffer* buffer = (struct Buffer*)object;

    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        ctx->devices[i].updateDescriptorSets(
            vk::WriteDescriptorSet()
            .setDstSet(descriptor_set->sets[i])
            .setDstBinding(binding)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(
                vk::DescriptorBufferInfo()
                .setBuffer(buffer->buffers[i])
                .setOffset(0)
                .setRange(VK_WHOLE_SIZE)
            ), nullptr
        );
    }
}