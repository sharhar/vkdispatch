#include "internal.h"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;
    descriptor_set->pools.resize(plan->ctx->deviceCount);
    descriptor_set->sets.resize(plan->ctx->deviceCount);

    for (int i = 0; i < plan->ctx->deviceCount; i++) {
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
        memset(&descriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = plan->poolSizes[i].size();
        descriptorPoolCreateInfo.pPoolSizes = plan->poolSizes[i].data();

        VK_CALL_RETNULL(vkCreateDescriptorPool(plan->ctx->devices[i], &descriptorPoolCreateInfo, NULL, &descriptor_set->pools[i]));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        memset(&descriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptor_set->pools[i];
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &plan->descriptorSetLayouts[i];

        VK_CALL_RETNULL(vkAllocateDescriptorSets(plan->ctx->devices[i], &descriptorSetAllocateInfo, &descriptor_set->sets[i]));
    }

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        vkDestroyDescriptorPool(descriptor_set->plan->ctx->devices[i], descriptor_set->pools[i], NULL);        
    }

    delete descriptor_set;
}

void descriptor_set_write_buffer_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object, unsigned long long offset, unsigned long long range) {
    struct Buffer* buffer = (struct Buffer*)object;

    for (int i = 0; i < descriptor_set->plan->ctx->deviceCount; i++) {
        VkDescriptorBufferInfo buffDesc;
        buffDesc.buffer = buffer->buffers[i]; //->handle();
        buffDesc.offset = offset;
        buffDesc.range = range == 0 ? VK_WHOLE_SIZE : range;

        VkWriteDescriptorSet writeDescriptor;
        memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
        writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptor.dstSet = descriptor_set->sets[i];
        writeDescriptor.dstBinding = binding;
        writeDescriptor.dstArrayElement = 0;
        writeDescriptor.descriptorCount = 1;
        writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptor.pImageInfo = NULL;
        writeDescriptor.pBufferInfo = &buffDesc;
        writeDescriptor.pTexelBufferView = NULL;

        vkUpdateDescriptorSets(descriptor_set->plan->ctx->devices[i], 1, &writeDescriptor, 0, NULL);
    }
}