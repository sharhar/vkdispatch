#include "../include/internal.hh"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;
    descriptor_set->pools.resize(plan->ctx->stream_indicies.size());
    descriptor_set->sets.resize(plan->ctx->stream_indicies.size());

    for (int i = 0; i < plan->ctx->stream_indicies.size(); i++) {
        int device_index = plan->ctx->stream_indicies[i].first;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
        memset(&descriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = plan->poolSizes[device_index].size();
        descriptorPoolCreateInfo.pPoolSizes = plan->poolSizes[device_index].data();

        VK_CALL_RETNULL(vkCreateDescriptorPool(plan->ctx->devices[device_index], &descriptorPoolCreateInfo, NULL, &descriptor_set->pools[i]));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        memset(&descriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptor_set->pools[i];
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &plan->descriptorSetLayouts[device_index];

        VK_CALL_RETNULL(vkAllocateDescriptorSets(plan->ctx->devices[device_index], &descriptorSetAllocateInfo, &descriptor_set->sets[i]));
    }

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    for (int i = 0; i < descriptor_set->plan->ctx->stream_indicies.size(); i++) {
        int device_index = descriptor_set->plan->ctx->stream_indicies[i].first;
        vkDestroyDescriptorPool(descriptor_set->plan->ctx->devices[device_index], descriptor_set->pools[i], NULL);        
    }

    delete descriptor_set;
}

void descriptor_set_write_buffer_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object, unsigned long long offset, unsigned long long range, int uniform) {
    struct Buffer* buffer = (struct Buffer*)object;

    for (int i = 0; i < descriptor_set->plan->ctx->stream_indicies.size(); i++) {
        int device_index = descriptor_set->plan->ctx->stream_indicies[i].first;

        VkDescriptorBufferInfo buffDesc;
        buffDesc.buffer = buffer->buffers[i];
        
        //if(buffer->per_device)
        //    buffDesc.buffer = buffer->buffers[device_index];

        buffDesc.offset = offset;
        buffDesc.range = range == 0 ? VK_WHOLE_SIZE : range;

        VkWriteDescriptorSet writeDescriptor;
        memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
        writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptor.dstSet = descriptor_set->sets[i];
        writeDescriptor.dstBinding = binding;
        writeDescriptor.dstArrayElement = 0;
        writeDescriptor.descriptorCount = 1;
        writeDescriptor.descriptorType = uniform == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptor.pImageInfo = NULL;
        writeDescriptor.pBufferInfo = &buffDesc;
        writeDescriptor.pTexelBufferView = NULL;

        vkUpdateDescriptorSets(descriptor_set->plan->ctx->devices[device_index], 1, &writeDescriptor, 0, NULL);
    }
}


void descriptor_set_write_image_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object, void* sampler_obj) {
    struct Image* image = (struct Image*)object;
    struct Sampler* sampler = (struct Sampler*)sampler_obj;

    for (int i = 0; i < descriptor_set->plan->ctx->stream_indicies.size(); i++) {
        int device_index = descriptor_set->plan->ctx->stream_indicies[i].first;

        VkDescriptorImageInfo imageDesc;
        imageDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageDesc.imageView = image->imageViews[i];
        imageDesc.sampler = sampler->samplers[i];

        VkWriteDescriptorSet writeDescriptor;
        memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
        writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptor.dstSet = descriptor_set->sets[i];
        writeDescriptor.dstBinding = binding;
        writeDescriptor.dstArrayElement = 0;
        writeDescriptor.descriptorCount = 1;
        writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptor.pImageInfo = &imageDesc;
        writeDescriptor.pBufferInfo = NULL;
        writeDescriptor.pTexelBufferView = NULL;

        vkUpdateDescriptorSets(descriptor_set->plan->ctx->devices[device_index], 1, &writeDescriptor, 0, NULL);
    }
}