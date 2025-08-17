#include "../internal.hh"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;
    descriptor_set->sets_handle = plan->ctx->handle_manager->register_stream_handle("DescriptorSet");
    descriptor_set->pools_handle = plan->ctx->handle_manager->register_stream_handle("DescriptorPool");

    struct Context* ctx = plan->ctx;

    uint64_t descriptor_set_layouts_handle = plan->descriptorSetLayouts_handle;

    uint64_t sets_handle = descriptor_set->sets_handle;
    uint64_t pools_handle = descriptor_set->pools_handle;

    unsigned int binding_count = plan->binding_count;
    VkDescriptorPoolSize* poolSizes = plan->poolSizes;

    command_list_record_command(ctx->command_list, 
        "descriptor-set-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, descriptor_set_layouts_handle, sets_handle, pools_handle, binding_count, poolSizes]
        (VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data, BarrierManager* barrier_manager) {
            LOG_VERBOSE("Creating Descriptor Set for device %d on stream %d recorder %d", device_index, stream_index, recorder_index);

            VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
            memset(&descriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
            descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCreateInfo.maxSets = 1;
            descriptorPoolCreateInfo.poolSizeCount = binding_count;
            descriptorPoolCreateInfo.pPoolSizes = poolSizes;

            LOG_VERBOSE("Creating Descriptor Pool for device %d on stream %d recorder %d", device_index, stream_index, recorder_index);

            VkDescriptorPool pool;
            VK_CALL(vkCreateDescriptorPool(ctx->devices[device_index], &descriptorPoolCreateInfo, NULL, &pool));

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
            memset(&descriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
            descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocateInfo.descriptorPool = pool;
            descriptorSetAllocateInfo.descriptorSetCount = 1;

            LOG_VERBOSE("Descriptor Set Layout Handle: %d", (uint64_t)ctx->handle_manager->get_handle(stream_index, descriptor_set_layouts_handle));

            descriptorSetAllocateInfo.pSetLayouts = (VkDescriptorSetLayout*)ctx->handle_manager->get_handle_pointer(stream_index, descriptor_set_layouts_handle);

            VkDescriptorSet set;
            VK_CALL(vkAllocateDescriptorSets(ctx->devices[device_index], &descriptorSetAllocateInfo, &set));

            ctx->handle_manager->set_handle(stream_index, sets_handle, (uint64_t)set);
            ctx->handle_manager->set_handle(stream_index, pools_handle, (uint64_t)pool);
        }
    );

    int submit_index = -2;
    command_list_submit_extern(plan->ctx->command_list, NULL, 1, submit_index, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(plan->ctx->command_list);
    RETURN_ON_ERROR(NULL)

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    struct Context* ctx = descriptor_set->plan->ctx;

    // for (int i = 0; i < descriptor_set->plan->ctx->streams.size(); i++) {
    //     VkDescriptorPool pool = descriptor_set->pools[i];

    //      command_list_record_command(descriptor_set->plan->ctx->command_list, 
    //         "descriptor-set-destroy",
    //         0,
    //         VK_PIPELINE_STAGE_TRANSFER_BIT,
    //         [ctx, pool](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
    //             vkDestroyDescriptorPool(ctx->devices[device_index], pool, NULL);
    //         }
    //     );

    //     command_list_submit_extern(ctx->command_list, NULL, 1, &i, 1, NULL, RECORD_TYPE_SYNC);
    //     command_list_reset_extern(ctx->command_list);
    //     RETURN_ON_ERROR(;)
    // }

    delete descriptor_set;
}

void descriptor_set_write_buffer_extern(
    struct DescriptorSet* descriptor_set,
    unsigned int binding,
    void* object,
    unsigned long long offset,
    unsigned long long range,
    int uniform,
    int read_access,
    int write_access) {

    struct Buffer* buffer = (struct Buffer*)object;

    struct Context* ctx = descriptor_set->plan->ctx;

    descriptor_set->buffer_barriers.push_back({buffer, read_access, write_access});

    uint64_t sets_handle = descriptor_set->sets_handle;

    command_list_record_command(ctx->command_list, 
        "descriptor-set-write-buffer",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [buffer, ctx, sets_handle, offset, range, binding, uniform](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data, BarrierManager* barrier_manager) {
            VkDescriptorBufferInfo buffDesc;
            buffDesc.buffer = buffer->buffers[stream_index];

            buffDesc.offset = offset;
            buffDesc.range = range == 0 ? VK_WHOLE_SIZE : range;

            VkWriteDescriptorSet writeDescriptor;
            memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
            writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptor.dstSet = (VkDescriptorSet)ctx->handle_manager->get_handle(stream_index, sets_handle);
            writeDescriptor.dstBinding = binding;
            writeDescriptor.dstArrayElement = 0;
            writeDescriptor.descriptorCount = 1;
            writeDescriptor.descriptorType = uniform == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptor.pImageInfo = NULL;
            writeDescriptor.pBufferInfo = &buffDesc;
            writeDescriptor.pTexelBufferView = NULL;

            vkUpdateDescriptorSets(ctx->devices[device_index], 1, &writeDescriptor, 0, NULL);
        }
    );

    int submit_index = -2;
    command_list_submit_extern(ctx->command_list, NULL, 1, submit_index, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(;)
}


void descriptor_set_write_image_extern(
    struct DescriptorSet* descriptor_set,
    unsigned int binding,
    void* object,
    void* sampler_obj,
    int read_access,
    int write_access) {

    struct Image* image = (struct Image*)object;
    struct Sampler* sampler = (struct Sampler*)sampler_obj;

    struct Context* ctx = descriptor_set->plan->ctx;

    uint64_t sets_handle = descriptor_set->sets_handle;

    command_list_record_command(ctx->command_list, 
        "descriptor-set-write-image",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [image, sampler, ctx, sets_handle, binding](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data, BarrierManager* barrier_manager) {
            VkDescriptorImageInfo imageDesc;
            imageDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageDesc.imageView = image->imageViews[stream_index];
            imageDesc.sampler = sampler->samplers[stream_index];

            VkWriteDescriptorSet writeDescriptor;
            memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
            writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptor.dstSet = (VkDescriptorSet)ctx->handle_manager->get_handle(stream_index, sets_handle);
            writeDescriptor.dstBinding = binding;
            writeDescriptor.dstArrayElement = 0;
            writeDescriptor.descriptorCount = 1;
            writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptor.pImageInfo = &imageDesc;
            writeDescriptor.pBufferInfo = NULL;
            writeDescriptor.pTexelBufferView = NULL;

            vkUpdateDescriptorSets(ctx->devices[device_index], 1, &writeDescriptor, 0, NULL);
        }
    );

    int submit_index = -2;
    command_list_submit_extern(ctx->command_list, NULL, 1, submit_index, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(;)
}