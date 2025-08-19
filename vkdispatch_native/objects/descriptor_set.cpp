#include "descriptor_set.hh"
#include "command_list.hh"
#include "buffer.hh"
#include "image.hh"

#include "../context/context.hh"
#include "../stages/stage_compute.hh"

#include "objects_extern.hh"

#include <shared_mutex>
#include <vector>

class BarrierInfoManager {
public:
    int ref_count;
    std::vector<BufferBarrierInfo*> buffer_barrier_lists;

    std::shared_mutex handle_mutex;

    BarrierInfoManager(int ref_count) {
        this->ref_count = ref_count;
    }

    void add_list(BufferBarrierInfo* list)  {
        std::unique_lock lock(handle_mutex);
        buffer_barrier_lists.push_back(list);
    }

    int decr_ref() {
        std::unique_lock lock(handle_mutex);
        ref_count--;

        if(ref_count > 0) {
            return ref_count;
        }

        for (BufferBarrierInfo* barrier_list : buffer_barrier_lists) {
            free(barrier_list);
        }
        buffer_barrier_lists.clear();
        
        return 0;
    }
};

void descriptor_set_add_buffer_info_list(struct DescriptorSet* desc_set, BufferBarrierInfo* list) {
    desc_set->barrier_info_manager->add_list(list);
}

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan) {
    struct DescriptorSet* descriptor_set = new struct DescriptorSet();
    descriptor_set->plan = plan;
    descriptor_set->sets_handle = plan->ctx->handle_manager->register_queue_handle("DescriptorSet");
    descriptor_set->pools_handle = plan->ctx->handle_manager->register_queue_handle("DescriptorPool");

    struct Context* ctx = plan->ctx;

    uint64_t descriptor_set_layouts_handle = plan->descriptorSetLayouts_handle;

    uint64_t sets_handle = descriptor_set->sets_handle;
    uint64_t pools_handle = descriptor_set->pools_handle;

    descriptor_set->barrier_info_manager = new BarrierInfoManager(ctx->queues.size());

    unsigned int binding_count = plan->binding_count;
    VkDescriptorPoolSize* poolSizes = plan->poolSizes;

    context_submit_command(ctx, "descriptor-set-init", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, descriptor_set_layouts_handle, sets_handle, pools_handle, binding_count, poolSizes]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            LOG_VERBOSE("Creating Descriptor Set for device %d on queue %d recorder %d", indicies.device_index, indicies.queue_index, indicies.recorder_index);

            VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
            memset(&descriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
            descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCreateInfo.maxSets = 1;
            descriptorPoolCreateInfo.poolSizeCount = binding_count;
            descriptorPoolCreateInfo.pPoolSizes = poolSizes;

            LOG_VERBOSE("Creating Descriptor Pool for device %d on queue %d recorder %d", indicies.device_index, indicies.queue_index, indicies.recorder_index);

            VkDescriptorPool h_pool;
            VK_CALL(vkCreateDescriptorPool(ctx->devices[indicies.device_index], &descriptorPoolCreateInfo, NULL, &h_pool));

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
            memset(&descriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
            descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocateInfo.descriptorPool = h_pool;
            descriptorSetAllocateInfo.descriptorSetCount = 1;

            //LOG_VERBOSE("Descriptor Set Layout Handle: %d", (uint64_t)ctx->handle_manager->get_handle(indicies.queue_index, descriptor_set_layouts_handle));

            descriptorSetAllocateInfo.pSetLayouts = (VkDescriptorSetLayout*)ctx->handle_manager->get_handle_pointer(indicies.queue_index, descriptor_set_layouts_handle, 0);

            VkDescriptorSet h_set;
            VK_CALL(vkAllocateDescriptorSets(ctx->devices[indicies.device_index], &descriptorSetAllocateInfo, &h_set));

            ctx->handle_manager->set_handle(indicies.queue_index, sets_handle, (uint64_t)h_set);
            ctx->handle_manager->set_handle(indicies.queue_index, pools_handle, (uint64_t)h_pool);
        }
    );

    return descriptor_set;
}

void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set) {
    LOG_VERBOSE("Destroying descriptor set with handle %p", descriptor_set);

    struct Context* ctx = descriptor_set->plan->ctx;

    uint64_t sets_handle = descriptor_set->sets_handle;
    uint64_t pools_handle = descriptor_set->pools_handle;

    BarrierInfoManager* barrier_info_manager = descriptor_set->barrier_info_manager;

    context_submit_command(ctx, "descriptor-set-destroy", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, sets_handle, pools_handle, barrier_info_manager]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            uint64_t set_timestamp = ctx->handle_manager->get_handle_timestamp(indicies.queue_index, sets_handle);

            ctx->queues[indicies.queue_index]->wait_for_timestamp(set_timestamp);

            VkDescriptorPool pool = (VkDescriptorPool)ctx->handle_manager->get_handle(indicies.queue_index, pools_handle, 0);

            if (pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(ctx->devices[indicies.device_index], pool, NULL);
            }

            ctx->handle_manager->destroy_handle(indicies.queue_index, sets_handle);
            ctx->handle_manager->destroy_handle(indicies.queue_index, pools_handle);

            LOG_VERBOSE("Descriptor Set destroyed for device %d on queue %d recorder %d", indicies.device_index, indicies.queue_index, indicies.recorder_index);

            int refs = barrier_info_manager->decr_ref();

            if(refs == 0) {
                delete barrier_info_manager;
            }
        }
    );

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

    uint64_t sets_handle = descriptor_set->sets_handle;
    uint64_t buffers_handle = buffer->buffers_handle;

    descriptor_set->buffer_barrier_list.push_back({buffers_handle, read_access, write_access});   
    
    context_submit_command(ctx, "descriptor-set-write-buffer", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, sets_handle, buffers_handle, offset, range, binding, uniform]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkDescriptorBufferInfo buffDesc;
            buffDesc.buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, buffers_handle, 0);

            buffDesc.offset = offset;
            buffDesc.range = range == 0 ? VK_WHOLE_SIZE : range;

            VkWriteDescriptorSet writeDescriptor;
            memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
            writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptor.dstSet = (VkDescriptorSet)ctx->handle_manager->get_handle(indicies.queue_index, sets_handle, 0);
            writeDescriptor.dstBinding = binding;
            writeDescriptor.dstArrayElement = 0;
            writeDescriptor.descriptorCount = 1;
            writeDescriptor.descriptorType = uniform == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptor.pImageInfo = NULL;
            writeDescriptor.pBufferInfo = &buffDesc;
            writeDescriptor.pTexelBufferView = NULL;

            vkUpdateDescriptorSets(ctx->devices[indicies.device_index], 1, &writeDescriptor, 0, NULL);
        }
    );
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
    uint64_t image_views_handle = image->image_views_handle;
    uint64_t samplers_handle = sampler->samplers_handle;

    context_submit_command(ctx, "descriptor-set-write-image", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, sets_handle, samplers_handle, image_views_handle, binding]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkSampler h_sampler = (VkSampler)ctx->handle_manager->get_handle(indicies.queue_index, samplers_handle, 0);
            VkImageView h_image_view = (VkImageView)ctx->handle_manager->get_handle(indicies.queue_index, image_views_handle, 0);

            VkDescriptorImageInfo imageDesc;
            imageDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageDesc.imageView = h_image_view;
            imageDesc.sampler = h_sampler;

            VkWriteDescriptorSet writeDescriptor;
            memset(&writeDescriptor, 0, sizeof(VkWriteDescriptorSet));
            writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptor.dstSet = (VkDescriptorSet)ctx->handle_manager->get_handle(indicies.queue_index, sets_handle, 0);
            writeDescriptor.dstBinding = binding;
            writeDescriptor.dstArrayElement = 0;
            writeDescriptor.descriptorCount = 1;
            writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptor.pImageInfo = &imageDesc;
            writeDescriptor.pBufferInfo = NULL;
            writeDescriptor.pTexelBufferView = NULL;

            vkUpdateDescriptorSets(ctx->devices[indicies.device_index], 1, &writeDescriptor, 0, NULL);
        }
    );
}

