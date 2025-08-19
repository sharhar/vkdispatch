#include "buffer.hh"
#include "objects_extern.hh"
#include "command_list.hh"

#include "../context/context.hh"
#include "../queue/signal.hh"

#include <cstring>

struct Buffer* buffer_create_extern(struct Context* ctx, unsigned long long size, int per_device) {
    if(size == 0) {
        set_error("Buffer size cannot be zero");
        return NULL;
    }

    sizeof(std::function<void(VkCommandBuffer, int, int)>);
    
    struct Buffer* buffer = new struct Buffer();
    
    LOG_INFO("Creating buffer of size %d with handle %p", size, buffer);
    
    buffer->ctx = ctx;
    buffer->size = size;

    buffer->signals_pointers_handle = ctx->handle_manager->register_queue_handle("Buffer Signals");

    for(int queue_index = 0; ctx->queues.size() > queue_index; queue_index++) {
        ctx->handle_manager->set_handle(queue_index, buffer->signals_pointers_handle, (uint64_t)new Signal());
    }

    buffer->buffers_handle = ctx->handle_manager->register_queue_handle("Buffer");
    buffer->allocations_handle = ctx->handle_manager->register_queue_handle("Buffer Allocations");
    buffer->staging_buffers_handle = ctx->handle_manager->register_queue_handle("Staging Buffer");
    buffer->staging_allocations_handle = ctx->handle_manager->register_queue_handle("Staging Buffer Allocations");

    uint64_t signals_pointers_handle = buffer->signals_pointers_handle;
    uint64_t buffers_handle = buffer->buffers_handle;
    uint64_t allocations_handle = buffer->allocations_handle;
    uint64_t staging_buffers_handle = buffer->staging_buffers_handle;
    uint64_t staging_allocations_handle = buffer->staging_allocations_handle;

    context_submit_command(ctx, "buffer-init", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, size, buffers_handle, allocations_handle, staging_buffers_handle, staging_allocations_handle, signals_pointers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkBufferCreateInfo bufferCreateInfo;
            memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = size;
            bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

            VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
            vmaAllocationCreateInfo.flags = 0;
            vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

            VkBuffer h_buffer;
            VmaAllocation h_allocation;

            ctx->vma_mutex.lock();
            VK_CALL(vmaCreateBuffer(ctx->allocators[indicies.device_index], &bufferCreateInfo, &vmaAllocationCreateInfo, &h_buffer, &h_allocation, NULL));
            ctx->vma_mutex.unlock();

            VkBufferCreateInfo stagingBufferCreateInfo;
            memset(&stagingBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
            stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingBufferCreateInfo.size = size;
            stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            vmaAllocationCreateInfo = {};
            vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

            VkBuffer h_staging_buffer;
            VmaAllocation h_staging_allocation;

            ctx->vma_mutex.lock();
            VK_CALL(vmaCreateBuffer(ctx->allocators[indicies.device_index], &stagingBufferCreateInfo, &vmaAllocationCreateInfo, &h_staging_buffer, &h_staging_allocation, NULL));
            ctx->vma_mutex.unlock();

            ctx->handle_manager->set_handle(indicies.queue_index, buffers_handle, (uint64_t)h_buffer);
            ctx->handle_manager->set_handle(indicies.queue_index, allocations_handle, (uint64_t)h_allocation);
            ctx->handle_manager->set_handle(indicies.queue_index, staging_buffers_handle, (uint64_t)h_staging_buffer);
            ctx->handle_manager->set_handle(indicies.queue_index, staging_allocations_handle, (uint64_t)h_staging_allocation);

            Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
            signal->notify();
    });

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    struct Context* ctx = buffer->ctx;

    for(int i = 0; i < ctx->queues.size(); i++) {
        int queue_index = i;

        uint64_t signals_pointers_handle = buffer->signals_pointers_handle;
        Signal* signal = (Signal*)ctx->handle_manager->get_handle(queue_index, signals_pointers_handle, 0);

        // wait for the recording thread to finish
        signal->wait();

        ctx->handle_manager->destroy_handle(queue_index, buffer->signals_pointers_handle);
    }

    uint64_t buffers_handle = buffer->buffers_handle;
    uint64_t allocations_handle = buffer->allocations_handle;
    uint64_t staging_buffers_handle = buffer->staging_buffers_handle;
    uint64_t staging_allocations_handle = buffer->staging_allocations_handle;

    context_submit_command(ctx, "buffer-destroy", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, buffers_handle, allocations_handle, staging_buffers_handle, staging_allocations_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            uint64_t buffer_timestamp = ctx->handle_manager->get_handle_timestamp(indicies.queue_index, buffers_handle);
            ctx->queues[indicies.queue_index]->wait_for_timestamp(buffer_timestamp);

            VkBuffer buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, buffers_handle, 0);
            VmaAllocation allocation = (VmaAllocation)ctx->handle_manager->get_handle(indicies.queue_index, allocations_handle, 0);
            VkBuffer stagingBuffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, staging_buffers_handle, 0);
            VmaAllocation stagingAllocation = (VmaAllocation)ctx->handle_manager->get_handle(indicies.queue_index, staging_allocations_handle, 0);

            ctx->vma_mutex.lock();

            if (buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(ctx->allocators[indicies.device_index], buffer, allocation);
            }
            if (stagingBuffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(ctx->allocators[indicies.device_index], stagingBuffer, stagingAllocation);
            }

            ctx->vma_mutex.unlock();

            ctx->handle_manager->destroy_handle(indicies.queue_index, buffers_handle);
            ctx->handle_manager->destroy_handle(indicies.queue_index, allocations_handle);
            ctx->handle_manager->destroy_handle(indicies.queue_index, staging_buffers_handle);
            ctx->handle_manager->destroy_handle(indicies.queue_index, staging_allocations_handle);

            LOG_VERBOSE("Buffer destroyed for device %d on queue %d recorder %d", indicies.device_index, indicies.queue_index, indicies.recorder_index);
        }
    );

    delete buffer;
}

void write_to_buffer(Context* ctx, struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int queue_index) {
    int device_index = ctx->queues[queue_index]->device_index;

    uint64_t signals_pointers_handle = buffer->signals_pointers_handle;
    Signal* signal = (Signal*)ctx->handle_manager->get_handle(queue_index, signals_pointers_handle, 0);

    // wait for the recording thread to finish
    signal->wait();
    signal->reset();

    // wait for the staging buffer to be ready
    uint64_t staging_buffer_timestamp = ctx->handle_manager->get_handle_timestamp(queue_index, buffer->staging_buffers_handle);
    ctx->queues[queue_index]->wait_for_timestamp(staging_buffer_timestamp);

    VmaAllocation staging_allocation = (VmaAllocation)ctx->handle_manager->get_handle(queue_index, buffer->staging_allocations_handle, 0);

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[device_index], staging_allocation, &mapped));
    memcpy(mapped, data, size);
    vmaUnmapMemory(ctx->allocators[device_index], staging_allocation);

    uint64_t buffers_handle = buffer->buffers_handle;
    uint64_t staging_buffers_handle = buffer->staging_buffers_handle;

    context_submit_command(ctx, "buffer-write", queue_index, NULL, RECORD_TYPE_SYNC,
        [ctx, offset, size, buffers_handle, staging_buffers_handle, signals_pointers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkBufferCopy bufferCopy;
            bufferCopy.size = size;
            bufferCopy.dstOffset = offset;
            bufferCopy.srcOffset = 0;

            VkMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                0,
                VK_ACCESS_HOST_WRITE_BIT,      // Source: Previous writes by the Host
                VK_ACCESS_TRANSFER_READ_BIT,   // Destination: vkCmdCopyBuffer will read
            };
            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_HOST_BIT,          // All operations on the host are done
                VK_PIPELINE_STAGE_TRANSFER_BIT,      // The transfer operation will start
                0, // No dependency flags needed here typically
                1, &barrier, 0, NULL, 0, NULL
            );

            VkBuffer stagingBuffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, staging_buffers_handle, timestamp);
            VkBuffer buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, buffers_handle, timestamp);

            vkCmdCopyBuffer(cmd_buffer, stagingBuffer, buffer, 1, &bufferCopy);

            Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
            signal->notify();
        }
    );
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index) {
    LOG_INFO("Writing data to buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    if(index != -1) {
        write_to_buffer(ctx, buffer, data, offset, size, index);
        return;
    }

    for(int i = 0; i < ctx->queues.size(); i++) {
        write_to_buffer(ctx, buffer, data, offset, size, i);
    }
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int queue_index) {
    LOG_INFO("Reading data from buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    uint64_t signals_pointers_handle = buffer->signals_pointers_handle;
    Signal* signal = (Signal*)ctx->handle_manager->get_handle(queue_index, signals_pointers_handle, 0);

    // wait for the recording thread to finish
    signal->wait();
    signal->reset();

    uint64_t buffers_handle = buffer->buffers_handle;
    uint64_t staging_buffers_handle = buffer->staging_buffers_handle;

    context_submit_command(ctx, "buffer-read", queue_index, NULL, RECORD_TYPE_SYNC,
        [ctx, offset, size, buffers_handle, staging_buffers_handle, signals_pointers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkBuffer stagingBuffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, staging_buffers_handle, timestamp);
            VkBuffer buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, buffers_handle, timestamp);

            VkBufferCopy bufferCopy;
            bufferCopy.size = size;
            bufferCopy.dstOffset = 0;
            bufferCopy.srcOffset = offset;

            vkCmdCopyBuffer(cmd_buffer, buffer, stagingBuffer, 1, &bufferCopy);

            VkMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                0,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT,
            };
            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, // No dependency flags needed here typically
                1, &barrier, 0, NULL, 0, NULL
            );

            Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
            signal->notify();
        }
    );

    // wait for the recording thread to finish again
    signal->wait();

    // wait for the staging buffer to be ready
    uint64_t staging_buffer_timestamp = ctx->handle_manager->get_handle_timestamp(queue_index, buffer->staging_buffers_handle);
    ctx->queues[queue_index]->wait_for_timestamp(staging_buffer_timestamp);
    
    int device_index = ctx->queues[queue_index]->device_index;

    VmaAllocation staging_allocation = (VmaAllocation)ctx->handle_manager->get_handle(queue_index, buffer->staging_allocations_handle, 0);
    
    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[device_index], staging_allocation, &mapped));
    memcpy(data, mapped, size);
    vmaUnmapMemory(ctx->allocators[device_index], staging_allocation);
}