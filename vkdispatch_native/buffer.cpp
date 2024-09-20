#include "internal.h"

struct Buffer* buffer_create_extern(struct Context* ctx, unsigned long long size, int per_device) {
    struct Buffer* buffer = new struct Buffer();
    
    LOG_INFO("Creating buffer of size %d with handle %p", size, buffer);
    
    buffer->ctx = ctx;
    
    LOG_INFO("Creating %d buffers (one per stream)", ctx->stream_indicies.size());

    buffer->allocations.resize(ctx->stream_indicies.size());
    buffer->buffers.resize(ctx->stream_indicies.size());
    buffer->stagingAllocations.resize(ctx->stream_indicies.size());
    buffer->stagingBuffers.resize(ctx->stream_indicies.size());

    for (int i = 0; i < buffer->buffers.size(); i++) {
        int device_index = ctx->stream_indicies[i].first;

        VkBufferCreateInfo bufferCreateInfo;
        memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = 0;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[device_index], &bufferCreateInfo, &vmaAllocationCreateInfo, &buffer->buffers[i], &buffer->allocations[i], NULL));

        VkBufferCreateInfo stagingBufferCreateInfo;
        memset(&stagingBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferCreateInfo.size = size;
        stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[device_index], &stagingBufferCreateInfo, &vmaAllocationCreateInfo, &buffer->stagingBuffers[i], &buffer->stagingAllocations[i], NULL));        
    }

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    LOG_INFO("Destroying buffer with handle %p", buffer);

    for(int i = 0; i < buffer->buffers.size(); i++) {
        int device_index = buffer->ctx->stream_indicies[i].first;

        vmaDestroyBuffer(buffer->ctx->allocators[device_index], buffer->buffers[i], buffer->allocations[i]);
        vmaDestroyBuffer(buffer->ctx->allocators[device_index], buffer->stagingBuffers[i], buffer->stagingAllocations[i]);
    }

    delete buffer;
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index) {
    LOG_INFO("Writing data to buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    command_list_begin_extern(ctx->command_list);
    command_list_append_command(ctx->command_list, [buffer, size, offset](VkDevice device, VkCommandBuffer cmd_buffer, int stream_index) {
        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;

        vkCmdCopyBuffer(cmd_buffer, buffer->stagingBuffers[stream_index], buffer->buffers[stream_index], 1, &bufferCopy);
    });
    command_list_end_extern(ctx->command_list);
    RETURN_ON_ERROR(;)

    int enum_count = index == -1 ? buffer->buffers.size() : 1;
    int start_index = index == -1 ? 0 : index;

    for (int i = 0; i < enum_count; i++) {
        int buffer_index = start_index + i;

        LOG_INFO("Writing data to buffer %d", buffer_index);

        int device_index = 0;
        
        auto stream_index = ctx->stream_indicies[buffer_index];
        device_index = stream_index.first;

        LOG_INFO("Writing data to buffer %d in device %d", buffer_index, device_index);

        void* mapped;
        VK_CALL(vmaMapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index], &mapped));
        memcpy(mapped, data, size);
        vmaUnmapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index]);

        command_list_submit_extern(ctx->command_list, NULL, 0, buffer_index, NULL);
        RETURN_ON_ERROR(;)
    }

    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(;)
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index) {
    LOG_INFO("Reading data from buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    int device_index = 0;
    
    auto stream_index = ctx->stream_indicies[index];
    device_index = stream_index.first;

    command_list_submit_command_and_reset(ctx->command_list, index, [buffer, size, offset](VkDevice device, VkCommandBuffer cmd_buffer, int stream_index) {
        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = offset;

        vkCmdCopyBuffer(cmd_buffer, buffer->buffers[stream_index], buffer->stagingBuffers[stream_index], 1, &bufferCopy);
    });

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[device_index], buffer->stagingAllocations[index], &mapped));
    memcpy(data, mapped, size);
    vmaUnmapMemory(ctx->allocators[device_index], buffer->stagingAllocations[index]);
}
