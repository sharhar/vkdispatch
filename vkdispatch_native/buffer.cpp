#include "internal.h"

struct Buffer* buffer_create_extern(struct Context* ctx, unsigned long long size, int per_device) {
    struct Buffer* buffer = new struct Buffer();
    
    LOG_INFO("Creating buffer of size %d with handle %p", size, buffer);
    
    buffer->ctx = ctx;

    if (per_device) {
        LOG_INFO("Creating %d buffers for each device", ctx->deviceCount);

        buffer->allocations.resize(ctx->deviceCount);
        buffer->buffers.resize(ctx->deviceCount);
        buffer->stagingAllocations.resize(ctx->deviceCount);
        buffer->stagingBuffers.resize(ctx->deviceCount);
        buffer->fences.resize(ctx->deviceCount);
        buffer->per_device = true;
    } else {
        LOG_INFO("Creating %d buffers for each stream", ctx->stream_indicies.size());

        buffer->allocations.resize(ctx->stream_indicies.size());
        buffer->buffers.resize(ctx->stream_indicies.size());
        buffer->stagingAllocations.resize(ctx->stream_indicies.size());
        buffer->stagingBuffers.resize(ctx->stream_indicies.size());
        buffer->fences.resize(ctx->stream_indicies.size());
        buffer->per_device = false;
    }

    for (int i = 0; i < buffer->buffers.size(); i++) {
        int device_index = i; 

        if(!per_device) {
            device_index = ctx->stream_indicies[i].first;
        }

        VkBufferCreateInfo bufferCreateInfo;
        memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = 0;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		vmaCreateBuffer(ctx->allocators[device_index], &bufferCreateInfo, &vmaAllocationCreateInfo, &buffer->buffers[i], &buffer->allocations[i], NULL);

        VkBufferCreateInfo stagingBufferCreateInfo;
        memset(&stagingBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferCreateInfo.size = size;
        stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[device_index], &stagingBufferCreateInfo, &vmaAllocationCreateInfo, &buffer->stagingBuffers[i], &buffer->stagingAllocations[i], NULL));        

        VkFenceCreateInfo fenceCreateInfo;
        memset(&fenceCreateInfo, 0, sizeof(VkFenceCreateInfo));
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CALL_RETNULL(vkCreateFence(ctx->devices[device_index], &fenceCreateInfo, NULL, &buffer->fences[i]));
    }

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    LOG_INFO("Destroying buffer with handle %p", buffer);

    for(int i = 0; i < buffer->buffers.size(); i++) {
        int device_index = i; 

        if(!buffer->per_device) {
            device_index = buffer->ctx->stream_indicies[i].first;
        }

        vkDestroyFence(buffer->ctx->devices[device_index], buffer->fences[i], NULL);
        vmaDestroyBuffer(buffer->ctx->allocators[device_index], buffer->buffers[i], buffer->allocations[i]);
        vmaDestroyBuffer(buffer->ctx->allocators[device_index], buffer->stagingBuffers[i], buffer->stagingAllocations[i]);
    }

    delete buffer;
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index) {
    LOG_INFO("Writing data to buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    int enum_count = index == -1 ? buffer->buffers.size() : 1;
    int start_index = index == -1 ? 0 : index;

    for (int i = 0; i < enum_count; i++) {
        int buffer_index = start_index + i;

        LOG_INFO("Writing data to buffer %d", buffer_index);

        Stream* stream = NULL;
        int device_index = 0;

        if(buffer->per_device) {
            stream = ctx->streams[buffer_index][0];
            device_index = buffer_index;
        } else {
            auto stream_index = ctx->stream_indicies[buffer_index];
            device_index = stream_index.first;
            stream = ctx->streams[stream_index.first][stream_index.second];
        }

        LOG_INFO("Writing data to buffer %d with stream %p", buffer_index, stream);

        VK_CALL(vkQueueWaitIdle(stream->queue));

        void* mapped;
        VK_CALL(vmaMapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index], &mapped));
        memcpy(mapped, data, size);
        vmaUnmapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index]);  

        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;

        VkCommandBuffer cmdBuffer = stream->begin();
        if(__error_string != NULL)
            return;
        
        vkCmdCopyBuffer(cmdBuffer, buffer->stagingBuffers[buffer_index], buffer->buffers[buffer_index], 1, &bufferCopy);
        
        VkFence fence = stream->submit();
        if(__error_string != NULL)
            return;
        VK_CALL(vkWaitForFences(buffer->ctx->devices[device_index], 1, &fence, VK_TRUE, UINT64_MAX));
    }
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index) {
    LOG_INFO("Reading data from buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    int buffer_index = index;

    Stream* stream = NULL;
    int device_index = 0;

    if(buffer->per_device) {
        stream = ctx->streams[buffer_index][0];
        device_index = buffer_index;
    } else {
        auto stream_index = ctx->stream_indicies[buffer_index];
        device_index = stream_index.first;
        stream = ctx->streams[stream_index.first][stream_index.second];
    }

    VkBufferCopy bufferCopy;
	bufferCopy.size = size;
	bufferCopy.dstOffset = 0;
	bufferCopy.srcOffset = offset;
	
    VK_CALL(vkQueueWaitIdle(stream->queue));

    VkCommandBuffer cmdBuffer = stream->begin();
    if(__error_string != NULL)
        return;
    
    vkCmdCopyBuffer(cmdBuffer, buffer->buffers[buffer_index], buffer->stagingBuffers[buffer_index], 1, &bufferCopy);
    
    VkFence fence = stream->submit();
    if(__error_string != NULL)
        return;
    VK_CALL(vkWaitForFences(buffer->ctx->devices[buffer_index], 1, &fence, VK_TRUE, UINT64_MAX));

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index], &mapped));
    memcpy(data, mapped, size);
    vmaUnmapMemory(ctx->allocators[device_index], buffer->stagingAllocations[buffer_index]);
} 

void buffer_copy_extern(struct Buffer* src, struct Buffer* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index) {
    /*
    assert(src->ctx == dst->ctx);

    int enum_count = device_index == -1 ? src->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = dst_offset;
        bufferCopy.srcOffset = src_offset;

        src->ctx->queues[dev_index]->waitIdle();
        dst->buffers[dev_index]->copyFrom(src->buffers[dev_index], src->ctx->queues[dev_index], bufferCopy);
    }
    */
}