#include "internal.h"

struct Buffer* buffer_create_extern(struct Context* ctx, unsigned long long size) {
    struct Buffer* buffer = new struct Buffer();
    
    LOG_INFO("Creating buffer of size %d with handle %p", size, buffer);
    
    buffer->ctx = ctx;
    buffer->allocations.resize(ctx->deviceCount);
    buffer->buffers.resize(ctx->deviceCount);
    buffer->stagingAllocations.resize(ctx->deviceCount);
    buffer->stagingBuffers.resize(ctx->deviceCount);
    buffer->fences.resize(ctx->deviceCount);

    for (int i = 0; i < ctx->deviceCount; i++) {
        VkBufferCreateInfo bufferCreateInfo;
        memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = 0;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		vmaCreateBuffer(ctx->allocators[i], &bufferCreateInfo, &vmaAllocationCreateInfo, &buffer->buffers[i], &buffer->allocations[i], NULL);

        VkBufferCreateInfo stagingBufferCreateInfo;
        memset(&stagingBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferCreateInfo.size = size;
        stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[i], &stagingBufferCreateInfo, &vmaAllocationCreateInfo, &buffer->stagingBuffers[i], &buffer->stagingAllocations[i], NULL));        

        VkFenceCreateInfo fenceCreateInfo;
        memset(&fenceCreateInfo, 0, sizeof(VkFenceCreateInfo));
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CALL_RETNULL(vkCreateFence(ctx->devices[i], &fenceCreateInfo, NULL, &buffer->fences[i]));
    }

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    LOG_INFO("Destroying buffer with handle %p", buffer);

    for(int i = 0; i < buffer->ctx->deviceCount; i++) {
        vkDestroyFence(buffer->ctx->devices[i], buffer->fences[i], NULL);
        vmaDestroyBuffer(buffer->ctx->allocators[i], buffer->buffers[i], buffer->allocations[i]);
        vmaDestroyBuffer(buffer->ctx->allocators[i], buffer->stagingBuffers[i], buffer->stagingAllocations[i]);
    }

    delete buffer;
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    LOG_INFO("Writing data to buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    int enum_count = device_index == -1 ? buffer->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        VK_CALL(vkQueueWaitIdle(buffer->ctx->streams[dev_index]->queue));

        void* mapped;
        VK_CALL(vmaMapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index], &mapped));
        memcpy(mapped, data, size);
        vmaUnmapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index]);  

        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;

        VkCommandBuffer cmdBuffer = buffer->ctx->streams[dev_index]->begin();
        if(__error_string != NULL)
            return;
        
        vkCmdCopyBuffer(cmdBuffer, buffer->stagingBuffers[dev_index], buffer->buffers[dev_index], 1, &bufferCopy);
        
        VkFence fence = buffer->ctx->streams[dev_index]->submit();
        if(__error_string != NULL)
            return;
        VK_CALL(vkWaitForFences(buffer->ctx->devices[dev_index], 1, &fence, VK_TRUE, UINT64_MAX));
    }
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    LOG_INFO("Reading data from buffer (%p) at offset %d with size %d", buffer, offset, size);

    struct Context* ctx = buffer->ctx;

    int dev_index = device_index == -1 ? 0 : device_index;

    VkBufferCopy bufferCopy;
	bufferCopy.size = size;
	bufferCopy.dstOffset = 0;
	bufferCopy.srcOffset = offset;
	
    VK_CALL(vkQueueWaitIdle(buffer->ctx->streams[dev_index]->queue));

    VkCommandBuffer cmdBuffer = buffer->ctx->streams[dev_index]->begin();
    if(__error_string != NULL)
        return;
    
    vkCmdCopyBuffer(cmdBuffer, buffer->buffers[dev_index], buffer->stagingBuffers[dev_index], 1, &bufferCopy);
    
    VkFence fence = buffer->ctx->streams[dev_index]->submit();
    if(__error_string != NULL)
        return;
    VK_CALL(vkWaitForFences(buffer->ctx->devices[dev_index], 1, &fence, VK_TRUE, UINT64_MAX));

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index], &mapped));
    memcpy(data, mapped, size);
    vmaUnmapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index]);
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