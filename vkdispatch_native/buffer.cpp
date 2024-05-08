#include "internal.h"

#include <iostream>

struct Buffer* buffer_create_extern(struct Context* context, unsigned long long size) {
    struct Context* ctx = (struct Context*)context;

    struct Buffer* buffer = new struct Buffer();
    buffer->ctx = context;

    for (int i = 0; i < context->deviceCount; i++) {
        vk::BufferCreateInfo bufferCreateInfo = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer);        
        
        VmaAllocation vmaAllocation;

        VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = 0;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		vmaAllocationCreateInfo.pUserData = &vmaAllocation;

        VkBufferCreateInfo bufferCreateInfoStruct = static_cast<VkBufferCreateInfo>(bufferCreateInfo);
        VkBuffer temp_buffer;
		VK_CALL(vmaCreateBuffer(ctx->allocators[i], &bufferCreateInfoStruct, &vmaAllocationCreateInfo, &temp_buffer, &vmaAllocation, NULL));

        LOG_INFO("Buffer created %p", temp_buffer);

        buffer->allocations.push_back(vmaAllocation);
        buffer->buffers.push_back(temp_buffer);

        vk::BufferCreateInfo stagingBufferCreateInfo = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);

        VmaAllocation vmaAllocationStaging;

        vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		vmaAllocationCreateInfo.pUserData = &vmaAllocationStaging;

        VkBufferCreateInfo stagingBufferCreateInfoStruct = static_cast<VkBufferCreateInfo>(stagingBufferCreateInfo);
        VkBuffer temp_staging_buffer;
        VK_CALL(vmaCreateBuffer(ctx->allocators[i], &stagingBufferCreateInfoStruct, &vmaAllocationCreateInfo, &temp_staging_buffer, &vmaAllocationStaging, NULL));

        LOG_INFO("Staging buffer created %p", temp_staging_buffer);

        buffer->stagingAllocations.push_back(vmaAllocationStaging);
        buffer->stagingBuffers.push_back(temp_staging_buffer);

        buffer->fences.push_back(ctx->devices[i].createFence(
            vk::FenceCreateInfo()
            .setFlags(vk::FenceCreateFlagBits::eSignaled)
        ));
    }

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    struct Context* ctx = (struct Context*)buffer->ctx;
    
    for (int i = 0; i < buffer->ctx->deviceCount; i++) {
        vmaDestroyBuffer(ctx->allocators[i], buffer->buffers[i], buffer->allocations[i]);
        vmaDestroyBuffer(ctx->allocators[i], buffer->stagingBuffers[i], buffer->stagingAllocations[i]);
    }

    delete buffer;
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    struct Context* ctx = (struct Context*)buffer->ctx;

    int enum_count = device_index == -1 ? buffer->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;
        
        LOG_INFO("Writing buffer data to device %d", dev_index);

        ctx->streams[dev_index]->queue.waitIdle();

        void* mapped;
        VK_CALL(vmaMapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index], &mapped));
        memcpy(mapped, data, size);
        vmaUnmapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index]);  

        vk::CommandBuffer cmd_buffer = ctx->streams[dev_index]->begin();

        cmd_buffer.copyBuffer(buffer->stagingBuffers[dev_index], buffer->buffers[dev_index], 
            vk::BufferCopy()
            .setSrcOffset(0)
            .setDstOffset(offset)
            .setSize(size)
        );

        vk::Fence& fence = ctx->streams[dev_index]->submit();
        ctx->devices[dev_index].waitForFences(fence, VK_TRUE, UINT64_MAX);
    }
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    struct Context* ctx = (struct Context*)buffer->ctx;

    LOG_INFO("Reading buffer data");

    int dev_index = device_index == -1 ? 0 : device_index;

    vk::CommandBuffer cmd_buffer = ctx->streams[dev_index]->begin();

    cmd_buffer.copyBuffer(buffer->buffers[dev_index], buffer->stagingBuffers[dev_index],
        vk::BufferCopy()
        .setSrcOffset(offset)
        .setDstOffset(0)
        .setSize(size)
    );

    vk::Fence& fence = ctx->streams[dev_index]->submit();
    ctx->devices[dev_index].waitForFences(fence, VK_TRUE, UINT64_MAX);

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index], &mapped));
    memcpy(data, mapped, size);
    vmaUnmapMemory(ctx->allocators[dev_index], buffer->stagingAllocations[dev_index]);

    LOG_INFO("Buffer data read");
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