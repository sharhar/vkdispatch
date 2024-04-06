#include "internal.h"

struct Buffer* create_buffer_extern(struct DeviceContext* device_context, unsigned long long size) {
    struct Buffer* buffer = new struct Buffer();
    buffer->ctx = device_context;
    buffer->buffers = new VKLBuffer[device_context->deviceCount];
    buffer->stagingBuffers = new VKLBuffer[device_context->deviceCount];

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    for (int i = 0; i < device_context->deviceCount; i++) {
        buffer->buffers[i].create(
            VKLBufferCreateInfo()
            .device(&device_context->devices[i])
            .size(size)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
            .usage(usage)
        );

        buffer->stagingBuffers[i].create(
            VKLBufferCreateInfo()
            .device(&device_context->devices[i])
            .size(size)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
		    .flagsVMA(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
            .usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        );
    }

    return buffer;
}

void destroy_buffer_extern(struct Buffer* buffer) {
    for (int i = 0; i < buffer->ctx->deviceCount; i++) {
        buffer->buffers[i].destroy();
        buffer->stagingBuffers[i].destroy();
    }

    delete[] buffer->buffers;
    delete[] buffer->stagingBuffers;
    delete buffer;
}