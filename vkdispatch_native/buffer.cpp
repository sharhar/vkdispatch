#include "internal.h"

struct Buffer* buffer_create_extern(struct Context* context, unsigned long long size) {
    struct Buffer* buffer = new struct Buffer();
    buffer->ctx = context;
    buffer->buffers = (VKLBuffer**)malloc(sizeof(VKLBuffer*) * context->deviceCount);
    buffer->stagingBuffers = (VKLBuffer**)malloc(sizeof(VKLBuffer*) * context->deviceCount);


    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    for (int i = 0; i < context->deviceCount; i++) {
        buffer->buffers[i] = new VKLBuffer(
            VKLBufferCreateInfo()
            .device(context->devices[i])
            .size(size)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
            .usage(usage)
        );

        buffer->stagingBuffers[i] = new VKLBuffer(
            VKLBufferCreateInfo()
            .device(context->devices[i])
            .size(size)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
		    .flagsVMA(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
            .usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        );
    }

    return buffer;
}

void buffer_destroy_extern(struct Buffer* buffer) {
    for (int i = 0; i < buffer->ctx->deviceCount; i++) {
        buffer->buffers[i]->destroy();
        buffer->stagingBuffers[i]->destroy();

        delete buffer->buffers[i];
        delete buffer->stagingBuffers[i];
    }

    free((void*)buffer->buffers);
    free((void*)buffer->stagingBuffers);
    delete buffer;
}

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    int enum_count = device_index == -1 ? buffer->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        buffer->stagingBuffers[dev_index]->setData(data, size, 0);

        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;

        buffer->buffers[dev_index]->copyFrom(buffer->stagingBuffers[dev_index], buffer->ctx->queues[dev_index], bufferCopy);
    }
}

void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index) {
    int dev_index = device_index == -1 ? 0 : device_index;

    VkBufferCopy bufferCopy;
	bufferCopy.size = size;
	bufferCopy.dstOffset = 0;
	bufferCopy.srcOffset = offset;
	
	buffer->stagingBuffers[dev_index]->copyFrom(buffer->buffers[dev_index], buffer->ctx->queues[dev_index], bufferCopy);	
	buffer->stagingBuffers[dev_index]->getData(data, size, 0);
}

void buffer_copy_extern(struct Buffer* src, struct Buffer* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index) {
    assert(src->ctx == dst->ctx);

    int enum_count = device_index == -1 ? src->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        VkBufferCopy bufferCopy;
        bufferCopy.size = size;
        bufferCopy.dstOffset = dst_offset;
        bufferCopy.srcOffset = src_offset;

        dst->buffers[dev_index]->copyFrom(src->buffers[dev_index], src->ctx->queues[dev_index], bufferCopy);
    }
}