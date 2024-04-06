#include "internal.h"

struct DeviceContext* create_device_context_extern(int* device_indicies, int* submission_thread_couts, int device_count) {
    printf("Creating device context with %d devices\n", device_count);

    struct DeviceContext* ctx = new struct DeviceContext();
    ctx->deviceCount = device_count;
    ctx->devices = new VKLDevice[device_count];
    ctx->queues = (const VKLQueue**)malloc(sizeof(VKLQueue*) * device_count);
    ctx->submissionThreadCounts = (uint32_t*)malloc(sizeof(uint32_t) * device_count);

    for (int i = 0; i < device_count; i++) {
        ctx->devices[i].create(
            VKLDeviceCreateInfo()
            .physicalDevice(&_ctx.instance.getPhysicalDevices()[device_indicies[i]])
            .queueTypeCount(VKL_QUEUE_TYPE_ALL, 1)
        );

        ctx->queues[i] = ctx->devices[i].getQueue(VKL_QUEUE_TYPE_ALL, 0);
        ctx->submissionThreadCounts[i] = submission_thread_couts[i];
    }

    return ctx;
}

void destroy_device_context_extern(struct DeviceContext* device_context) {
    for (int i = 0; i < device_context->deviceCount; i++) {
        device_context->devices[i].destroy();
    }

    delete[] device_context->devices;
    free((void*)device_context->queues);
    free((void*)device_context->submissionThreadCounts);
    delete device_context;
}