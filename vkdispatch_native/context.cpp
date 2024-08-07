#include "internal.h"
#include <vector>

struct Context* context_create_extern(int* device_indicies, int* submission_thread_couts, int device_count) {
    struct Context* ctx = new struct Context();
    ctx->deviceCount = device_count;
    ctx->devices = (VKLDevice**)malloc(sizeof(VKLDevice*) * device_count);
    ctx->queues = (const VKLQueue**)malloc(sizeof(VKLQueue*) * device_count);
    ctx->commandBuffers = (VKLCommandBuffer**)malloc(sizeof(VKLCommandBuffer*) * device_count);
    ctx->fences = (VkFence*)malloc(sizeof(VkFence) * device_count);
    ctx->submissionThreadCounts = (uint32_t*)malloc(sizeof(uint32_t) * device_count);

    const std::vector<VKLPhysicalDevice*>&  phyisicalDevices = _instance.instance.getPhysicalDevices();

    for (int i = 0; i < device_count; i++) {
        ctx->devices[i] = new VKLDevice(
            VKLDeviceCreateInfo()
            .physicalDevice(phyisicalDevices[device_indicies[i]])
            .queueTypeCount(VKL_QUEUE_TYPE_ALL, 1)
        );

        ctx->queues[i] = ctx->devices[i]->getQueue(VKL_QUEUE_TYPE_ALL, 0);
        ctx->commandBuffers[i] = new VKLCommandBuffer(ctx->queues[i]);
        ctx->fences[i] = ctx->devices[i]->createFence(VK_FENCE_CREATE_SIGNALED_BIT);
        ctx->submissionThreadCounts[i] = submission_thread_couts[i];
    }

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    return ctx;
}

void context_destroy_extern(struct Context* context) {
    for (int i = 0; i < context->deviceCount; i++) {
        context->devices[i]->destroy();
        delete context->devices[i];
    }

    free((void*)context->devices);
    free((void*)context->queues);
    free((void*)context->submissionThreadCounts);
    delete context;
}