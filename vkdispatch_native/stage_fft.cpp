#include "internal.h"

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size) {
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;

    plan->apps = new VkFFTApplication[ctx->deviceCount];
    plan->configs = new VkFFTConfiguration[ctx->deviceCount];
    plan->launchParams = new VkFFTLaunchParams[ctx->deviceCount];

    for (int i = 0; i < ctx->deviceCount; i++) {
        plan->launchParams[i] = {};
        plan->configs[i] = {};
        plan->apps[i] = {};

        plan->configs[i].FFTdim = dims;
        plan->configs[i].size[0] = rows;
        plan->configs[i].size[1] = cols;
        plan->configs[i].size[2] = depth;

        plan->configs[i].makeForwardPlanOnly = true;
        plan->configs[i].physicalDevice = ctx->devices[i]->physical()->pHandle();
        plan->configs[i].device = ctx->devices[i]->pHandle();
        plan->configs[i].queue = ctx->queues[i]->pHandle();
        plan->configs[i].commandPool = ctx->commandBuffers[i]->pPool();
        plan->configs[i].fence = &ctx->fences[i];
        plan->configs[i].isCompilerInitialized = true;
        plan->configs[i].bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
        *plan->configs[i].bufferSize = buffer_size;
        
        VkFFTResult resFFT = initializeVkFFT(&plan->apps[i], plan->configs[i]);
        if (resFFT != VKFFT_SUCCESS) {
            LOG_ERROR("Failed to initialize VkFFT %d", resFFT);
            return NULL;
        }
    }

    return plan;
}

struct FFTRecordInfo {
    struct FFTPlan* plan;
    struct Buffer* buffer;
    int inverse;
};

void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse) {
    struct FFTRecordInfo* my_fft_info = (struct FFTRecordInfo*)malloc(sizeof(struct FFTRecordInfo));
    my_fft_info->plan = plan;
    my_fft_info->buffer = buffer;
    my_fft_info->inverse = inverse;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = command_list->ctx->commandBuffers[0]->pool();
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
 
    VkCommandBuffer commandBuffer = {};
	VkResult res =vkAllocateCommandBuffers(command_list->ctx->devices[0]->handle(), &commandBufferAllocateInfo, &commandBuffer);
    if (res != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate command buffer %d", res);
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    if (res != VK_SUCCESS) {
        LOG_ERROR("Failed to begin command buffer %d", res);
    }

    my_fft_info->plan->launchParams[0].buffer = my_fft_info->buffer->buffers[0]->pHandle();
    my_fft_info->plan->launchParams[0].commandBuffer = &commandBuffer;

    VkFFTResult fftRes = VkFFTAppend(&my_fft_info->plan->apps[0], my_fft_info->inverse, &my_fft_info->plan->launchParams[0]);
    if (fftRes != VKFFT_SUCCESS) {
        LOG_ERROR("Failed to append VkFFT %d", fftRes);
    }

    res = vkEndCommandBuffer(commandBuffer);
    if(res != VK_SUCCESS) {
        LOG_ERROR("Failed to end command buffer %d", res);
    }

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
    res = vkQueueSubmit(command_list->ctx->queues[0]->handle(), 1, &submitInfo, command_list->ctx->fences[0]);
    if(res != VK_SUCCESS) {
        LOG_ERROR("Failed to submit queue %d", res);
    }

	res = vkWaitForFences(command_list->ctx->devices[0]->handle(), 1, &command_list->ctx->fences[0], VK_TRUE, 100000000000);
	if (res != 0) {
        LOG_ERROR("Failed to wait for fence %d", res);
    }

	res = vkResetFences(command_list->ctx->devices[0]->handle(), 1, &command_list->ctx->fences[0]);
    if (res != 0) {
        LOG_ERROR("Failed to reset fence %d", res);
    }

    vkFreeCommandBuffers(command_list->ctx->devices[0]->handle(), command_list->ctx->commandBuffers[0]->pool(), 1, &commandBuffer);
}