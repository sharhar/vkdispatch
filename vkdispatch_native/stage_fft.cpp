#include "internal.h"

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size) {
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;

    plan->apps = new VkFFTApplication[ctx->deviceCount];
    plan->configs = new VkFFTConfiguration[ctx->deviceCount];
    plan->launchParams = new VkFFTLaunchParams[ctx->deviceCount];
    plan->fences = new VkFence[ctx->deviceCount];

    for (int i = 0; i < ctx->deviceCount; i++) {
        plan->launchParams[i] = {};
        plan->configs[i] = {};
        plan->apps[i] = {};

        plan->configs[i].FFTdim = dims;
        plan->configs[i].size[0] = rows;
        plan->configs[i].size[1] = cols;
        plan->configs[i].size[2] = depth;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VK_CALL(vkCreateFence(ctx->devices[i]->handle(), &fenceInfo, nullptr, &plan->fences[i]));

        plan->configs[i].physicalDevice = ctx->devices[i]->physical()->pHandle();
        plan->configs[i].device = ctx->devices[i]->pHandle();
        plan->configs[i].queue = &ctx->streams[i]->queue;
        plan->configs[i].commandPool = &ctx->streams[i]->commandPool;
        plan->configs[i].fence = &plan->fences[i];
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

    command_list->stages.push_back({
        [](VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device) {
            LOG_INFO("Executing FFT");

            struct FFTRecordInfo* my_fft_info = (struct FFTRecordInfo*)stage->user_data;

            my_fft_info->plan->launchParams[device].buffer = &my_fft_info->buffer->buffers[device];
            my_fft_info->plan->launchParams[device].commandBuffer = &cmd_buffer;

            VkFFTResult fftRes = VkFFTAppend(&my_fft_info->plan->apps[device], my_fft_info->inverse, &my_fft_info->plan->launchParams[device]);
            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("Failed to append VkFFT %d", fftRes);
            }
        },
        my_fft_info,
        0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    });
}