#include "internal.h"

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size, unsigned int do_r2c) {
    LOG_INFO("Creating FFT plan with handle %p", ctx);
    
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;

    plan->apps = new VkFFTApplication[ctx->stream_indicies.size()];
    plan->configs = new VkFFTConfiguration[ctx->stream_indicies.size()];
    plan->launchParams = new VkFFTLaunchParams[ctx->stream_indicies.size()];
    plan->fences = new VkFence[ctx->stream_indicies.size()];

    for (int i = 0; i < ctx->stream_indicies.size(); i++) {
        plan->launchParams[i] = {};
        plan->configs[i] = {};
        plan->apps[i] = {};

        plan->configs[i].FFTdim = dims;
        plan->configs[i].size[0] = rows;
        plan->configs[i].size[1] = cols;
        plan->configs[i].size[2] = depth;

        int device_index = ctx->stream_indicies[i].first;
        int stream_index = ctx->stream_indicies[i].second;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CALL_RETNULL(vkCreateFence(ctx->devices[device_index], &fenceInfo, nullptr, &plan->fences[i]));

        plan->configs[i].physicalDevice = &ctx->physicalDevices[device_index];
        plan->configs[i].device = &ctx->devices[device_index];
        plan->configs[i].queue = &ctx->streams[device_index][stream_index]->queue;
        plan->configs[i].commandPool = &ctx->streams[device_index][stream_index]->commandPool;
        plan->configs[i].fence = &plan->fences[i];
        plan->configs[i].isCompilerInitialized = true;
        plan->configs[i].bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
        *plan->configs[i].bufferSize = buffer_size;
        plan->configs[i].performR2C = do_r2c;

        VkFFTResult resFFT = initializeVkFFT(&plan->apps[stream_index], plan->configs[stream_index]);
        if (resFFT != VKFFT_SUCCESS) {
            set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
        }
    }

    return plan;
}

void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse) {
    LOG_INFO("Recording FFT");

    command_list_append_command(command_list, [plan, buffer, inverse] (VkDevice device, VkCommandBuffer cmd_buffer, int stream_index) {
        plan->launchParams[stream_index].buffer = &buffer->buffers[stream_index];
        plan->launchParams[stream_index].commandBuffer = &cmd_buffer;

        VkFFTResult fftRes = VkFFTAppend(&plan->apps[stream_index], inverse, &plan->launchParams[stream_index]);
        if (fftRes != VKFFT_SUCCESS) {
            set_error("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
        }
    });

}