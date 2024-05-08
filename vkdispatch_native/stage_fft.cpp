#include "internal.h"

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size) {
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;

    for (int i = 0; i < ctx->deviceCount; i++) {
        plan->launchParams.push_back({});
        plan->configs.push_back({});
        plan->apps.push_back({});
        plan->datas.push_back({});

        plan->datas[i].physicalDevice = ctx->physicalDevices[i];
        plan->datas[i].device = ctx->devices[i];
        plan->datas[i].queue = ctx->streams[i]->queue;
        plan->datas[i].commandPool = ctx->streams[i]->commandPool;
        plan->datas[i].fence = ctx->devices[i].createFence(vk::FenceCreateInfo());
        plan->datas[i].bufferSize = buffer_size;

        plan->configs[i].FFTdim = dims;
        plan->configs[i].size[0] = rows;
        plan->configs[i].size[1] = cols;
        plan->configs[i].size[2] = depth;
        plan->configs[i].physicalDevice = &plan->datas[i].physicalDevice;
        plan->configs[i].device = &plan->datas[i].device;
        plan->configs[i].queue = &plan->datas[i].queue;
        plan->configs[i].commandPool = &plan->datas[i].commandPool;
        plan->configs[i].fence = &plan->datas[i].fence;
        plan->configs[i].bufferSize = &plan->datas[i].bufferSize;
        plan->configs[i].isCompilerInitialized = true;
        
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
        [](vk::CommandBuffer& cmd_buffer, struct Stage* stage, void* instance_data, int device) {
            LOG_INFO("Executing FFT");

            struct FFTRecordInfo* my_fft_info = (struct FFTRecordInfo*)stage->user_data;

            VkBuffer temp_buf = my_fft_info->buffer->buffers[device];
            VkCommandBuffer temp_cmd = static_cast<VkCommandBuffer>(cmd_buffer);

            my_fft_info->plan->launchParams[device].buffer = &temp_buf;
            my_fft_info->plan->launchParams[device].commandBuffer = &temp_cmd;

            VkFFTResult fftRes = VkFFTAppend(&my_fft_info->plan->apps[device], my_fft_info->inverse, &my_fft_info->plan->launchParams[device]);
            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("Failed to append VkFFT %d", fftRes);
            }
        },
        my_fft_info,
        0,
        vk::PipelineStageFlagBits::eComputeShader
    });
}