#include "../include/internal.hh"

#include <vkFFT.h>

struct FFTPlan {
    struct Context* ctx;
    VkFence* fences;
    VkFFTApplication* apps;
    VkFFTConfiguration* configs;
    VkFFTLaunchParams* launchParams;
    int recorder_count;
};

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size, unsigned int do_r2c) {
    LOG_INFO("Creating FFT plan with handle %p", ctx);
    
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;

    int plan_count = ctx->stream_indicies.size() * ctx->streams[0][0]->recording_thread_count;

    int recorder_count = ctx->streams[0][0]->recording_thread_count;

    plan->apps = new VkFFTApplication[plan_count];
    plan->configs = new VkFFTConfiguration[plan_count];
    plan->launchParams = new VkFFTLaunchParams[plan_count];
    plan->fences = new VkFence[plan_count];
    plan->recorder_count = recorder_count;

    

    //Signal* signals = new Signal[plan_count];

    for (int i = 0; i < ctx->stream_indicies.size(); i++) {
        struct CommandInfo command = {};
        command.type = COMMAND_TYPE_NOOP;
        command.pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        command_list_record_command(ctx->command_list, command);
        
        Signal signal;
        command_list_submit_extern(ctx->command_list, NULL, 1, &i, 1, &signal); //buffer->per_device, &signal);
        command_list_reset_extern(ctx->command_list);
        RETURN_ON_ERROR(NULL)

        signal.wait();

        for(int j = 0; j < recorder_count; j++) {
            int recorder_index = i * recorder_count + j;

            plan->launchParams[recorder_index] = {};
            plan->configs[recorder_index] = {};
            plan->apps[recorder_index] = {};

            plan->configs[recorder_index].FFTdim = dims;
            plan->configs[recorder_index].size[0] = rows;
            plan->configs[recorder_index].size[1] = cols;
            plan->configs[recorder_index].size[2] = depth;

            int device_index = ctx->stream_indicies[i].first;
            int stream_index = ctx->stream_indicies[i].second;

            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VK_CALL_RETNULL(vkCreateFence(ctx->devices[device_index], &fenceInfo, nullptr, &plan->fences[recorder_index]));

            plan->configs[recorder_index].physicalDevice = &ctx->physicalDevices[device_index];
            plan->configs[recorder_index].device = &ctx->devices[device_index];
            plan->configs[recorder_index].queue = &ctx->streams[device_index][stream_index]->queue;
            plan->configs[recorder_index].commandPool = &ctx->streams[device_index][stream_index]->commandPools[j];
            plan->configs[recorder_index].fence = &plan->fences[recorder_index];
            plan->configs[recorder_index].isCompilerInitialized = true;
            plan->configs[recorder_index].bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
            *plan->configs[recorder_index].bufferSize = buffer_size;
            plan->configs[recorder_index].performR2C = do_r2c;

            VkFFTResult resFFT = initializeVkFFT(&plan->apps[recorder_index], plan->configs[recorder_index]);
            if (resFFT != VKFFT_SUCCESS) {
                set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
            }

        }
    
        //struct CommandInfo command = {};
        //command.type = COMMAND_TYPE_FFT_INIT;
        //command.pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        //command.info.fft_init_info.plan = plan;

        //command_list_record_command(ctx->command_list, command);
        //command_list_submit_extern(ctx->command_list, NULL, 1, &i, 1, 0, &signals[i]);
        //command_list_reset_extern(ctx->command_list);
        //RETURN_ON_ERROR(NULL);
    }

    //for(int i = 0; i < ctx->stream_indicies.size(); i++) {
    //    signals[i].wait();
    //}

    //delete[] signals;

    return plan;
}

void stage_fft_plan_init_internal(const struct FFTInitRecordInfo& info, int device_index, int stream_index, int recorder_index) {
    int index = stream_index * info.plan->recorder_count + recorder_index;

    VkFFTResult resFFT = initializeVkFFT(&info.plan->apps[index], info.plan->configs[index]);
    if (resFFT != VKFFT_SUCCESS) {
        set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
    }
}

void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse) {
    //LOG_INFO("Recording FFT");

    //if (buffer->per_device) {
    //    set_error("FFT cannot be performed on per-device buffer!");
    //    return;
    //}

    struct CommandInfo command = {};
    command.type = COMMAND_TYPE_FFT_EXEC;
    command.pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    command.info.fft_exec_info.plan = plan;
    command.info.fft_exec_info.buffer = buffer;
    command.info.fft_exec_info.inverse = inverse;

    command_list_record_command(command_list, command);
}

void stage_fft_plan_exec_internal(VkCommandBuffer cmd_buffer, const struct FFTExecRecordInfo& info, int device_index, int stream_index, int recorder_index) {
    int index = stream_index * info.plan->recorder_count + recorder_index;

    info.plan->launchParams[index].buffer = &info.buffer->buffers[stream_index];
    info.plan->launchParams[index].commandBuffer = &cmd_buffer;

    VkFFTResult fftRes = VkFFTAppend(&info.plan->apps[index], info.inverse, &info.plan->launchParams[index]);

    if (fftRes != VKFFT_SUCCESS) {
        LOG_ERROR("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
    }
}