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

    int plan_count = ctx->streams.size() * ctx->streams[0]->recording_thread_count;

    int recorder_count = ctx->streams[0]->recording_thread_count;

    plan->apps = new VkFFTApplication[plan_count];
    plan->configs = new VkFFTConfiguration[plan_count];
    plan->launchParams = new VkFFTLaunchParams[plan_count];
    plan->fences = new VkFence[plan_count];
    plan->recorder_count = recorder_count;

    for(int i = 0; i < plan_count; i++) {
        plan->launchParams[i] = {};
        plan->configs[i] = {};
        plan->apps[i] = {};
    }

    command_list_record_command(ctx->command_list, 
        "fft-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, plan, recorder_count, dims, rows, cols, depth, do_r2c](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            for(int j = 0; j < recorder_count; j++) {
                int app_index = stream_index * recorder_count + j;

                //VkFFTConfiguration config = {};

                plan->configs[stream_index].FFTdim = dims;
                plan->configs[stream_index].size[0] = rows;
                plan->configs[stream_index].size[1] = cols;
                plan->configs[stream_index].size[2] = depth;

                plan->configs[stream_index].physicalDevice = &ctx->physicalDevices[device_index];
                plan->configs[stream_index].device = &ctx->devices[device_index];
                plan->configs[stream_index].queue = &ctx->streams[stream_index]->queue;
                plan->configs[stream_index].commandPool = &ctx->streams[stream_index]->commandPools[j];
                plan->configs[stream_index].fence = &plan->fences[app_index];
                plan->configs[stream_index].isCompilerInitialized = true;
                plan->configs[stream_index].bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
                *plan->configs[stream_index].bufferSize = rows * cols * depth * sizeof(float) * 2;//1024 * 1024;
                plan->configs[stream_index].performR2C = do_r2c;

                VkFFTResult resFFT = initializeVkFFT(&plan->apps[app_index], plan->configs[stream_index]);
                if (resFFT != VKFFT_SUCCESS) {
                    set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
                }

            }
        }
    );

    int submit_index = -2;
    command_list_submit_extern(ctx->command_list, NULL, 1, &submit_index, 1, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(NULL)

    return plan;
}

void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse) {
    LOG_VERBOSE("Recording FFT");

    command_list_record_command(command_list, 
        "fft-exec",
        0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        [plan, buffer, inverse](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            int index = stream_index * plan->recorder_count + recorder_index;

            plan->launchParams[index].buffer = &buffer->buffers[stream_index];
            plan->launchParams[index].commandBuffer = &cmd_buffer;

            for(int i = 0; i < plan->configs[stream_index].FFTdim; i++) {
                LOG_INFO("%d: %d", i, plan->configs[stream_index].size[i]);
            }

            LOG_INFO("Buffer size: %d", *plan->configs[stream_index].bufferSize);

            LOG_INFO("Buffer object size: %d", buffer->size);

            LOG_INFO("Executing FFT with inverse %d", inverse);

            VkFFTResult fftRes = VkFFTAppend(&plan->apps[index], inverse, &plan->launchParams[index]);

            LOG_INFO("FFT executed");

            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
            }
        }
    );
}