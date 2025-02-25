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

struct FFTPlan* stage_fft_plan_create_extern(
    struct Context* ctx, 
    unsigned long long dims, 
    unsigned long long rows, 
    unsigned long long cols, 
    unsigned long long depth, 
    unsigned long long buffer_size, 
    unsigned int do_r2c,
    int omit_rows,
    int omit_cols,
    int omit_depth) {
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
        //plan->configs[i] = {};
        plan->apps[i] = {};
    }

    command_list_record_command(ctx->command_list, 
        "fft-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, plan, recorder_count, dims, rows, cols, depth, do_r2c, omit_rows, omit_cols, omit_depth]
        (VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            LOG_VERBOSE("Initializing FFT on device %d, stream %d, recorder %d", device_index, stream_index, recorder_index);

            for(int j = 0; j < recorder_count; j++) {
                LOG_VERBOSE("Initializing FFT for recorder %d", j);

                int app_index = stream_index * recorder_count + j;

                VkFFTConfiguration config = {};

                config.FFTdim = dims;
                config.size[0] = rows;
                config.size[1] = cols;
                config.size[2] = depth;

                config.omitDimension[0] = omit_rows;
                config.omitDimension[1] = omit_cols;
                config.omitDimension[2] = omit_depth;

                LOG_VERBOSE("FFT Configuration: %d, %d, %d, %d, %d, %d, %d", config.FFTdim, config.size[0], config.size[1], config.size[2], config.omitDimension[0], config.omitDimension[1], config.omitDimension[2]);

                unsigned long long true_rows = rows;

                if(do_r2c) {
                    true_rows = (rows / 2) + 1;
                }

                config.physicalDevice = &ctx->physicalDevices[device_index];
                config.device = &ctx->devices[device_index];
                config.queue = &ctx->streams[stream_index]->queue;
                config.commandPool = &ctx->streams[stream_index]->commandPools[j];
                config.fence = &plan->fences[app_index];
                config.isCompilerInitialized = true;
                config.bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
                *config.bufferSize = true_rows * cols * depth * sizeof(float) * 2;
                config.performR2C = do_r2c;

                LOG_VERBOSE("FFT Configuration: %p, %p, %p, %p, %p, %p, %p", config.physicalDevice, config.device, config.queue, config.commandPool, config.fence, config.bufferSize, config.performR2C);

                plan->ctx->glslang_mutex.lock();

                LOG_VERBOSE("Initializing VkFFT");

                VkFFTResult resFFT = initializeVkFFT(&plan->apps[app_index], config);
                if (resFFT != VKFFT_SUCCESS) {
                    set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
                }

                LOG_VERBOSE("VkFFT Initialized");

                plan->ctx->glslang_mutex.unlock();
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

            VkFFTResult fftRes = VkFFTAppend(&plan->apps[index], inverse, &plan->launchParams[index]);

            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
            }
        }
    );
}