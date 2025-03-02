#include "../include/internal.hh"

#include <vkFFT.h>

struct FFTPlan {
    struct Context* ctx;
    uint64_t fences_handle;
    uint64_t vkfft_applications_handle;
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
    int omit_depth,
    int normalize) {
    LOG_INFO("Creating FFT plan with handle %p", ctx);
    
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;
    int recorder_count = ctx->streams[0]->recording_thread_count;

    int plan_count = ctx->streams.size() * recorder_count;
    uint64_t fences_handle = ctx->handle_manager->register_handle("Fences", plan_count, false);
    uint64_t vkfft_applications_handle = ctx->handle_manager->register_handle("VkFFTApplications", plan_count, false);

    plan->recorder_count = recorder_count;
    plan->fences_handle = fences_handle;
    plan->vkfft_applications_handle = vkfft_applications_handle;

    command_list_record_command(ctx->command_list, 
        "fft-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, recorder_count, fences_handle, vkfft_applications_handle, dims, rows, cols, depth, do_r2c, omit_rows, omit_cols, omit_depth, normalize]
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

                glslang_resource_t* resource = reinterpret_cast<glslang_resource_t*>(ctx->glslang_resource_limits);

                config.maxComputeWorkGroupCount[0] = resource->max_compute_work_group_count_x;
                config.maxComputeWorkGroupCount[1] = resource->max_compute_work_group_count_y;
                config.maxComputeWorkGroupCount[2] = resource->max_compute_work_group_count_z;

                config.maxComputeWorkGroupSize[0] = resource->max_compute_work_group_size_x;
                config.maxComputeWorkGroupSize[1] = resource->max_compute_work_group_size_y;
                config.maxComputeWorkGroupSize[2] = resource->max_compute_work_group_size_z;

                config.normalize = normalize;

                LOG_VERBOSE("FFT Configuration: %d, %d, %d, %d, %d, %d, %d", config.FFTdim, config.size[0], config.size[1], config.size[2], config.omitDimension[0], config.omitDimension[1], config.omitDimension[2]);

                unsigned long long true_rows = rows;

                if(do_r2c) {
                    true_rows = (rows / 2) + 1;
                }

                VkFence* fence = (VkFence*)ctx->handle_manager->get_handle_pointer(stream_index, fences_handle);

                VkFenceCreateInfo fenceInfo = {};
                fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                fenceInfo.pNext = NULL;
                fenceInfo.flags = 0;
                VK_CALL(vkCreateFence(ctx->devices[device_index], &fenceInfo, NULL, fence));

                config.physicalDevice = &ctx->physicalDevices[device_index];
                config.device = &ctx->devices[device_index];
                config.queue = &ctx->streams[stream_index]->queue;
                config.commandPool = &ctx->streams[stream_index]->commandPools[j];
                config.fence = fence;
                config.isCompilerInitialized = true;
                config.bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
                *config.bufferSize = true_rows * cols * depth * sizeof(float) * 2;
                config.performR2C = do_r2c;
                config.glslang_mutex = &ctx->glslang_mutex;

                LOG_VERBOSE("FFT Configuration: %p, %p, %p, %p, %p, %p, %p", config.physicalDevice, config.device, config.queue, config.commandPool, config.fence, config.bufferSize, config.performR2C);

                VkFFTApplication* application = new VkFFTApplication();

                VkFFTResult resFFT = initializeVkFFT(application, config);
                if (resFFT != VKFFT_SUCCESS) {
                    set_error("(VkFFTResult is %d) initializeVkFFT inside '%s' at %s:%d\n", resFFT, __FUNCTION__, __FILE__, __LINE__);
                }

                ctx->handle_manager->set_handle(app_index, vkfft_applications_handle, (uint64_t)application);
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

    struct Context* ctx = plan->ctx;

    int recorder_count = plan->recorder_count;
    uint64_t vkfft_applications_handle = plan->vkfft_applications_handle;

    command_list_record_command(command_list, 
        "fft-exec",
        0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        [ctx, recorder_count, vkfft_applications_handle, buffer, inverse](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            int index = stream_index * recorder_count + recorder_index;

            VkFFTLaunchParams launchParams = {};
            launchParams.buffer = &buffer->buffers[stream_index];
            launchParams.commandBuffer = &cmd_buffer;

            VkFFTApplication* application = (VkFFTApplication*)ctx->handle_manager->get_handle(stream_index, vkfft_applications_handle);

            VkFFTResult fftRes = VkFFTAppend(application, inverse, &launchParams);

            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
            }
        }
    );
}