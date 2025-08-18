#include "stage_fft.hh"
#include "stages_extern.hh"

#include "../context/context.hh"
#include "../objects/buffer.hh"
#include "../objects/command_list.hh"
#include "../objects/objects_extern.hh"

#include <vkFFT.h>

struct FFTPlan {
    struct Context* ctx;
    uint64_t fences_handle;
    uint64_t vkfft_applications_handle;
    int recorder_count;
    uint64_t input_size;
};

void print_vkfft_config(VkFFTConfiguration* config) {
     LOG_INFO(R"(
 VkConfig:
     Size: (%d, %d, %d)
     Omit Dimention: (%d, %d, %d)
     Input Buffer Size: %d
     Is Input Formatted: %d
     Frequency Zero Padding: %d
     Kernel Convolution: %d
     Perform Convolution: %d
     Coordinate Features: %d
     Number Kernels: %d
     Kernel Size: %d
     Normalize: %d
     Buffer Size: %d
     Perform R2C: %d
     Number Batches: %d
     )", 
     config->size[0], config->size[1], config->size[2],
     config->omitDimension[0], config->omitDimension[1], config->omitDimension[2],
     *config->inputBufferSize,
     config->isInputFormatted,
     config->frequencyZeroPadding,
     config->kernelConvolution,
     config->performConvolution,
     config->coordinateFeatures,
     config->numberKernels,
     *config->kernelSize,
     config->normalize,
     *config->bufferSize,
     config->performR2C,
     config->numberBatches);
     //config->singleKernelMultipleBatches);
 }

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
    int normalize,
    unsigned long long pad_left_rows, unsigned long long pad_right_rows,
    unsigned long long pad_left_cols, unsigned long long pad_right_cols,
    unsigned long long pad_left_depth, unsigned long long pad_right_depth,
    int frequency_zeropadding,
    int kernel_num,
    int kernel_convolution,
    int conjugate_convolution,
    int convolution_features,
    unsigned long long input_buffer_size,
    int num_batches,
    int single_kernel_multiple_batches,
    int keep_shader_code) {
    LOG_INFO("Creating FFT plan with handle %p", ctx);
    
    struct FFTPlan* plan = new struct FFTPlan();
    plan->ctx = ctx;
    int recorder_count = ctx->queues[0]->recording_thread_count;

    int plan_count = ctx->queues.size() * recorder_count;
    uint64_t fences_handle = ctx->handle_manager->register_handle("Fences", plan_count, false);
    uint64_t vkfft_applications_handle = ctx->handle_manager->register_handle("VkFFTApplications", plan_count, false);

    plan->input_size = input_buffer_size;

    plan->recorder_count = recorder_count;
    plan->fences_handle = fences_handle;
    plan->vkfft_applications_handle = vkfft_applications_handle;

    command_list_record_command(ctx->command_list, 
        "fft-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, recorder_count, 
        fences_handle, vkfft_applications_handle, 
        dims, rows, cols, depth, do_r2c, 
        omit_rows, omit_cols, omit_depth, normalize,
        pad_left_rows, pad_right_rows,
        pad_left_cols, pad_right_cols,
        pad_left_depth, pad_right_depth,
        frequency_zeropadding,
        kernel_num,
        kernel_convolution,
        conjugate_convolution,
        convolution_features,
        input_buffer_size,
        num_batches,
        single_kernel_multiple_batches,
        keep_shader_code]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            LOG_VERBOSE("Initializing FFT on device %d, queue %d, recorder %d", indicies.device_index, indicies.queue_index, indicies.recorder_index);

            VkFFTConfiguration config = {};

            config.FFTdim = dims;
            config.size[0] = rows;
            config.size[1] = cols;
            config.size[2] = depth;

            config.disableSetLocale = 1;

            config.omitDimension[0] = omit_rows;
            config.omitDimension[1] = omit_cols;
            config.omitDimension[2] = omit_depth;

            config.performZeropadding[0] = pad_right_rows != 0;
            config.performZeropadding[1] = pad_right_cols != 0;
            config.performZeropadding[2] = pad_right_depth != 0;

            config.fft_zeropad_left[0] = pad_left_rows;
            config.fft_zeropad_left[1] = pad_left_cols;
            config.fft_zeropad_left[2] = pad_left_depth;

            config.fft_zeropad_right[0] = pad_right_rows;
            config.fft_zeropad_right[1] = pad_right_cols;
            config.fft_zeropad_right[2] = pad_right_depth;

            config.keepShaderCode = keep_shader_code;

            config.inputBufferSize = (uint64_t*)malloc(sizeof(uint64_t));
            *config.inputBufferSize = input_buffer_size;
            config.isInputFormatted = input_buffer_size > 0;

            config.frequencyZeroPadding = frequency_zeropadding;

            unsigned long long true_rows = rows;

            if(do_r2c) {
                true_rows = (rows / 2) + 1;
            }

            config.kernelConvolution = kernel_convolution;

            config.performConvolution = kernel_num > 0;
            config.conjugateConvolution = conjugate_convolution;
            config.coordinateFeatures = convolution_features;
            config.numberKernels = kernel_num;
            config.kernelSize = (uint64_t*)malloc(sizeof(uint64_t));
            *config.kernelSize = 2 * sizeof(float) * kernel_num * convolution_features * true_rows * config.size[1] * config.size[2];

            //config.singleKernelMultipleBatches = single_kernel_multiple_batches;

            glslang_resource_t* resource = reinterpret_cast<glslang_resource_t*>(ctx->glslang_resource_limits);

            config.maxComputeWorkGroupCount[0] = resource->max_compute_work_group_count_x;
            config.maxComputeWorkGroupCount[1] = resource->max_compute_work_group_count_y;
            config.maxComputeWorkGroupCount[2] = resource->max_compute_work_group_count_z;

            config.maxComputeWorkGroupSize[0] = resource->max_compute_work_group_size_x;
            config.maxComputeWorkGroupSize[1] = resource->max_compute_work_group_size_y;
            config.maxComputeWorkGroupSize[2] = resource->max_compute_work_group_size_z;

            config.normalize = normalize;

            int convolution_multiplier = 1;

            if(kernel_num > 0) {
                convolution_multiplier = kernel_num * convolution_features;
            }         
            
            config.bufferSize = (uint64_t*)malloc(sizeof(uint64_t));
            *config.bufferSize = num_batches * convolution_multiplier * true_rows * cols * depth * sizeof(float) * 2;
            config.performR2C = do_r2c;

            config.numberBatches = num_batches;

            config.isCompilerInitialized = true;
            config.glslang_mutex = &ctx->glslang_mutex;
            config.queue_mutex = &ctx->queues[indicies.queue_index]->queue_usage_mutex;
            config.physicalDevice = &ctx->physicalDevices[indicies.device_index];
            config.device = &ctx->devices[indicies.device_index];
            config.queue = &ctx->queues[indicies.queue_index]->queue;

            print_vkfft_config(&config);
            
            for(int j = 0; j < recorder_count; j++) {
                LOG_VERBOSE("Initializing FFT for recorder %d", j);

                int app_index = indicies.queue_index * recorder_count + j;

                VkFence* fence = (VkFence*)ctx->handle_manager->get_handle_pointer(app_index, fences_handle, 0);

                VkFenceCreateInfo fenceInfo = {};
                fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                fenceInfo.pNext = NULL;
                fenceInfo.flags = 0;
                VK_CALL(vkCreateFence(ctx->devices[indicies.device_index], &fenceInfo, NULL, fence));

                config.commandPool = &ctx->queues[indicies.queue_index]->commandPools[j];
                config.fence = fence;

                LOG_VERBOSE("Doing FFT Init");

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
    command_list_submit_extern(ctx->command_list, NULL, 1, submit_index, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(NULL)

    return plan;
}

void stage_fft_plan_destroy_extern(FFTPlan* plan) {
    LOG_WARNING("Destroying FFT plan with handle %p", plan);
    
}

void stage_fft_record_extern(
    struct CommandList* command_list, 
    struct FFTPlan* plan, 
    struct Buffer* buffer, int inverse, 
    struct Buffer* kernel,
    struct Buffer* input_buffer) {
    LOG_VERBOSE("Recording FFT");

    struct Context* ctx = plan->ctx;

    int recorder_count = plan->recorder_count;
    uint64_t vkfft_applications_handle = plan->vkfft_applications_handle;

    uint64_t buffer_handle = buffer->buffers_handle;
    uint64_t kernel_handle = kernel ? kernel->buffers_handle : 0;
    uint64_t input_buffer_handle = input_buffer ? input_buffer->buffers_handle : 0;

    command_list_record_command(command_list, 
        "fft-exec",
        0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        [ctx, plan, recorder_count, vkfft_applications_handle, inverse, buffer_handle, kernel_handle, input_buffer_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            int index = indicies.queue_index * recorder_count + indicies.recorder_index;

            VkFFTLaunchParams launchParams = {};
            launchParams.buffer = (VkBuffer*)ctx->handle_manager->get_handle_pointer(indicies.queue_index, buffer_handle, timestamp);
            launchParams.commandBuffer = &cmd_buffer;

            if(kernel_handle != 0) {
                launchParams.kernel = (VkBuffer*)ctx->handle_manager->get_handle_pointer(indicies.queue_index, kernel_handle, timestamp);
            }

            if(input_buffer_handle != 0) {
                launchParams.inputBuffer = (VkBuffer*)ctx->handle_manager->get_handle_pointer(indicies.queue_index, input_buffer_handle, timestamp);
            }

            VkFFTApplication* application = (VkFFTApplication*)ctx->handle_manager->get_handle(index, vkfft_applications_handle, timestamp);

            VkFFTResult fftRes = VkFFTAppend(application, inverse, &launchParams);

            if (fftRes != VKFFT_SUCCESS) {
                LOG_ERROR("(VkFFTResult is %d) VkFFTAppend inside '%s' at %s:%d\n", fftRes, __FUNCTION__, __FILE__, __LINE__);
            }
        }
    );
}

const char* stage_fft_axis_code(struct FFTPlan* plan, int axis) {
    return NULL;
}