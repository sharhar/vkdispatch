#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include <VKL/VKL.h>
#include <vkFFT.h>
#include <vector>

#include "base.h"
#include "init.h"
#include "context.h"
#include "buffer.h"
#include "image.h"
#include "stage_transfer.h"
#include "stage_fft.h"
#include "stage_compute.h"
#include "command_list.h"

typedef struct {
    VKLInstance instance;
    struct PhysicalDeviceProperties* devices;
} MyInstance;

extern MyInstance _instance;

struct Context {
    uint32_t deviceCount;
    VKLDevice** devices;
    const VKLQueue** queues;
    VKLCommandBuffer** commandBuffers;
    VkFence* fences;
    uint32_t* submissionThreadCounts;
};

struct Buffer {
    struct Context* ctx;
    VKLBuffer** buffers;
    VKLBuffer** stagingBuffers;
};

struct Image {
    struct Context* ctx;
    VKLImage** images;
    VKLImageView** imageViews;
    VKLBuffer** stagingBuffers;

    uint32_t block_size;
};

typedef void (*PFN_stage_record)(VKLCommandBuffer* cmd_buffer, struct Stage* stage, void* instance_data, int device);

struct Stage {
    PFN_stage_record record;
    void* user_data;
    size_t instance_data_size;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct Stage> stages;
};

struct FFTPlan {
    struct Context* ctx;
    VkFFTApplication* apps;
    VkFFTConfiguration* configs;
    VkFFTLaunchParams* launchParams;
};

struct ComputePlan {
    struct Context* ctx;
    VKLPipelineLayout** pipelineLayouts;
    VKLPipeline** pipelines;
    VKLDescriptorSet** descriptorSets;
    unsigned int pc_size;
};

#endif // INTERNAL_H