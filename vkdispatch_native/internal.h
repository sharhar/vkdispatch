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
#include "descriptor_set.h"

typedef struct {
    VKLInstance instance;
    struct PhysicalDeviceProperties* devices;
} MyInstance;

extern MyInstance _instance;

class Stream {
public:
    Stream(VkDevice device, VkQueue queue, int queueFamilyIndex, uint32_t command_buffer_count);
    void destroy();

    VkCommandBuffer& begin();
    VkFence& submit();

    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkFence> fences;
    std::vector<VkSemaphore> semaphores;
    int current_index;
};

struct Context {
    uint32_t deviceCount;
    VKLDevice** devices;
    //const VKLQueue** queues;
    //VKLCommandBuffer** commandBuffers;
    //VkFence* fences;
    std::vector<Stream*> streams;
    uint32_t* submissionThreadCounts;
};

struct Buffer {
    struct Context* ctx;
    VKLBuffer** buffers;
    VKLBuffer** stagingBuffers;
};

struct Image {
    struct Context* ctx;
    
    //VKLImage** images;
    //VKLImageView** imageViews;
    //VKLBuffer** stagingBuffers;

    uint32_t block_size;
};

typedef void (*PFN_stage_record)(VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device);

struct Stage {
    PFN_stage_record record;
    void* user_data;
    size_t instance_data_size;
    VkPipelineStageFlags stage;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct Stage> stages;
};

struct FFTPlan {
    struct Context* ctx;
    VkFence* fences;
    VkFFTApplication* apps;
    VkFFTConfiguration* configs;
    VkFFTLaunchParams* launchParams;
};

struct ComputePlan {
    struct Context* ctx;
    std::vector<VkShaderModule> modules;
    std::vector<std::vector<VkDescriptorPoolSize>> poolSizes;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkPipelineLayout> pipelineLayouts;
    std::vector<VkPipeline> pipelines;
    unsigned int binding_count;
    unsigned int pc_size;
};

struct DescriptorSet {
    struct ComputePlan* plan;
    std::vector<VkDescriptorSet> sets;
    std::vector<VkDescriptorPool> pools;
};

#endif // INTERNAL_H