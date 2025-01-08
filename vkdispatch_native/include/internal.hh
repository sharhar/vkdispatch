#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include "base.hh"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void set_error(const char* format, ...);

#include <vulkan/vk_enum_string_helper.h>

#define VK_CHECK_RETURN(EXPRESSION, RET_EXPR)                            \
{                                                                        \
    VkResult ___result = (EXPRESSION);                                   \
    if(___result != VK_SUCCESS) {                                        \
        set_error("(VkResult is %s (%d)) " #EXPRESSION " inside '%s' at %s:%d\n", string_VkResult(___result), ___result, __FUNCTION__, __FILE__, __LINE__); \
        RET_EXPR;                                                 \
    }                                                                    \
}

#define VK_CALL_RETURN(EXPRESSION, RET_EXPR) VK_CHECK_RETURN(EXPRESSION, return RET_EXPR;)

#define RETURN_ON_ERROR(RET_EXPR) \
{                                   \
    if(get_error_string_extern() != NULL) {    \
            return RET_EXPR;        \
    }                               \
}

#define VK_GOTO(EXPRESSION, LOCATION) VK_CHECK_RETURN(EXPRESSION, goto LOCATION)
#define VK_CALL(EXPRESSION) VK_CALL_RETURN(EXPRESSION, ;)
#define VK_CALL_RETNULL(EXPRESSION) VK_CALL_RETURN(EXPRESSION, NULL)

#include <vector>

#include "init.hh"
#include "errors.hh"
#include "context.hh"
#include "buffer.hh"
#include "image.hh"
#include "stage_transfer.hh"
#include "stage_fft.hh"
#include "stage_compute.hh"
#include "command_list.hh"
#include "descriptor_set.hh"
#include "work_queue.hh"
#include "signal.hh"
#include "stream.hh"
#include "conditional.hh"
#include "log.hh"

typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkPhysicalDeviceFeatures2> features;
    std::vector<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT> atomicFloatFeatures;
    std::vector<VkPhysicalDeviceProperties2> properties;
    std::vector<VkPhysicalDeviceSubgroupProperties> subgroup_properties;
    std::vector<struct PhysicalDeviceDetails> device_details;
    std::vector<std::vector<VkQueueFamilyProperties>> queue_family_properties;
} MyInstance;

extern MyInstance _instance;

struct Context {
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<std::pair<int, int>> stream_indicies;
    std::vector<std::vector<Stream*>> streams;
    std::vector<VmaAllocator> allocators;

    struct CommandList* command_list;
    WorkQueue* work_queue;
};

struct Buffer {
    struct Context* ctx;
    std::vector<VkBuffer> buffers;
    std::vector<VmaAllocation> allocations;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
};

struct Image {
    struct Context* ctx;
    VkExtent3D extent;
    uint32_t layers;
    uint32_t mip_levels;

    std::vector<VkImage> images;
    std::vector<VmaAllocation> allocations;
    std::vector<VkImageView> imageViews;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
    uint32_t block_size;

    std::vector<VkImageMemoryBarrier> barriers;
};

struct Sampler {
    struct Context* ctx;
    std::vector<VkSampler> samplers;
};

typedef void (*PFN_stage_record)(VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device_index, int stream_index);

struct Stage {
    PFN_stage_record record;
    void* user_data;
    size_t instance_data_size;
    VkPipelineStageFlags stage;
};

enum CommandType {
    COMMAND_TYPE_NOOP = 0,
    COMMAND_TYPE_BUFFER_COPY = 1,
    COMMAND_TYPE_BUFFER_READ = 2,
    COMMAND_TYPE_BUFFER_WRITE = 3,
    COMMAND_TYPE_IMAGE_READ = 4,
    COMMAND_TYPE_IMAGE_MIP_MAP = 5,
    COMMAND_TYPE_IMAGE_WRITE = 6,
    COMMAND_TYPE_FFT_INIT = 7,
    COMMAND_TYPE_FFT_EXEC = 8,
    COMMAND_TYPE_COMPUTE = 9,
    COMMAND_TYPE_CONDITIONAL = 10,
    COMMAND_TYPE_CONDITIONAL_END = 11
};

struct CommandInfo {
    enum CommandType type;
    VkPipelineStageFlags pipeline_stage;
    union {
        struct BufferCopyInfo buffer_copy_info;
        struct BufferReadInfo buffer_read_info;
        struct BufferWriteInfo buffer_write_info;
        struct ImageReadInfo image_read_info;
        struct ImageMipMapInfo image_mip_map_info;
        struct ImageWriteInfo image_write_info;
        struct FFTInitRecordInfo fft_init_info;
        struct FFTExecRecordInfo fft_exec_info;
        struct ComputeRecordInfo compute_info;
        struct ConditionalRecordInfo conditional_info;
    } info;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct CommandInfo> commands;
    size_t compute_instance_size;
    size_t program_id;
    size_t conditional_boolean_count;
};

void command_list_record_command(struct CommandList* command_list, struct CommandInfo command);

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