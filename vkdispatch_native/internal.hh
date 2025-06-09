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

#include <chrono>
#include <functional>
#include <memory>

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
#include <shared_mutex>
#include <unordered_map>

#include "./context/init.hh"
#include "./context/errors.hh"
#include "./context/context.hh"
#include "./context/log.hh"

#include "./objects/buffer.hh"
#include "./objects/image.hh"
#include "./objects/command_list.hh"
#include "./objects/descriptor_set.hh"

#include "./stages/stage_transfer.hh"
#include "./stages/stage_fft.hh"
#include "./stages/stage_compute.hh"

#include "./stream/work_queue.hh"
#include "./stream/signal.hh"
#include "./stream/stream.hh"

struct HandleHeader {
    uint64_t handle;
    size_t count;
    uint64_t* data;
    bool per_device;
    const char* name;
};

class HandleManager {
public:
    uint64_t next_handle;
    int stream_count;
    int* stream_to_device_map;
    std::shared_mutex handle_mutex;

    std::unordered_map<uint64_t, struct HandleHeader> handles;

    HandleManager(Context* ctx);

    uint64_t register_device_handle(const char* name);
    uint64_t register_stream_handle(const char* name);
    uint64_t register_handle(const char* name, size_t count, bool per_device);

    void set_handle(int64_t index, uint64_t handle, uint64_t value);
    void set_handle_per_device(int device_index, uint64_t handle, std::function<uint64_t(int)> value_func);
    uint64_t get_handle(int64_t index, uint64_t handle);
    uint64_t* get_handle_pointer(int64_t index, uint64_t handle);
    uint64_t get_handle_no_lock(int64_t index, uint64_t handle);
    uint64_t* get_handle_pointer_no_lock(int64_t index, uint64_t handle);
    void destroy_handle(int64_t index, uint64_t handle, std::function<void(uint64_t)> destroy_func);
};

typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkPhysicalDeviceFeatures2> features;
    std::vector<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT> atomicFloatFeatures;
    std::vector<VkPhysicalDeviceShaderFloat16Int8Features> float16int8Features;
    std::vector<VkPhysicalDevice16BitStorageFeatures> storage16bit;
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
    std::vector<std::vector<int>> stream_index_map;
    std::vector<Stream*> streams;
    std::vector<VmaAllocator> allocators;

    HandleManager* handle_manager;

    std::mutex glslang_mutex;

    void* glslang_resource_limits;

    struct CommandList* command_list;
    WorkQueue* work_queue;
};

struct Buffer {
    struct Context* ctx;
    uint64_t size;
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

struct CommandInfo {
    std::shared_ptr<std::function<void(VkCommandBuffer, int, int, int, void*)>> func;
    VkPipelineStageFlags pipeline_stage;
    size_t pc_size;
    const char* name;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct CommandInfo> commands;
    size_t compute_instance_size;
    size_t program_id;
};

void command_list_record_command(
    struct CommandList* command_list, 
    const char* name,
    size_t pc_size,
    VkPipelineStageFlags pipeline_stage,
    std::function<void(VkCommandBuffer, int, int, int, void*)> func
);

struct ComputePlan {
    struct Context* ctx;
    uint64_t descriptorSetLayouts_handle;
    uint64_t pipelineLayouts_handle;
    uint64_t pipelines_handle;
    
    VkDescriptorPoolSize* poolSizes;
    VkDescriptorSetLayoutBinding* bindings;

    unsigned int binding_count;
    unsigned int pc_size;

    uint32_t* code;
    size_t code_size;
};

struct DescriptorSet {
    struct ComputePlan* plan;
    uint64_t sets_handle;
    uint64_t pools_handle;
};

#endif // INTERNAL_H