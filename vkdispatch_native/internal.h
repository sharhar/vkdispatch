#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

//#include <VKL/VKL.h>

//#ifdef _DEBUG
#define VK_CALL(result) {VkResult ___result = result; if(___result != VK_SUCCESS) { printf("(VkResult = %d) " #result " in %s in %s\n", ___result, __FUNCTION__, __FILE__); }}
//#endif

//#ifndef _DEBUG
//#define VK_CALL(result) result;
//#endif

#define VKL_VALIDATION

#ifndef VKDISPATCH_USE_VOLK
#include <vulkan/vulkan.h>
#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk/volk.h>

#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <stdarg.h>

inline void log_message(const char* level, const char* format, ...) {
    va_list args;
    va_start(args, format);

    // Estimate the size of the full message
    int size = snprintf(NULL, 0, "%s %s\n", level, format) + 1; // +1 for the null terminator
    char* full_format = (char*)malloc(size);
    if (full_format != NULL) {
        snprintf(full_format, size, "%s %s\n", level, format);
        vprintf(full_format, args);
        free(full_format);
    }

    va_end(args);
}

//#define LOGGING_INFO
#define LOGGING_ERROR

#ifdef LOGGING_INFO

inline void log_message_noendl(const char* level, const char* format, ...) {
    va_list args;
    va_start(args, format);

    // Estimate the size of the full message
    int size = snprintf(NULL, 0, "%s%s", level, format) + 1; // +1 for the null terminator
    char* full_format = (char*)malloc(size);
    if (full_format != NULL) {
        snprintf(full_format, size, "%s%s", level, format);
        vprintf(full_format, args);
        free(full_format);
    }

    va_end(args);
}

#define LOG_INFO(format, ...) log_message("[INFO] ", format, ##__VA_ARGS__)
#define LOG_INFO_NOENDL(format, ...) log_message_noendl("[INFO] ", format, ##__VA_ARGS__)

#define LOG_NIL(format, ...) log_message("", format, ##__VA_ARGS__)
#define LOG_NIL_NOENDL(format, ...) log_message_noendl("", format, ##__VA_ARGS__)

#else

#define LOG_INFO(format, ...)
#define LOG_INFO_NOENDL(format, ...)

#define LOG_NIL(format, ...)
#define LOG_NIL_NOENDL(format, ...)

#endif

#ifdef LOGGING_ERROR
#define LOG_ERROR(format, ...) log_message("[ERROR]", format, ##__VA_ARGS__)
#else
#define LOG_ERROR(format, ...)
#endif

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
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkPhysicalDeviceFeatures2> features;
    std::vector<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT> atomicFloatFeatures;
    std::vector<VkPhysicalDeviceProperties2> properties;
    std::vector<VkPhysicalDeviceSubgroupProperties> subgroup_properties;
    std::vector<struct PhysicalDeviceDetails> device_details;
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
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<Stream*> streams;
    std::vector<VmaAllocator> allocators;
};

struct Buffer {
    struct Context* ctx;
    std::vector<VkBuffer> buffers;
    std::vector<VmaAllocation> allocations;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
    std::vector<VkFence> fences;
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