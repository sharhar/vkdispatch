#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

//#include <VKL/VKL.h>

#define VK_CALL(result) {VkResult ___result = result; if(___result != VK_SUCCESS) { printf("(VkResult = %d) " #result " in %s in %s\n", ___result, __FUNCTION__, __FILE__); }}


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

#include <vulkan/vulkan.hpp>

#include <vkFFT.h>
#include <vector>

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
#define LOGGING_WARNING
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

#ifdef LOGGING_WARNING
#define LOG_WARNING(format, ...) log_message("[WARNING]", format, ##__VA_ARGS__)
#else
#define LOG_WARNING(format, ...)
#endif

#ifdef LOGGING_ERROR
#define LOG_ERROR(format, ...) log_message("[ERROR]", format, ##__VA_ARGS__)
#else
#define LOG_ERROR(format, ...)
#endif

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
    vk::Instance instance;
    std::vector<struct PhysicalDeviceProperties> devices;
} MyInstance;

extern MyInstance _instance;

class Stream {
public:
    Stream(vk::Device device, vk::Queue queue, int queueFamilyIndex, uint32_t command_buffer_count);
    void destroy();

    vk::CommandBuffer& begin();
    vk::Fence& submit();

    vk::Device device;
    vk::Queue queue;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Fence> fences;
    std::vector<vk::Semaphore> semaphores;
    int current_index;
};

struct Context {
    uint32_t deviceCount;
    std::vector<vk::PhysicalDevice> physicalDevices;
    std::vector<vk::Device> devices;
    std::vector<Stream*> streams;
    std::vector<VmaAllocator> allocators;
    std::vector<uint32_t> submissionThreadCounts;
};

struct Buffer {
    struct Context* ctx;
    std::vector<vk::Buffer> buffers;
    std::vector<VmaAllocation> allocations;
    std::vector<vk::Buffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
    std::vector<vk::Fence> fences;
};

struct Image {
    struct Context* ctx;
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> imageViews;
    std::vector<vk::Buffer> stagingBuffers;
    uint32_t block_size;
};

typedef void (*PFN_stage_record)(vk::CommandBuffer& cmd_buffer, struct Stage* stage, void* instance_data, int device);

struct Stage {
    PFN_stage_record record;
    void* user_data;
    size_t instance_data_size;
    vk::PipelineStageFlags stage;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct Stage> stages;
};

struct FFTPlan {
    struct FFTData {
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        VkCommandPool commandPool;
        VkFence fence;
        uint64_t bufferSize;
    };

    std::vector<FFTData> datas;

    struct Context* ctx;
    std::vector<VkFFTApplication> apps;
    std::vector<VkFFTConfiguration> configs;
    std::vector<VkFFTLaunchParams> launchParams;
};

struct ComputePlan {
    struct Context* ctx;
    std::vector<vk::ShaderModule> modules;
    std::vector<std::vector<vk::DescriptorPoolSize>> poolSizes;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PipelineLayout> pipelineLayouts;
    std::vector<vk::Pipeline> pipelines;
    uint32_t binding_count;
    uint32_t pc_size;
};

struct DescriptorSet {
    struct ComputePlan* plan;
    std::vector<vk::DescriptorSet> sets;
    std::vector<vk::DescriptorPool> pools;
};

#endif // INTERNAL_H