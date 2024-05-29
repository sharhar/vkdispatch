#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include "base.h"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <stdarg.h>

extern std::mutex __log_mutex;
extern LogLevel __log_level_limit;

inline void log_message(LogLevel log_level, const char* prefix, const char* postfix, const char* format, ...) {
    if(log_level >= __log_level_limit) {
        __log_mutex.lock();

        va_list args;
        va_start(args, format);

        printf("%s", prefix);
        vprintf(format, args);
        printf("%s", postfix);

        va_end(args);

        __log_mutex.unlock();
    }
}

#ifdef LOG_VERBOSE_ENABLED
#define LOG_VERBOSE(format, ...) log_message(LOG_LEVEL_VERBOSE, "[VERBOSE] ", "\n", format, ##__VA_ARGS__)
#else
#define LOG_VERBOSE(format, ...)
#endif

#define LOG_INFO(format, ...) log_message(LOG_LEVEL_INFO, "[INFO] ", "\n", format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) log_message(LOG_LEVEL_WARNING, "[WARNING] ", "\n", format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_message(LOG_LEVEL_ERROR, "[ERROR] ", "\n", format, ##__VA_ARGS__)

extern std::mutex __error_mutex;
extern const char* __error_string;

inline void set_error(const char* format, ...) {
    __error_mutex.lock();
    
    va_list args;
    va_start(args, format);

    if (__error_string != NULL) {
        free((void*)__error_string);
    }

    #ifdef _WIN32
    int length = _vscprintf(format, args) + 1;
    __error_string = (const char*)malloc(length * sizeof(char));
    vsprintf_s((char*)__error_string, length, format, args);
    #else
    vasprintf((char**)&__error_string, format, args);
    #endif
    va_end(args);

    __error_mutex.unlock();
}

#include <vulkan/vk_enum_string_helper.h>

#define VK_CALL_RETURN(EXPRESSION, RET_EXPR)                             \
{                                                                        \
    VkResult ___result = (EXPRESSION);                                   \
    if(___result != VK_SUCCESS) {                                        \
        set_error("(VkResult is %s (%d)) " #EXPRESSION " inside '%s' at %s:%d\n", string_VkResult(___result), ___result, __FUNCTION__, __FILE__, __LINE__); \
        return RET_EXPR;                                                 \
    }                                                                    \
}

#define RETURN_ON_ERROR(RET_EXPR) \
{                                   \
    if(__error_string != NULL) {    \
            return RET_EXPR;        \
    }                               \
}

#define VK_CALL(EXPRESSION) VK_CALL_RETURN(EXPRESSION, ;)
#define VK_CALL_RETNULL(EXPRESSION) VK_CALL_RETURN(EXPRESSION, NULL)

#include <vkFFT.h>
#include <vector>

#include "init.h"
#include "errors.h"
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
    std::vector<std::vector<VkQueueFamilyProperties>> queue_family_properties;
} MyInstance;

extern MyInstance _instance;

struct WorkInfo {
    struct CommandList* command_list;
    char* instance_data;
    int index;
    unsigned int instance_count;
};

class Stream;

struct ThreadInfo {
    struct Context* ctx;
    Stream* stream;
    int index;
    std::atomic<bool>* done;
};

class Stream {
public:
    Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, uint32_t command_buffer_count, int stream_index);
    void destroy();

    void start_thread();

    VkCommandBuffer& begin();
    void submit();

    void wait_idle();

    struct Context* ctx;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;

    std::condition_variable cv_main_done;
    std::condition_variable cv_async_done;
    std::mutex mutex;
    
    std::vector<VkFence> fences;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> semaphores;
    std::vector<VkFence> wait_tasks;
    
    std::atomic<bool> done;
    struct ThreadInfo thread_info;
    std::thread work_thread;
    int current_index;
    int stream_index;
};

struct Context {
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<std::pair<int, int>> stream_indicies;
    std::vector<std::vector<Stream*>> streams;
    std::vector<VmaAllocator> allocators;

    struct CommandList* command_list;
    std::vector<struct WorkInfo> work_info_list;
    std::mutex mutex;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
};

VkFence context_submit_work(struct Context* context, struct WorkInfo work_info);

struct Buffer {
    struct Context* ctx;
    std::vector<VkBuffer> buffers;
    std::vector<VmaAllocation> allocations;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
    std::vector<VkFence> fences;

    bool per_device;
};

struct Image {
    struct Context* ctx;
    
    //VKLImage** images;
    //VKLImageView** imageViews;
    //VKLBuffer** stagingBuffers;

    uint32_t block_size;
};

typedef void (*PFN_stage_record)(VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device_index, int stream_index);

struct Stage {
    PFN_stage_record record;
    void* user_data;
    size_t instance_data_size;
    VkPipelineStageFlags stage;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct Stage> stages;
    unsigned int instance_block_size;
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