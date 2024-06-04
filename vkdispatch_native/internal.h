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
#include <functional>
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


/**
 * @brief Represents a signal that can be used for synchronization.
 *
 * This class provides a simple signal mechanism that can be used for synchronization between threads.
 * It allows one thread to notify other threads that a certain condition has occurred.
 */
class Signal {
public:
    /**
     * @brief Creates a new signal. Must be called from the main thread!!
     */
    Signal();

    /**
     * @brief Notifies the signal. Must be called from a stream thread!!
     *
     * This function sets the state of the signal to true, indicating that the condition has occurred.
     * It wakes up any waiting threads.
     */
    void notify();

    /**
     * @brief Waits for the signal. Must be called from the main thread!!
     *
     * This function blocks the calling thread until the signal is notified.
     * If the signal is already in the notified state, the function returns immediately.
     */
    void wait();
    
    std::mutex mutex;
    std::condition_variable cv;
    bool state;
};

struct WorkInfo {
    struct CommandList* command_list;
    char* instance_data;
    int index;
    unsigned int instance_count;
    Signal* signal;
};

class Queue {
public:
    Queue(int max_size);

    void stop();
    void push(struct WorkInfo elem);
    bool pop(struct WorkInfo* elem, std::function<bool(struct WorkInfo arg)> check);

    std::mutex mutex;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
    std::vector<struct WorkInfo> data;
    int max_size;
    bool running;
};

class Stream {
public:
    Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index);
    void destroy();

    void thread_worker();

    void wait_idle();

    struct Context* ctx;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;
    
    std::vector<VkCommandBuffer> commandBuffers;
    VkFence fence;
    
    std::thread work_thread;
    int current_index;
    int stream_index;

    struct CommandList* command_list;

    //Signal init_signal;
};

struct Context {
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<std::pair<int, int>> stream_indicies;
    std::vector<std::vector<Stream*>> streams;
    std::vector<VmaAllocator> allocators;

    struct CommandList* command_list;
    Queue* work_queue;

    //std::vector<struct WorkInfo*> work_info_list;
    //std::mutex mutex;
    //std::condition_variable cv_push;
    //std::condition_variable cv_pop;
};

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

    size_t staging_count;
    size_t max_batch_count;
    size_t instance_size;
    
    int staging_index;
    std::vector<char*> staging_spaces;
};

void command_list_record_stage(struct CommandList* command_list, struct Stage stage, bool sync = true);

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