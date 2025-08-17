#ifndef SRC_DEVICE_CONTEXT_H_
#define SRC_DEVICE_CONTEXT_H_

#include "../base.hh"

#include "../libs/VMA.h"

#include "../queue/queue.hh"
#include "../queue/work_queue.hh"

#include "handles.hh"

struct Context {
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<std::vector<int>> queue_index_map;
    std::vector<Queue*> queues;
    std::vector<VmaAllocator> allocators;

    HandleManager* handle_manager;

    std::mutex glslang_mutex;

    void* glslang_resource_limits;

    struct CommandList* command_list;
    WorkQueue* work_queue;
};

#endif  // SRC_DEVICE_CONTEXT_H_