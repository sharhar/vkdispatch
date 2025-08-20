#ifndef SRC_DEVICE_CONTEXT_H_
#define SRC_DEVICE_CONTEXT_H_

#include "../base.hh"

#include "../libs/VMA.hh"

#include "../queue/queue.hh"
#include "../queue/barrier_manager.hh"
#include "../queue/work_queue.hh"

#include "handles.hh"

#include <functional>

struct Context {
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkDevice> devices;
    std::vector<std::vector<int>> queue_index_map;
    std::vector<Queue*> queues;
    std::vector<VmaAllocator> allocators;

    HandleManager* handle_manager;

    std::mutex glslang_mutex;
    std::mutex vma_mutex;

    void* glslang_resource_limits;

    struct CommandList* command_list;
    WorkQueue* work_queue;
};

void context_submit_command(
    Context* context, 
    const char* name,
    int queue_index,
    RecordType record_type,
    std::function<void(VkCommandBuffer, struct ExecIndicies, void*, BarrierManager*, uint64_t)> func
);

#endif  // SRC_DEVICE_CONTEXT_H_