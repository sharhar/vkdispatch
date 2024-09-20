#ifndef _STREAM_SRC_STREAM_H
#define _STREAM_SRC_STREAM_H

#include "base.h"

#include <thread>

class Stream {
public:
    Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index);
    void destroy();

    void thread_worker();

    struct Context* ctx;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;
    void* data_buffer;
    size_t data_buffer_size;
    
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkFence> fences;
    std::vector<VkSemaphore> semaphores;
    std::vector<Semaphore*> semaphore_objects;
    
    std::thread work_thread;
    int current_index;
    int stream_index;
};

#endif