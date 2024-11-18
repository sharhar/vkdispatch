#ifndef _STREAM_SRC_STREAM_H
#define _STREAM_SRC_STREAM_H

#include "base.hh"

#include <atomic>
#include <queue>
#include <thread>

struct RecordingResultData {
    bool* state;
    VkCommandBuffer commandBuffer;
};

struct WorkQueueItem {
    int current_index;
    int next_index;
    struct WorkHeader* work_header;
    Signal* signal;
    RecordingResultData* recording_result;
};

class Fence {
public:
    Fence(VkDevice device);

    void waitAndReset();
    void signalSubmission();

    void destroy();

    VkDevice device;
    VkFence fence;

    bool submitted;

    std::mutex mutex;
    std::condition_variable cv;
};

class Stream {
public:
    Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index);
    void destroy();

    void ingest_worker();
    void record_worker(int worker_id);
    void submit_worker();

    struct Context* ctx;
    VkDevice device;
    VkQueue queue;
    VkCommandPool* commandPools;
    void* data_buffer;
    size_t data_buffer_size;

    std::atomic<bool> run_stream;
    
    std::vector<VkCommandBuffer>* commandBufferVectors;
    bool** commandBufferStates;
    
    std::vector<Fence*> fences;
    
    std::vector<VkSemaphore> semaphores;
    std::vector<struct RecordingResultData> recording_results;

    std::thread ingest_thread;
    std::thread* record_threads;
    int recording_thread_count;

    std::mutex record_submit_mutex;
    std::thread submit_thread;

    std::mutex submit_queue_mutex;
    std::condition_variable submit_queue_cv;
    std::queue<struct WorkQueueItem> submit_queue;

    std::mutex record_queue_mutex;
    std::condition_variable record_queue_cv;
    std::queue<struct WorkQueueItem> record_queue;

    int stream_index;
};

#endif