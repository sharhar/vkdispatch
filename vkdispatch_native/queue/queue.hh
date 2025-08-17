#ifndef _QUEUE_SRC_QUEUE_H
#define _QUEUE_SRC_QUEUE_H

#include "../base.hh"

#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

struct RecordingResultData {
    bool* state;
    VkCommandBuffer commandBuffer;
};

struct WorkQueueItem {
    uint64_t current_index;
    struct WorkHeader* work_header;
    Signal* signal;
    RecordingResultData* recording_result;
    VkPipelineStageFlags waitStage;
};

class Queue {
public:
    Queue(
        struct Context* ctx,
        VkDevice device,
        VkQueue queue,
        int queueFamilyIndex,
        int device_index,
        int queue_index,
        int recording_thread_count,
        int inflight_cmd_buffer_count);

    void destroy();

    void ingest_worker();
    void record_worker(int worker_id);
    void submit_worker();

    void fused_worker();

    struct Context* ctx;
    VkDevice device;
    int device_index;
    VkQueue queue;
    VkCommandPool* commandPools;
    void* data_buffer;
    size_t data_buffer_size;

    std::atomic<bool> run_queue;
    
    std::vector<VkCommandBuffer>* commandBufferVectors;
    bool* commandBufferStates;

    VkSemaphore timeline_semaphore;
    uint64_t next_signal = 1;
    uint64_t last_completed = 0;
    
    std::vector<struct RecordingResultData> recording_results;

    std::thread ingest_thread;
    std::thread* record_threads;
    int recording_thread_count;
    int inflight_cmd_buffer_count;

    std::mutex record_submit_mutex;
    bool sync_record;
    std::vector<bool> record_thread_states;
    std::thread submit_thread;

    std::mutex submit_queue_mutex;
    std::condition_variable submit_queue_cv;
    std::queue<struct WorkQueueItem> submit_queue;

    std::mutex record_queue_mutex;
    std::condition_variable record_queue_cv;
    std::queue<struct WorkQueueItem> record_queue;

    std::mutex queue_usage_mutex;

    int queue_index;
};

#endif