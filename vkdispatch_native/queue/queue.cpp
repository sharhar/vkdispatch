#include "queue.hh"
#include "barrier_manager.hh"
#include "work_queue.hh"
#include "signal.hh"

#include "../context/context.hh"
#include "../objects/command_list.hh"

Queue::Queue(
    struct Context* ctx,
    VkDevice device,
    VkQueue queue, 
    int queueFamilyIndex,
    int device_index,
    int queue_index,
    int recording_thread_count,
    int inflight_cmd_buffer_count) {

    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->device_index = device_index;
    this->queue_index = queue_index;
    this->data_buffer = malloc(1024 * 1024);
    this->data_buffer_size = 1024 * 1024;
    this->recording_thread_count = recording_thread_count;
    this->sync_record = false;
    this->record_thread_states.resize(recording_thread_count);
    this->run_queue.store(false);
    this->inflight_cmd_buffer_count = inflight_cmd_buffer_count;

    LOG_INFO("Creating queue with VkDevice %p, VkQueue %p, queue family index %d", device, queue, queueFamilyIndex);

    commandPools = new VkCommandPool[recording_thread_count];
    commandBufferVectors = new std::vector<VkCommandBuffer>[recording_thread_count];
    commandBufferStates = new bool[inflight_cmd_buffer_count];

    for(int j = 0; j < inflight_cmd_buffer_count; j++) {
            commandBufferStates[j] = false;
        }

    for(int i = 0; i < recording_thread_count; i++) {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CALL(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPools[i]));

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPools[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = inflight_cmd_buffer_count;

        commandBufferVectors[i].resize(inflight_cmd_buffer_count);
        VK_CALL(vkAllocateCommandBuffers(device, &allocInfo, commandBufferVectors[i].data()));
    }

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreTypeCreateInfo semType{};
    semType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semType.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &semType;

    VK_CALL(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &timeline_semaphore));
    
    recording_results.resize(inflight_cmd_buffer_count);
    waitStages.resize(inflight_cmd_buffer_count);

    LOG_INFO("Creating %d fences and semaphores", inflight_cmd_buffer_count);

    this->run_queue.store(true);

    if(this->recording_thread_count > 1) {
        submit_thread = std::thread([this]() { this->submit_worker(); });
        
        record_threads = new std::thread[recording_thread_count];
        for(int i = 0; i < recording_thread_count; i++) {
            record_threads[i] = std::thread([this, i]() { this->record_worker(i); });
        }
        
        ingest_thread = std::thread([this]() { this->ingest_worker(); });
    } else {
        submit_thread = std::thread([this]() { this->fused_worker(); });
    }
}

void Queue::signal_stop() {
    LOG_INFO("Signaling stop for queue %d", this->queue_index);
    this->run_queue.store(false);
    this->record_queue_cv.notify_all();
    this->submit_queue_cv.notify_all();
}

void Queue::destroy() {
    signal_stop();

    LOG_INFO("Destroying queue with handle %p and ID %d", this, this->queue_index);
    
    LOG_INFO("Waiting for all threads to finish...");
    if(this->recording_thread_count > 1) {
        ingest_thread.join();
        
        for(int i = 0; i < recording_thread_count; i++) {
            record_threads[i].join();
        }

        delete[] record_threads;
    }

    LOG_INFO("Waiting for submit thread to finish...");

    submit_thread.join();

    LOG_INFO("Destroying semaphore and command pools...");

    vkDestroySemaphore(device, timeline_semaphore, nullptr);

    for(int i = 0; i < recording_thread_count; i++) {
        for(int j = 0; j < commandBufferVectors[i].size(); j++) {
            vkFreeCommandBuffers(device, commandPools[i], 1, &commandBufferVectors[i][j]);
        }

        vkDestroyCommandPool(device, commandPools[i], nullptr);
    }

    LOG_INFO("Freeing command pools and command buffers...");

    delete[] commandBufferStates;

    recording_results.clear();
}

void Queue::wait_for_timestamp(uint64_t timestamp) {
    uint64_t last_completed = 0;
    VK_CALL(vkGetSemaphoreCounterValue(device, timeline_semaphore, &last_completed));
    if (last_completed >= timestamp) {
        return;
    }

    while(last_completed < timestamp) {
        VkSemaphoreWaitInfo wi = {};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline_semaphore;
        wi.pValues     = &timestamp;
        VkResult result = vkWaitSemaphores(device, &wi, 1000000000);
        if (result != VK_TIMEOUT) {
            if(result != VK_SUCCESS) {
                LOG_ERROR("Failed to wait for semaphore: %d", result);
            }
            return;
        }

        if(!this->run_queue.load()) {
            return;
        }

        VK_CALL(vkGetSemaphoreCounterValue(device, timeline_semaphore, &last_completed));
    }
}

void ingest_work_item(
    struct WorkQueueItem& work_item,
    Queue* queue,
    WorkQueue* work_queue,
    struct WorkHeader* work_header,
    uint64_t current_index) {

    LOG_VERBOSE("Ingesting work item for queue %d, current index %llu", queue->queue_index, current_index);

    if (current_index + 1 > queue->inflight_cmd_buffer_count) {
        queue->wait_for_timestamp(current_index + 1 - queue->inflight_cmd_buffer_count);
    }
        
    if(!work_queue->pop(&work_header, queue->queue_index)) {
        LOG_INFO("Thread worker for device %d, queue %d has no more work", queue->device_index, queue->queue_index);
        queue->run_queue.store(false);
        return;
    }

    work_item.current_index = current_index;
    work_item.work_header = work_header;
    work_item.signal = work_header->signal;
    work_item.recording_result = &queue->recording_results[current_index % queue->inflight_cmd_buffer_count];
    work_item.recording_result->state = &queue->commandBufferStates[current_index % queue->inflight_cmd_buffer_count];
    work_item.waitStage = &queue->waitStages[current_index % queue->inflight_cmd_buffer_count];
}

void Queue::ingest_worker() {
    struct Context* ctx = this->ctx;
    WorkQueue* work_queue = ctx->work_queue;
    struct WorkHeader* work_header = NULL;

    uint64_t current_index = 0;

    while(this->run_queue.load()) {
        struct WorkQueueItem work_item;

        ingest_work_item(work_item, this, work_queue, work_header, current_index);
        current_index++;

        if(!this->run_queue.load()) {
            break;
        }
        
        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);
            this->submit_queue.push(work_item);
        }

        {
            std::unique_lock<std::mutex> lock(this->record_queue_mutex);
            this->record_queue.push(work_item);
            this->record_queue_cv.notify_one();
        }
    }

    LOG_INFO("Thread worker for device %d, queue %d has quit", device_index, queue_index);
}

int record_work_item(
    struct WorkQueueItem& work_item,
    Queue* queue,
    BarrierManager& barrier_manager,
    int cmd_buffer_index,
    int worker_id) {
     
    VkCommandBuffer cmd_buffer = queue->commandBufferVectors[worker_id][cmd_buffer_index];

    work_item.recording_result->commandBuffer = cmd_buffer;

    cmd_buffer_index = (cmd_buffer_index + 1) % queue->commandBufferVectors[worker_id].size();

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CALL_RETURN(vkBeginCommandBuffer(cmd_buffer, &beginInfo), cmd_buffer_index);

    work_item.waitStage[0] = 0;

    std::shared_ptr<std::vector<struct CommandInfo>> command_buffer = work_item.work_header->commands;

    struct ExecIndicies exec_indices = {};
    exec_indices.device_index = queue->device_index;
    exec_indices.queue_index = queue->queue_index;
    exec_indices.recorder_index = worker_id;

    LOG_INFO("Recording work item %p on queue %d, worker %d, instance count %d", work_item.work_header, queue->queue_index, worker_id, work_item.work_header->instance_count);

    char* current_instance_data = (char*)&work_item.work_header[1];
    for(size_t instance = 0; instance < work_item.work_header->instance_count; instance++) {
        for (size_t i = 0; i < command_buffer->size(); i++) {
            LOG_VERBOSE("Recording command %d of type %s on worker %d", i, command_buffer->operator[](i).name, worker_id);

            command_buffer->operator[](i).func->operator()(cmd_buffer, exec_indices, current_instance_data, &barrier_manager, work_item.current_index + 1);
            current_instance_data += command_buffer->operator[](i).pc_size;

            work_item.waitStage[0] |= command_buffer->operator[](i).pipeline_stage; 
        }
    }
    
    VK_CALL_RETURN(vkEndCommandBuffer(cmd_buffer), cmd_buffer_index);

    barrier_manager.reset();

    queue->ctx->work_queue->finish(work_item.work_header);

    return cmd_buffer_index;
}

void Queue::record_worker(int worker_id) {
    VkMemoryBarrier memory_barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
    };

    int cmd_buffer_index = 0;

    bool doing_synchronouns_record = false;

    BarrierManager barrier_manager(ctx);

    while(this->run_queue.load()) {
        struct WorkQueueItem work_item;

        {   
            LOG_VERBOSE("Record Worker %d waiting for work", worker_id);

            std::unique_lock<std::mutex> lock(this->record_queue_mutex);

            LOG_VERBOSE("Record Worker %d has lock, sync = %d, flag = %d", worker_id, this->sync_record, doing_synchronouns_record);

            if(doing_synchronouns_record) {
                this->sync_record = false;
                doing_synchronouns_record = false;

                this->record_queue_cv.notify_all();
            }

            this->record_thread_states[worker_id] = false;

            this->record_queue_cv.wait(lock, [this]() {
                if(!this->run_queue.load()) {
                    return true;
                }

                if(this->sync_record) {
                    return false;
                }

                if(this->record_queue.empty()) {
                    return false;
                }

                struct WorkQueueItem temp_item = this->record_queue.front();

                if(temp_item.work_header->record_type == RECORD_TYPE_SYNC)
                    for(int i = 0; i < this->recording_thread_count; i++)
                        if(this->record_thread_states[i])
                            return false;


                return true;
            });

            if(!this->run_queue.load()) {
                break;
            }

            this->record_thread_states[worker_id] = true;

            work_item = this->record_queue.front();

            if(work_item.work_header->record_type == RECORD_TYPE_SYNC) {
                this->sync_record = true;
                doing_synchronouns_record = true;
            }
            
            this->record_queue.pop();

            LOG_INFO("Record Worker %d has work %p of index (%d)", worker_id, work_item.work_header, work_item.current_index);
        }

        cmd_buffer_index = record_work_item(work_item, this, barrier_manager, cmd_buffer_index, worker_id);

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);
            work_item.recording_result->state[0] = true;
            this->submit_queue_cv.notify_all();
        }
    }
}

void submit_work_item(
    struct WorkQueueItem& work_item,
    Queue* queue) {

    const uint64_t signalValue = work_item.current_index + 1;

    VkTimelineSemaphoreSubmitInfo timeline_submit_info = { };
    timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit_info.waitSemaphoreValueCount   = 1;
    timeline_submit_info.pWaitSemaphoreValues      = &work_item.current_index;
    timeline_submit_info.signalSemaphoreValueCount = 1;
    timeline_submit_info.pSignalSemaphoreValues    = &signalValue;

    if (work_item.waitStage[0] == 0) {
        work_item.waitStage[0] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    VkSubmitInfo submit_info = { };
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_submit_info;
    submit_info.waitSemaphoreCount   = 1;
    submit_info.pWaitSemaphores      = &queue->timeline_semaphore;
    submit_info.pWaitDstStageMask    = &work_item.waitStage[0];
    submit_info.commandBufferCount   = 1;
    submit_info.pCommandBuffers      = &work_item.recording_result->commandBuffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores    = &queue->timeline_semaphore;

    VK_CALL(vkQueueSubmit(queue->queue, 1, &submit_info, VK_NULL_HANDLE));

    if (work_item.signal != nullptr) {
        VkSemaphoreWaitInfo wait_info = { };
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &queue->timeline_semaphore;
        wait_info.pValues     = &signalValue;
        VK_CALL(vkWaitSemaphores(queue->device, &wait_info, UINT64_MAX));
        work_item.signal->notify();
    }
}

void Queue::submit_worker() {
    while(this->run_queue.load()) {
        struct WorkQueueItem work_item;

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);

            this->submit_queue_cv.wait(lock, [this]() {
                if(!this->run_queue.load()) {
                    return true;
                }
                
                if(this->submit_queue.empty()) {
                    return false;
                }

                return this->submit_queue.front().recording_result->state[0];
            });

            if(!this->run_queue.load()) {
                break;
            }

            work_item = this->submit_queue.front();
            work_item.recording_result->state[0] = false;

            this->submit_queue.pop();
        }

        submit_work_item(work_item, this);
    }
}

void Queue::fused_worker() {
    struct Context* ctx = this->ctx;
    WorkQueue* work_queue = ctx->work_queue;
    struct WorkHeader* work_header = NULL;
    int current_index = 0;
    int cmd_buffer_index = 0;
    BarrierManager barrier_manager(ctx);

    while(this->run_queue.load()) {
        struct WorkQueueItem work_item;

        LOG_INFO("Fused Worker waiting for work");

        ingest_work_item(work_item, this, work_queue, work_header, current_index);
        current_index++;

        LOG_INFO("Fused Worker has work %p of index (%d)", work_item.work_header, work_item.current_index);
        cmd_buffer_index = record_work_item(work_item, this, barrier_manager, cmd_buffer_index, 0);
        
        LOG_INFO("Fused Worker recorded work %p of index (%d)", work_item.work_header, work_item.current_index);
        submit_work_item(work_item, this);
    }
}