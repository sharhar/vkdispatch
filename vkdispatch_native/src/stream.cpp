#include "../internal.hh"

Fence::Fence(VkDevice device) {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CALL(vkCreateFence(device, &fenceInfo, nullptr, &fence));

    this->device = device;
    this->submitted = true;
}

void Fence::waitAndReset() {
    std::unique_lock<std::mutex> lock(mutex);

    cv.wait(lock, [this]() {
        return this->submitted;
    });

    VK_CALL(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CALL(vkResetFences(device, 1, &fence));

    this->submitted = false;
}

void Fence::doSubmit(VkQueue queue, VkSubmitInfo* submitInfo, Signal* signal, std::mutex* queue_usage_mutex) {
    std::unique_lock<std::mutex> lock(mutex);

    queue_usage_mutex->lock();

    VK_CALL(vkQueueSubmit(queue, 1, submitInfo, fence));

    queue_usage_mutex->unlock();

    if(signal != NULL) {
        VK_CALL(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        signal->notify();
    }

    this->submitted = true;
    cv.notify_all();
}

void Fence::destroy() {
    vkDestroyFence(device, fence, nullptr);
}

Stream::Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int device_index, int stream_index) {
    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->device_index = device_index;
    this->stream_index = stream_index;
    this->data_buffer = malloc(1024 * 1024);
    this->data_buffer_size = 1024 * 1024;
    this->recording_thread_count = 2;
    this->sync_record = false;
    this->record_thread_states.resize(recording_thread_count);
    this->run_stream.store(false);

    int inflight_cmd_buffer_count = 4;

    LOG_INFO("Creating stream with device %p, queue %p, queue family index %d", device, queue, queueFamilyIndex);

    commandPools = new VkCommandPool[recording_thread_count];
    commandBufferVectors = new std::vector<VkCommandBuffer>[recording_thread_count];
    commandBufferStates = new bool*[recording_thread_count];

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

        commandBufferStates[i] = new bool[inflight_cmd_buffer_count];
        
        for(int j = 0; j < inflight_cmd_buffer_count; j++) {
            commandBufferStates[i][j] = false;
        }
    }

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    fences.resize(inflight_cmd_buffer_count);
    semaphores.resize(inflight_cmd_buffer_count);
    recording_results.resize(inflight_cmd_buffer_count);

    LOG_INFO("Creating %d fences and semaphores", inflight_cmd_buffer_count);

    for(int i = 0; i < inflight_cmd_buffer_count; i++) {
        fences[i] = new Fence(device);
        VK_CALL(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphores[i]));
    }

    LOG_INFO("Created stream with %d fences and semaphores", inflight_cmd_buffer_count);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(commandBufferVectors[0][0], &beginInfo));
    VK_CALL(vkEndCommandBuffer(commandBufferVectors[0][0]));
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphores.data()[semaphores.size() - 1];
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBufferVectors[0][0];

    LOG_INFO("Submitting initial command buffer");
    VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    LOG_INFO("Waiting for initial command buffer");
    VK_CALL(vkQueueWaitIdle(queue));

    this->run_stream.store(true);

    submit_thread = std::thread([this]() { this->submit_worker(); });
    
    record_threads = new std::thread[recording_thread_count];
    for(int i = 0; i < recording_thread_count; i++) {
        record_threads[i] = std::thread([this, i]() { this->record_worker(i); });
    }
    
    ingest_thread = std::thread([this]() { this->ingest_worker(); });
}

void Stream::destroy() {
    this->run_stream.store(false);
    this->record_queue_cv.notify_all();
    this->submit_queue_cv.notify_all();

    ingest_thread.join();
    
    for(int i = 0; i < recording_thread_count; i++) {
        record_threads[i].join();
    }

    delete[] record_threads;
    
    submit_thread.join();
    
    for(int i = 0; i < semaphores.size(); i++) {
        vkDestroySemaphore(device, semaphores[i], nullptr);
    }

    for(int i = 0; i < fences.size(); i++) {
        fences[i]->destroy();
    }

    for(int i = 0; i < recording_thread_count; i++) {
        for(int j = 0; j < commandBufferVectors[i].size(); j++) {
            vkFreeCommandBuffers(device, commandPools[i], 1, &commandBufferVectors[i][j]);
        }

        vkDestroyCommandPool(device, commandPools[i], nullptr);
        delete[] commandBufferStates[i];
    }

    //vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());
    //vkDestroyCommandPool(device, commandPool, nullptr);

    fences.clear();
    semaphores.clear();
    recording_results.clear();

}

void Stream::ingest_worker() {
    struct Context* ctx = this->ctx;
    WorkQueue* work_queue = ctx->work_queue;
    struct WorkHeader* work_header = NULL;

    int current_index = fences.size() - 1;

    while(this->run_stream.load()) {
        fences[current_index]->waitAndReset();
        
        if(!work_queue->pop(&work_header, stream_index)) {
            LOG_INFO("Thread worker for device %d, stream %d has no more work", device_index, stream_index);
            this->run_stream = false;
            break;
        }

        struct WorkQueueItem work_item;
        work_item.current_index = current_index;
        work_item.work_header = work_header;
        work_item.signal = work_header->signal;
        work_item.recording_result = &recording_results[current_index];
        work_item.recording_result->state = &commandBufferStates[0][current_index];

        int last_index = current_index;
        current_index = (current_index + 1) % fences.size();
        
        work_item.next_index = current_index;
        
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

    LOG_INFO("Thread worker for device %d, stream %d has quit", device_index, stream_index);
}

void Stream::record_worker(int worker_id) {
    VkMemoryBarrier memory_barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
    };

    int cmd_buffer_index = 0;

    bool doing_synchronouns_record = false;

    while(this->run_stream.load()) {
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
                if(!this->run_stream.load()) {
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

            if(!this->run_stream.load()) {
                break;
            }

            this->record_thread_states[worker_id] = true;

            work_item = this->record_queue.front();

            if(work_item.work_header->record_type == RECORD_TYPE_SYNC) {
                this->sync_record = true;
                doing_synchronouns_record = true;
            }
            
            this->record_queue.pop();

            LOG_INFO("Record Worker %d has work %p of index (%d) with next index (%d)", worker_id, work_item.work_header, work_item.current_index, work_item.next_index);
        }

        VkCommandBuffer cmd_buffer = commandBufferVectors[worker_id][cmd_buffer_index];

        work_item.recording_result->commandBuffer = cmd_buffer;

        cmd_buffer_index = (cmd_buffer_index + 1) % commandBufferVectors[worker_id].size();

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CALL(vkBeginCommandBuffer(cmd_buffer, &beginInfo));

        std::shared_ptr<std::vector<struct CommandInfo>> command_buffer = work_item.work_header->commands;

        char* current_instance_data = (char*)&work_item.work_header[1];
        for(size_t instance = 0; instance < work_item.work_header->instance_count; instance++) {
            for (size_t i = 0; i < command_buffer->size(); i++) {
                LOG_VERBOSE("Recording command %d of type %s on worker %d", i, command_buffer->operator[](i).name, worker_id);

                LOG_VERBOSE("Executing command %d", i);
                command_buffer->operator[](i).func->operator()(cmd_buffer, device_index, stream_index, worker_id, current_instance_data);
                current_instance_data += command_buffer->operator[](i).pc_size;

                LOG_VERBOSE("Command %d executed", i);

                if(i < command_buffer->size() - 1) {
                    vkCmdPipelineBarrier(
                        cmd_buffer, 
                        command_buffer->operator[](i).pipeline_stage, 
                        command_buffer->operator[](i+1).pipeline_stage, 
                        0, 1, 
                        &memory_barrier, 
                        0, 0, 0, 0);
                } else if (instance != work_item.work_header->instance_count - 1 && i == command_buffer->size() - 1) {
                    vkCmdPipelineBarrier(
                        cmd_buffer, 
                        command_buffer->operator[](i).pipeline_stage, 
                        command_buffer->operator[](0).pipeline_stage, 
                        0, 1, 
                        &memory_barrier, 
                        0, 0, 0, 0);
                }
            }
        }
        
        VK_CALL(vkEndCommandBuffer(cmd_buffer));

        ctx->work_queue->finish(work_item.work_header);

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);
            work_item.recording_result->state[0] = true;
            this->submit_queue_cv.notify_all();
        }
    }
}

void Stream::submit_worker() {
    while(this->run_stream.load()) {
        struct WorkQueueItem work_item;

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);

            this->submit_queue_cv.wait(lock, [this]() {
                if(!this->run_stream.load()) {
                    return true;
                }
                
                if(this->submit_queue.empty()) {
                    return false;
                }

                return this->submit_queue.front().recording_result->state[0];
            });

            if(!this->run_stream.load()) {
                break;
            }

            work_item = this->submit_queue.front();
            work_item.recording_result->state[0] = false;

            this->submit_queue.pop();
        }


        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores[work_item.current_index];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &work_item.recording_result->commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores[work_item.next_index];

        LOG_VERBOSE("Submitting command buffer for work item %p", work_item.work_header);
        
        fences[work_item.current_index]->doSubmit(queue, &submitInfo, work_item.signal, &this->queue_usage_mutex);

        // VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[work_item.current_index]->fence));
    
        // if(work_item.signal != NULL) {
        //     VK_CALL(vkWaitForFences(device, 1, &fences[work_item.current_index]->fence, VK_TRUE, UINT64_MAX));
        //     work_item.signal->notify();
        // }

        // fences[work_item.current_index]->signalSubmission();
    }
}