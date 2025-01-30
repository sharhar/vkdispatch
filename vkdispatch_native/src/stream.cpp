#include "../include/internal.hh"

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

void Fence::signalSubmission() {
    std::unique_lock<std::mutex> lock(mutex);
    this->submitted = true;
    cv.notify_all();
}

void Fence::destroy() {
    vkDestroyFence(device, fence, nullptr);
}

Stream::Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index) {
    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->stream_index = stream_index;
    this->data_buffer = malloc(1024 * 1024);
    this->data_buffer_size = 1024 * 1024;
    this->recording_thread_count = 2;

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
    int device_index = ctx->stream_indicies[stream_index].first;
    struct WorkHeader* work_header = NULL;

    int current_index = fences.size() - 1;

    while(this->run_stream.load()) {

        //{
        //    std::unique_lock<std::mutex> lock(fence_muticies[current_index]);
        //    VK_CALL(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX));
        //    VK_CALL(vkResetFences(device, 1, &fences[current_index]));
        //}

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

static int record_command(const struct CommandInfo& command_info, VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* current_instance_data) {
    if(command_info.type == COMMAND_TYPE_NOOP)
        return 0;

    if(command_info.type == COMMAND_TYPE_BUFFER_COPY) {
        stage_transfer_copy_buffer_exec_internal(cmd_buffer, command_info.info.buffer_copy_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_BUFFER_READ) {
        buffer_read_exec_internal(cmd_buffer, command_info.info.buffer_read_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_BUFFER_WRITE) {
        buffer_write_exec_internal(cmd_buffer, command_info.info.buffer_write_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_IMAGE_READ) {
        image_read_exec_internal(cmd_buffer, command_info.info.image_read_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_IMAGE_WRITE) {
        image_write_exec_internal(cmd_buffer, command_info.info.image_write_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_IMAGE_MIP_MAP) {
        image_generate_mipmaps_internal(cmd_buffer, command_info.info.image_mip_map_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_FFT_INIT) {
        stage_fft_plan_init_internal(command_info.info.fft_init_info, device_index, stream_index, recorder_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_FFT_EXEC) {
        stage_fft_plan_exec_internal(cmd_buffer, command_info.info.fft_exec_info, device_index, stream_index, recorder_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_COMPUTE) {
        stage_compute_plan_exec_internal(cmd_buffer, command_info.info.compute_info, current_instance_data, device_index, stream_index);
        return 0;
    }

    return -1;
}

static bool get_bitmap_boolean(uint8_t* bitmap, size_t index) {
    return (bitmap[index / 8] & (1 << (index % 8))) != 0;
}

static int record_program_instance(
    const struct ProgramHeader* program_header, 
    const struct CommandInfo* command_info_buffer, 
    VkCommandBuffer cmd_buffer, 
    int device_index, 
    int stream_index, 
    int worker_id, 
    char** pCurrent_instance_data,
    uint8_t** pConditionals_bitmap,
    size_t* pConditional_bitmap_size_bytes,
    bool last_instance
) {
    VkMemoryBarrier memory_barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
    };

    // Copy over the conditional bitmap
    if(program_header->conditional_boolean_count > 0) {
        size_t bitmap_size_bytes = (program_header->conditional_boolean_count + 7) / 8;

        if(bitmap_size_bytes > *pConditional_bitmap_size_bytes) {
            *pConditionals_bitmap = (uint8_t*)realloc(*pConditionals_bitmap, bitmap_size_bytes);
            memset(*pConditionals_bitmap, 0, bitmap_size_bytes);
            *pConditional_bitmap_size_bytes = bitmap_size_bytes;
        }

        memcpy(*pConditionals_bitmap, *pCurrent_instance_data, bitmap_size_bytes);
        *pCurrent_instance_data += bitmap_size_bytes;
    }

    bool condition_active = false;
    bool inside_condition = false;

    // Record the commands
    for (size_t i = 0; i < program_header->command_count; i++) {
        LOG_INFO("Recording command %d of type %d on worker %d", i, command_info_buffer[i].type, worker_id);
        
        if(command_info_buffer[i].type == COMMAND_TYPE_CONDITIONAL) {
            LOG_INFO("Conditional command");

            if(condition_active) {
                LOG_ERROR("Nested conditionals are not supported");
                return 1;
            }

            if(!get_bitmap_boolean(*pConditionals_bitmap, command_info_buffer[i].info.conditional_info.conditional_boolean_index)) {
                condition_active = true;
                LOG_INFO("Condition is active");
            }

            inside_condition = true;

            continue;
        }

        if(command_info_buffer[i].type == COMMAND_TYPE_CONDITIONAL_END) {
            if(!inside_condition) {
                LOG_ERROR("Conditional end without a conditional");
                return 1;
            }

            condition_active = false;
            inside_condition = false;

            LOG_INFO("Condition is inactive");

            continue;
        }

        char* instance_data = *pCurrent_instance_data;

        if(command_info_buffer[i].type == COMMAND_TYPE_COMPUTE) {
            *pCurrent_instance_data += command_info_buffer[i].info.compute_info.pc_size;
        }

        if(condition_active) {
            continue;
        }

        LOG_INFO("Executing command %d", i);

        if(record_command(command_info_buffer[i], cmd_buffer, device_index, stream_index, worker_id, instance_data) != 0) {
            LOG_ERROR("Unknown command type %d", command_info_buffer[i].type);
            return 1;
        }

        if(i < program_header->command_count - 1) {
            LOG_VERBOSE("Barrier between command %d and %d", i, i+1);
            vkCmdPipelineBarrier(
                cmd_buffer, 
                command_info_buffer[i].pipeline_stage, 
                command_info_buffer[i+1].pipeline_stage, 
                0, 1, 
                &memory_barrier, 
                0, 0, 0, 0);
        } else if (!last_instance && i == program_header->command_count - 1) {
            vkCmdPipelineBarrier(
                cmd_buffer, 
                command_info_buffer[i].pipeline_stage, 
                command_info_buffer[0].pipeline_stage, 
                0, 1, 
                &memory_barrier, 
                0, 0, 0, 0);
        }
    }

    return 0;
}

void Stream::record_worker(int worker_id) {
    int device_index = ctx->stream_indicies[stream_index].first;

    VkMemoryBarrier memory_barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
    };

    int cmd_buffer_index = 0;

    size_t conditional_bitmap_size_bytes = 1024;

    uint8_t* conditionals_bitmap = (uint8_t*)malloc(conditional_bitmap_size_bytes);
    memset(conditionals_bitmap, 0, 1024);

    while(this->run_stream.load()) {
        struct WorkQueueItem work_item;

        {   
            LOG_VERBOSE("Record Worker %d waiting for work", worker_id);

            std::unique_lock<std::mutex> lock(this->record_queue_mutex);

            LOG_VERBOSE("Record Worker %d has lock", worker_id);

            this->record_queue_cv.wait(lock, [this]() {
                if(!this->run_stream.load()) {
                    return true;
                }

                return !this->record_queue.empty();
            });

            if(!this->run_stream.load()) {
                break;
            }


            work_item = this->record_queue.front();
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

        struct ProgramHeader* program_header = work_item.work_header->program_header;
        struct CommandInfo* command_info_buffer = (struct CommandInfo*)&program_header[1];

        char* current_instance_data = (char*)&work_item.work_header[1];
        for(size_t instance = 0; instance < work_item.work_header->instance_count; instance++) {
            if(record_program_instance(
                    program_header, 
                    command_info_buffer, 
                    cmd_buffer, 
                    device_index, 
                    stream_index, 
                    worker_id, 
                    &current_instance_data,
                    &conditionals_bitmap,
                    &conditional_bitmap_size_bytes,
                    instance == work_item.work_header->instance_count - 1
                ) != 0) 
            {
                return;
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
        
        VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[work_item.current_index]->fence));
    
        if(work_item.signal != NULL) {
            VK_CALL(vkWaitForFences(device, 1, &fences[work_item.current_index]->fence, VK_TRUE, UINT64_MAX));
            work_item.signal->notify();
        }

        fences[work_item.current_index]->signalSubmission();
    }
}