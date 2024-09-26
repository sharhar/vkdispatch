#include "../include/internal.h"

Stream::Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index) {
    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->stream_index = stream_index;
    this->data_buffer = malloc(1024 * 1024);
    this->data_buffer_size = 1024 * 1024;
    this->recording_thread_count = 2;

    LOG_INFO("Creating stream with device %p, queue %p, queue family index %d", device, queue, queueFamilyIndex);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CALL(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

    int command_buffer_count = 4;

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = command_buffer_count;

    commandBuffers.resize(command_buffer_count);
    VK_CALL(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    fences.resize(command_buffer_count);
    semaphores.resize(command_buffer_count);

    LOG_INFO("Creating %d fences and semaphores", command_buffer_count);

    commandBufferStates = new std::atomic<bool>[command_buffer_count];

    for(int i = 0; i < command_buffer_count; i++) {
        VK_CALL(vkCreateFence(device, &fenceInfo, nullptr, &fences[i]));
        VK_CALL(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphores[i]));
        commandBufferStates[i].store(false);
        //commandBufferStates.push_back(false);
    }

    LOG_INFO("Created stream with %d fences and semaphores", command_buffer_count);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(commandBuffers[0], &beginInfo));
    VK_CALL(vkEndCommandBuffer(commandBuffers[0]));
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphores.data()[semaphores.size() - 1];
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[0];

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
    for(int i = 0; i < semaphores.size(); i++) {
        vkDestroySemaphore(device, semaphores[i], nullptr);
    }

    for(int i = 0; i < fences.size(); i++) {
        vkDestroyFence(device, fences[i], nullptr);
    }

    vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());
    vkDestroyCommandPool(device, commandPool, nullptr);

    fences.clear();
    semaphores.clear();
    commandBuffers.clear();
}

void Stream::ingest_worker() {
    struct Context* ctx = this->ctx;
    WorkQueue* work_queue = ctx->work_queue;
    int device_index = ctx->stream_indicies[stream_index].first;
    struct WorkHeader* work_header = NULL;

    int current_index = commandBuffers.size() - 1;

    while(this->run_stream.load()) {
        VK_CALL(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX));
        VK_CALL(vkResetFences(device, 1, &fences[current_index]));

        /*

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CALL(vkBeginCommandBuffer(commandBuffers[current_index], &beginInfo));

        LOG_VERBOSE("Recording command buffer %d for stream %d", current_index, stream_index);

        VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
        };

        */

        if(!work_queue->pop(&work_header, stream_index)) {
            LOG_INFO("Thread worker for device %d, stream %d has no more work", device_index, stream_index);
            this->run_stream = false;
            break;
        }

        int last_index = current_index;
        current_index = (current_index + 1) % commandBuffers.size();

        struct WorkQueueItem work_item;
        work_item.current_index = last_index;
        work_item.next_index = current_index;
        work_item.work_header = work_header;
        work_item.signal = work_header->signal;
        work_item.state = &commandBufferStates[last_index];

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);
            this->submit_queue.push(work_item);
            this->submit_queue_cv.notify_all();
        }

        {
            std::unique_lock<std::mutex> lock(this->record_queue_mutex);
            this->record_queue.push(work_item);
            this->record_queue_cv.notify_one();
        }

        /*

        struct ProgramHeader* program_header = work_header->program_header;
        struct CommandInfo* command_info_buffer = (struct CommandInfo*)&program_header[1];
        Signal* signal = work_header->signal;

        char* current_instance_data = (char*)&work_header[1];
        for(size_t instance = 0; instance < work_header->instance_count; instance++) {
            for (size_t i = 0; i < program_header->command_count; i++) {
                switch(command_info_buffer[i].type) {
                    case COMMAND_TYPE_NOOP: {
                        break;
                    }
                    case COMMAND_TYPE_BUFFER_COPY: {
                        stage_transfer_copy_buffer_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.buffer_copy_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_BUFFER_READ: {
                        buffer_read_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.buffer_read_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_BUFFER_WRITE: {
                        buffer_write_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.buffer_write_info, device_index, stream_index);
                        break;   
                    }
                    case COMMAND_TYPE_IMAGE_READ: {
                        image_read_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.image_read_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_IMAGE_WRITE: {
                        image_write_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.image_write_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_FFT_INIT: {
                        stage_fft_plan_init_internal(command_info_buffer[i].info.fft_init_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_FFT_EXEC: {
                        stage_fft_plan_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.fft_exec_info, device_index, stream_index);
                        break;
                    }
                    case COMMAND_TYPE_COMPUTE: {
                        stage_compute_plan_exec_internal(commandBuffers[current_index], command_info_buffer[i].info.compute_info, current_instance_data, device_index, stream_index);
                        current_instance_data += command_info_buffer[i].info.compute_info.pc_size;
                        break;   
                    }
                    default: {
                        //set_error("Unknown command type %d", work_info.command_list->commands[i].type);
                        LOG_ERROR("Unknown command type %d", command_info_buffer[i].type);
                        return;
                    }
                }

                RETURN_ON_ERROR(;)

                if(i < program_header->command_count - 1)
                    vkCmdPipelineBarrier(
                        commandBuffers[current_index], 
                        command_info_buffer[i].pipeline_stage, 
                        command_info_buffer[i+1].pipeline_stage, 
                        0, 1, 
                        &memory_barrier, 
                        0, 0, 0, 0);
            }
        }

        work_queue->finish(work_header);
        
        VK_CALL(vkEndCommandBuffer(commandBuffers[current_index]));

        int last_index = current_index;
        current_index = (current_index + 1) % commandBuffers.size();

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores[last_index];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[last_index];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores[current_index];

        LOG_VERBOSE("Submitting command buffer waiting on sempahore %p and signaling semaphore %p", semaphores[last_index], semaphores[current_index]);
        
        VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[last_index]));
        
        if(signal != NULL) {
            VK_CALL(vkWaitForFences(device, 1, &fences[last_index], VK_TRUE, UINT64_MAX));
            signal->notify();
        }

        */
    }

    LOG_INFO("Thread worker for device %d, stream %d has quit", device_index, stream_index);
}

static int record_command(const struct CommandInfo& command_info, VkCommandBuffer cmd_buffer, int device_index, int stream_index, void* current_instance_data) {
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

    if(command_info.type == COMMAND_TYPE_FFT_INIT) {
        stage_fft_plan_init_internal(command_info.info.fft_init_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_FFT_EXEC) {
        stage_fft_plan_exec_internal(cmd_buffer, command_info.info.fft_exec_info, device_index, stream_index);
        return 0;
    }

    if(command_info.type == COMMAND_TYPE_COMPUTE) {
        stage_compute_plan_exec_internal(cmd_buffer, command_info.info.compute_info, current_instance_data, device_index, stream_index);
        return command_info.info.compute_info.pc_size;
    }

    return -1;
}

void Stream::record_worker(int worker_id) {
    int device_index = ctx->stream_indicies[stream_index].first;

    VkMemoryBarrier memory_barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
    };

    while(this->run_stream.load()) {
        struct WorkQueueItem work_item;

        LOG_INFO("Waiting for work item to record");

        {
            std::unique_lock<std::mutex> lock(this->record_queue_mutex);

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
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CALL(vkBeginCommandBuffer(commandBuffers[work_item.current_index], &beginInfo));

        LOG_INFO("Recording work item %p for worker_id %d state %d", work_item.work_header, worker_id, work_item.state->load());

        struct ProgramHeader* program_header = work_item.work_header->program_header;
        struct CommandInfo* command_info_buffer = (struct CommandInfo*)&program_header[1];

        char* current_instance_data = (char*)&work_item.work_header[1];
        for(size_t instance = 0; instance < work_item.work_header->instance_count; instance++) {
            for (size_t i = 0; i < program_header->command_count; i++) {
                int pc_size = record_command(command_info_buffer[i], commandBuffers[work_item.current_index], device_index, stream_index, current_instance_data);
                RETURN_ON_ERROR(;)
                
                if(pc_size < 0) {
                    LOG_ERROR("Unknown command type %d", command_info_buffer[i].type);
                    return;
                }

                current_instance_data += pc_size;

                if(i < program_header->command_count - 1)
                    vkCmdPipelineBarrier(
                        commandBuffers[work_item.current_index], 
                        command_info_buffer[i].pipeline_stage, 
                        command_info_buffer[i+1].pipeline_stage, 
                        0, 1, 
                        &memory_barrier, 
                        0, 0, 0, 0);
            }
        }

        LOG_INFO("Finished recording work item %p for stream %d", work_item.work_header, stream_index);
        
        VK_CALL(vkEndCommandBuffer(commandBuffers[work_item.current_index]));

        ctx->work_queue->finish(work_item.work_header);

        work_item.state->store(true);
        this->submit_queue_cv.notify_all();
    }
}

void Stream::submit_worker() {
    while(this->run_stream.load()) {
        struct WorkQueueItem work_item;

        LOG_INFO("Waiting for work item to submit");

        {
            std::unique_lock<std::mutex> lock(this->submit_queue_mutex);

            this->submit_queue_cv.wait(lock, [this]() {
                if(!this->run_stream.load()) {
                    return true;
                }
                
                if(this->submit_queue.empty()) {
                    return false;
                }

                return this->submit_queue.front().state->load();
            });

            if(!this->run_stream.load()) {
                break;
            }

            work_item = this->submit_queue.front();
            this->submit_queue.pop();
        }

        work_item.state->store(false);

        LOG_INFO("Submitting work item %p", work_item.work_header);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores[work_item.current_index];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[work_item.current_index];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores[work_item.next_index];

        LOG_VERBOSE("Submitting command buffer waiting on sempahore %p and signaling semaphore %p", semaphores[work_item.current_index], semaphores[work_item.next_index]);
        
        VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[work_item.current_index]));
        
        if(work_item.signal != NULL) {
            VK_CALL(vkWaitForFences(device, 1, &fences[work_item.current_index], VK_TRUE, UINT64_MAX));
            work_item.signal->notify();
        }
    }
}