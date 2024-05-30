#include "internal.h"

Stream::Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, uint32_t command_buffer_count, int stream_index) {
    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->stream_index = stream_index;

    LOG_INFO("Creating stream with device %p, queue %p, queue family index %d, command buffer count %d", device, queue, queueFamilyIndex, command_buffer_count);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CALL(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

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

    current_index = commandBuffers.size() - 1;

    LOG_INFO("Creating %d fences and semaphores", command_buffer_count);

    for(int i = 0; i < command_buffer_count; i++) {
        VK_CALL(vkCreateFence(device, &fenceInfo, nullptr, &fences[i]));
        VK_CALL(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphores[i]));
    }

    LOG_INFO("Created stream with %d fences and semaphores", command_buffer_count);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(commandBuffers[0], &beginInfo));
    VK_CALL(vkEndCommandBuffer(commandBuffers[0]));
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = semaphores.size() - 1;
    submitInfo.pSignalSemaphores = &semaphores.data()[1];
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[0];

    LOG_INFO("Submitting initial command buffer");

    VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    LOG_INFO("Waiting for initial command buffer");

    VK_CALL(vkQueueWaitIdle(queue));

    this->done = false;

    thread_info = {};
    thread_info.ctx = ctx;
    thread_info.done = &done;
    thread_info.index = stream_index;
    thread_info.stream = this;

    command_list = command_list_create_extern(ctx);

    command_list->stages.push_back({[] (VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device_index, int stream_index) {

        },
        NULL,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT
    });

    work_thread = std::thread([this]() { this->thread_worker(); });
    //init_signal.wait();
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
}

void Stream::wait_idle() {
    Signal* signal = new Signal();
    command_list_submit_extern(this->command_list, NULL, 1, &stream_index, 1, 0, signal);
    signal->wait();
    delete signal;
}

void Stream::thread_worker() {
    struct Context* ctx = this->ctx;

    std::mutex* mutex = &ctx->mutex;
    std::vector<struct WorkInfo*>* work_info_list = &ctx->work_info_list;

    std::condition_variable* cv_push = &ctx->cv_push;
    std::condition_variable* cv_pop = &ctx->cv_pop;

    int device_index = ctx->stream_indicies[stream_index].first;

    //init_signal.notify();

    LOG_INFO("Starting Thread worker for stream %d", stream_index);

    struct WorkInfo* work_info = NULL;
    
    while (!done) {
        std::unique_lock<std::mutex> lock(*mutex);

        int work_info_index = -1;
        int* work_info_index_ptr = &work_info_index;
        cv_push->wait(lock, [this, work_info_list, &work_info, work_info_index_ptr] () {
            if(done) {
                *work_info_index_ptr = 0;
                return true;
            }
            
            if(work_info_list->size() == 0) {
                return false;
            }

            *work_info_index_ptr = -1;

            for(int i = 0; i < work_info_list->size(); i++) {
                if(work_info_list->at(i)->index == stream_index || work_info_list->at(i)->index == -1) {
                    *work_info_index_ptr = i;
                    work_info = work_info_list->at(i);
                    return true;
                }
            }

            return false;
        });
        
        if(done) {
            lock.unlock();
            return;
        }

        work_info_list->erase(work_info_list->begin() + work_info_index);

        VkCommandBuffer cmd_buffer = commandBuffers[current_index];

        VkCommandBufferBeginInfo beginInfo = {};
        
        VK_GOTO(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX), notify_all);
        VK_GOTO(vkResetFences(device, 1, &fences[current_index]), notify_all);

        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_GOTO(vkBeginCommandBuffer(cmd_buffer, &beginInfo), notify_all);
        
notify_all:
        cv_pop->notify_all();

        lock.unlock();

        RETURN_ON_ERROR(;)

        VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
        };
        
        char* current_instance_data = work_info->instance_data;
        for(size_t instance = 0; instance < work_info->instance_count; instance++) {
            LOG_VERBOSE("Recording instance %d", instance);

            for (size_t i = 0; i < work_info->command_list->stages.size(); i++) {
                LOG_VERBOSE("Recording stage %d", i);
                work_info->command_list->stages[i].record(cmd_buffer, &work_info->command_list->stages[i], current_instance_data, device_index, stream_index);
                RETURN_ON_ERROR(;)

                if(i < work_info->command_list->stages.size() - 1)
                    vkCmdPipelineBarrier(
                        cmd_buffer, 
                        work_info->command_list->stages[i].stage, 
                        work_info->command_list->stages[i+1].stage, 
                        0, 1, 
                        &memory_barrier, 
                        0, 0, 0, 0);
                current_instance_data += work_info->command_list->stages[i].instance_data_size;
            }
        }

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
        
        VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[last_index]));

        if(work_info->signal) {
            VK_CALL(vkWaitForFences(device, 1, &fences[last_index], VK_TRUE, UINT64_MAX));
            work_info->signal->notify();
        }

        delete work_info;
    }

    LOG_INFO("Thread worker for device %d, stream %d has quit", device_index, stream_index);
}