#include "internal.h"

static void thread_worker(struct ThreadInfo* info);

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
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    fences.resize(command_buffer_count);
    semaphores.resize(command_buffer_count);

    current_index = commandBuffers.size() - 1;

    LOG_INFO("Creating %d fences and semaphores", command_buffer_count);

    for(int i = 0; i < command_buffer_count; i++) {
        fenceInfo.flags = i == current_index ? 0 : VK_FENCE_CREATE_SIGNALED_BIT;
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

    VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[current_index]));

    LOG_INFO("Waiting for initial command buffer");

    VK_CALL(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX));

    this->done = false;

    thread_info = {};
    thread_info.ctx = ctx;
    thread_info.done = &done;
    thread_info.index = stream_index;
    thread_info.stream = this;
}

void Stream::start_thread() {
    std::unique_lock<std::mutex> lock(mutex);
    
    LOG_INFO("Starting thread worker");

    work_thread = std::thread(thread_worker, &thread_info);

    LOG_INFO("Thread worker started");

    //cv_main_done.notify_all();

    LOG_INFO("Notified all");

    cv_async_done.wait(lock, [] () { return true; });

    LOG_INFO("Waiting for async done");
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

VkCommandBuffer& Stream::begin() {
    VK_CALL_RETURN(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX), commandBuffers[current_index]);
    VK_CALL_RETURN(vkResetFences(device, 1, &fences[current_index]), commandBuffers[current_index]);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL_RETURN(vkBeginCommandBuffer(commandBuffers[current_index], &beginInfo), commandBuffers[current_index]);

    return commandBuffers[current_index];
}

void Stream::submit() {
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
}

void Stream::wait_idle() {
    std::unique_lock<std::mutex> lock(mutex);

    for(int i = 0; i < fences.size(); i++) {
        LOG_WARNING("Fence status for fence %d: %d", i, vkGetFenceStatus(device, fences[i]));
    }

    VK_CALL(vkWaitForFences(device, fences.size(), fences.data(), VK_TRUE, UINT64_MAX));
}

static void thread_worker(struct ThreadInfo* info) {
    Stream* stream = info->stream;

    LOG_INFO("Waiting for mutex on new worker thread");
    
    //std::unique_lock<std::mutex> lock(stream->mutex);

    LOG_INFO("Waiting for main");

    //stream->cv_main_done.wait(lock);

    LOG_INFO("Notified");

    VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
    };

    struct Context* ctx = info->ctx;

    int index = info->index;
    std::atomic<bool>* done = info->done;
    std::mutex* mutex = &ctx->mutex;
    std::vector<struct WorkInfo>* work_info_list = &ctx->work_info_list;

    std::condition_variable* cv_push = &ctx->cv_push;
    std::condition_variable* cv_pop = &ctx->cv_pop;

    int device_index = ctx->stream_indicies[info->index].first;
    int stream_index = ctx->stream_indicies[info->index].second;

    stream->cv_async_done.notify_all();

    LOG_INFO("Thread worker for device %d, stream %d", device_index, stream_index);

    struct WorkInfo* work_info = (struct WorkInfo*)malloc(sizeof(struct WorkInfo));
    
    while (!*done) {
        std::unique_lock<std::mutex> lock(*mutex);

        LOG_INFO("Thread worker for device %d, stream %d waiting for work", device_index, stream_index);
        LOG_INFO("%d", info->ctx->work_info_list.size());

        int work_info_index = -1;
        int* work_info_index_ptr = &work_info_index;
        auto check_for_available_work = [done, index, work_info_list, work_info, work_info_index_ptr] () {
            LOG_INFO("Doing conditional variable check at index %d for thread %d", index, std::this_thread::get_id());
            LOG_INFO("Done: %p", done);
            LOG_INFO("Work Info List: %p", work_info_list);
            LOG_INFO("Work Info List Size: %d", work_info_list->size());

            if(*done) {
                LOG_INFO("Done flag set, returning false");
                *work_info_index_ptr = 0;
                return true;
            }

            LOG_INFO("Checking work list size");
            
            if(work_info_list->size() == 0) {
                LOG_INFO("No work info available, returning false");
                return false;
            }

            *work_info_index_ptr = -1;

            LOG_INFO("Checking for compatible work in list (my index is %d)", index);

            for(int i = 0; i < work_info_list->size(); i++) {
                LOG_INFO("Checking work info for index %d", work_info_list->at(i).index);
                if(work_info_list->at(i).index == index || work_info_list->at(i).index == -1) {
                    LOG_INFO("Found available work at index %d", i);
                    *work_info_index_ptr = i;
                    *work_info = work_info_list->at(i);
                    return true;
                }
            }

            LOG_INFO("No work info found, returning false");

            return false;
        };
        

        LOG_INFO("Thread %d: Checking for available work", std::this_thread::get_id());

        if(!check_for_available_work()) {
            LOG_INFO("Waiting for available work");
            LOG_WARNING("Thread %d: Waiting for push", std::this_thread::get_id());
            cv_push->wait(lock, check_for_available_work);
        }

        LOG_INFO("Found available work at index %d", work_info_index);
        
        if(*done) {
            lock.unlock();
            return;
        }
        
        LOG_INFO("Removing work info from list at index %d", work_info_index);

        work_info_list->erase(work_info_list->begin() + work_info_index);

        LOG_INFO("Notifying all");

        VkCommandBuffer cmd_buffer = stream->begin();
        
        cv_pop->notify_all();

        LOG_INFO("Unlocking");

        lock.unlock();

        char* current_instance_data = work_info->instance_data;

        if(__error_string != NULL)
            return;

        for(size_t instance = 0; instance < work_info->instance_count; instance++) {
            LOG_VERBOSE("Recording instance %d", instance);

            for (size_t i = 0; i < work_info->command_list->stages.size(); i++) {
                LOG_VERBOSE("Recording stage %d", i);
                work_info->command_list->stages[i].record(cmd_buffer, &work_info->command_list->stages[i], current_instance_data, 0, 0);

                if(__error_string != NULL)
                    return;

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

        stream->submit();

        //*(work_info->fence) = std::move(stream->submit());
    }

    LOG_INFO("Thread worker for device %d, stream %d done", device_index, stream_index);
}
