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

    work_thread = std::thread(thread_worker, &thread_info);
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

VkFence& Stream::submit() {
    VK_CALL_RETURN(vkEndCommandBuffer(commandBuffers[current_index]), fences[current_index]);

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
    
    VK_CALL_RETURN(vkQueueSubmit(queue, 1, &submitInfo, fences[last_index]), fences[last_index]);

    return fences[last_index];
}

static void thread_worker(struct ThreadInfo* info) {
    VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
    };

    struct Context* ctx = info->ctx;

    int device_index = ctx->stream_indicies[info->index].first;
    int stream_index = ctx->stream_indicies[info->index].second;

    LOG_INFO("Thread worker for device %d, stream %d", device_index, stream_index);

    Stream* stream = ctx->streams[device_index][stream_index];

    LOG_INFO("Thread worker for device %d, stream %d", device_index, stream_index);

    while (!*info->done) {
        std::unique_lock<std::mutex> lock(info->ctx->mutex);

        int* work_info_index = NULL;
        struct WorkInfo* work_info = NULL;
        auto check_for_available_work = [info, work_info, work_info_index] {
            if(*info->done) {
                *work_info_index = 0;
                return true;
            }
            
            if(info->ctx->work_info_list.size() == 0)
                return false;

            *work_info_index = -1;

            for(int i = 0; i < info->ctx->work_info_list.size(); i++) {
                if(info->ctx->work_info_list[i].index == info->index || info->ctx->work_info_list[i].index == -1) {
                    *work_info_index = i;
                    *work_info = info->ctx->work_info_list[i];
                    return true;
                }
            }

            return false;
        };
        
        LOG_INFO("Checking for available work");

        if(!check_for_available_work()) {
            LOG_INFO("Waiting for available work");
            info->ctx->cv_push.wait(lock, check_for_available_work);
        }

        LOG_INFO("Found available work at index %d", *work_info_index);
        
        if(*info->done) {
            lock.unlock();
            return;
        }
        
        LOG_INFO("Removing work info from list");

        info->ctx->work_info_list.erase(info->ctx->work_info_list.begin() + *work_info_index);

        LOG_INFO("Notifying all");

        info->ctx->cv_pop.notify_all();

        LOG_INFO("Unlocking");

        lock.unlock();

        char* current_instance_data = work_info->instance_data;

        VkCommandBuffer cmd_buffer = stream->begin();
        if(__error_string != NULL)
            return;

        for(size_t instance = 0; instance < work_info->instance_count; instance++) {
            LOG_VERBOSE("Recording instance %d", instance);

            for (size_t i = 0; i < work_info->command_list->stages.size(); i++) {
                LOG_VERBOSE("Recording stage %d", i);
                work_info->command_list->stages[i].record(cmd_buffer, &work_info->command_list->stages[i], current_instance_data, 0);

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

        *(work_info->fence) = std::move(stream->submit());
    }

    LOG_INFO("Thread worker for device %d, stream %d done", device_index, stream_index);
}
