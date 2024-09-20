#include "internal.h"

Stream::Stream(struct Context* ctx, VkDevice device, VkQueue queue, int queueFamilyIndex, int stream_index) {
    this->ctx = ctx;
    this->device = device;
    this->queue = queue;
    this->stream_index = stream_index;
    this->data_buffer = malloc(1024 * 1024);
    this->data_buffer_size = 1024 * 1024;

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
   
    semaphore_objects.resize(command_buffer_count);
    for(int i = 0; i < semaphore_objects.size(); i++) {
        semaphore_objects[i] = NULL;
    }

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
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphores.data()[current_index];
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[0];

    LOG_INFO("Submitting initial command buffer");
    VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    LOG_INFO("Waiting for initial command buffer");
    VK_CALL(vkQueueWaitIdle(queue));

    work_thread = std::thread([this]() { this->thread_worker(); });
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

void Stream::thread_worker() {
    struct Context* ctx = this->ctx;
    WorkQueue* work_queue = ctx->work_queue;
    int device_index = ctx->stream_indicies[stream_index].first;
    struct WorkHeader* work_header = NULL;

    bool run_stream = true;
    while(run_stream) {
        auto doFenceWait = [this] () {
            VK_CALL(vkWaitForFences(device, 1, &fences[current_index], VK_TRUE, UINT64_MAX));
            VK_CALL(vkResetFences(device, 1, &fences[current_index]));
        };

        if(this->semaphore_objects[current_index] == NULL) {
            doFenceWait();
        } else {
            this->semaphore_objects[current_index]->finishJob({device, fences[current_index]}, doFenceWait);
            this->semaphore_objects[current_index] = NULL;
        }

        LOG_VERBOSE("Recording command buffer %d for stream %d", current_index, stream_index);

        if(!work_queue->pop(&work_header, stream_index)) {
            LOG_INFO("Thread worker for device %d, stream %d has no more work", device_index, stream_index);
            run_stream = false;
            break;
        }

        //char* current_instance_data = (char*)&work_header[1];

        int last_index = current_index;
        current_index = (current_index + 1) % commandBuffers.size();

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores[last_index];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &work_header->command_list->cmd_buffers[stream_index];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores[current_index];

        LOG_VERBOSE("Submitting command buffer waiting on sempahore %p and signaling semaphore %p", semaphores[last_index], semaphores[current_index]);
        
        VK_CALL(vkQueueSubmit(queue, 1, &submitInfo, fences[last_index]));
        
        semaphore_objects[current_index] = work_header->command_list->semaphore;
        semaphore_objects[current_index]->submitJob(device, {device, fences[last_index]});
        work_queue->finish(work_header);
        
        if(work_header->signal != NULL) {
            VK_CALL(vkWaitForFences(device, 1, &fences[last_index], VK_TRUE, UINT64_MAX));
            work_header->signal->notify();
        }
    }

    LOG_INFO("Thread worker for device %d, stream %d has quit", device_index, stream_index);
}