#include "internal.h"

Stream::Stream(VkDevice device, VkQueue queue, int queueFamilyIndex, uint32_t command_buffer_count) {
    this->device = device;
    this->queue = queue;

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