#include "internal.h"

Stream::Stream(vk::Device device, vk::Queue queue, int queueFamilyIndex, uint32_t command_buffer_count) {
    this->device = device;
    this->queue = queue;

    this->commandPool = device.createCommandPool(
        vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(queueFamilyIndex)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
    );

    this->commandBuffers = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo()
        .setCommandPool(this->commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(command_buffer_count)
    );
    
    this->current_index = commandBuffers.size() - 1;

    for(int i = 0; i < command_buffer_count; i++) {
        this->fences.push_back(device.createFence(
            vk::FenceCreateInfo()
            .setFlags(i == current_index ? vk::FenceCreateFlags() : vk::FenceCreateFlagBits::eSignaled)
        ));

        this->semaphores.push_back(device.createSemaphore(vk::SemaphoreCreateInfo()));
    }

    commandBuffers[0].begin(
        vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
    );

    commandBuffers[0].end();

    queue.submit(
        vk::SubmitInfo()
        .setPSignalSemaphores(&semaphores.data()[1])
        .setSignalSemaphoreCount(semaphores.size() - 1)
        .setCommandBuffers(commandBuffers[0])
    , fences[current_index]);

    device.waitForFences(fences[current_index], VK_TRUE, UINT64_MAX);
}

void Stream::destroy() {
    for(int i = 0; i < fences.size(); i++) {
        device.destroyFence(fences[i]);
    }

    device.freeCommandBuffers(commandPool, commandBuffers);
    device.destroyCommandPool(commandPool); 

    fences.clear();
}

vk::CommandBuffer& Stream::begin() {
    device.waitForFences(fences[current_index], VK_TRUE, UINT64_MAX);
    device.resetFences(fences[current_index]);

    commandBuffers[current_index].begin(
        vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
    );

    return commandBuffers[current_index];
}

vk::Fence& Stream::submit() {
    commandBuffers[current_index].end();

    vk::Fence& result = fences[current_index];

    int last_index = current_index;
    current_index = (current_index + 1) % commandBuffers.size();

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eAllCommands;

    queue.submit(
        vk::SubmitInfo()
        .setWaitDstStageMask(waitStage)
        .setWaitSemaphores(semaphores[last_index])
        .setSignalSemaphores(semaphores[current_index])
        .setCommandBuffers(commandBuffers[last_index])
    , result);

    return result;
}