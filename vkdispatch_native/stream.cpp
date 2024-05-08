#include "internal.h"

Stream::Stream(vk::Device device, vk::Queue queue, int queueFamilyIndex, uint32_t command_buffer_count) {
    this->device = device;
    this->queue = queue;
    this->current_index = 0;

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

    this->fences.resize(command_buffer_count);

    for(int i = 0; i < command_buffer_count; i++) {
        this->fences[i] = device.createFence(
            vk::FenceCreateInfo()
            .setFlags(vk::FenceCreateFlagBits::eSignaled)
        );
    }
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
    current_index = (current_index + 1) % commandBuffers.size();

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

    queue.submit(
        vk::SubmitInfo()
        .setCommandBufferCount(1)
        .setPCommandBuffers(&commandBuffers[current_index])
    , fences[current_index]);

    return fences[current_index];
}