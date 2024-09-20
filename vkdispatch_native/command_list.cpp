#include "internal.h"

struct CommandList* command_list_create_extern(struct Context* context) {
    struct CommandList* command_list = new struct CommandList();
    LOG_INFO("Creating command list with handle %p", command_list);

    command_list->ctx = context;
    command_list->instance_size = 0;
    command_list->ready = false;
    command_list->cmd_buffers = new VkCommandBuffer[context->stream_indicies.size()];
    command_list->semaphore = new Semaphore();

    LOG_INFO("Init command list with handle %p", command_list);

    for(int i = 0; i < context->stream_indicies.size(); i++) {
        auto stream_indicies = context->stream_indicies[i];
        int device_index = stream_indicies.first;
        int stream_index = stream_indicies.second;

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = context->streams[device_index][stream_index]->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        

        VK_CALL_RETNULL(vkAllocateCommandBuffers(context->devices[device_index], &allocInfo, &command_list->cmd_buffers[i]));

        LOG_INFO("Allocating command buffer %d as (%p)", i, command_list->cmd_buffers[i]);
    }

    return command_list;
}

void command_list_destroy_extern(struct CommandList* command_list) {
    LOG_INFO("Destroying command list with handle %p", command_list);
    delete command_list;
}

void command_list_begin_extern(struct CommandList* command_list) {
    if(command_list->ready)
        command_list_reset_extern(command_list);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    
    LOG_INFO("Begin command list with handle %p", command_list);

    for(int i = 0; i < command_list->ctx->stream_indicies.size(); i++) {
        LOG_INFO("Begin command buffer %d which is (%p)", i, command_list->cmd_buffers[i]);
        VK_CALL(vkBeginCommandBuffer(command_list->cmd_buffers[i], &beginInfo));
    }
}

void command_list_end_extern(struct CommandList* command_list) {
    for(int i = 0; i < command_list->ctx->stream_indicies.size(); i++) {
        VK_CALL(vkEndCommandBuffer(command_list->cmd_buffers[i]));
    }

    command_list->ready = true;
}

unsigned int command_list_register_instance_data(struct CommandList* command_list, unsigned int instance_size) {
    size_t offset = command_list->instance_size;
    command_list->instance_size += instance_size;
    return offset;
}

void command_list_get_instance_size_extern(struct CommandList* command_list, unsigned long long* instance_size) {
    *instance_size = command_list->instance_size;
}

void command_list_wait_extern(struct CommandList* command_list) {
    command_list->semaphore->waitForIdle();
}

void command_list_reset_extern(struct CommandList* command_list) {
    command_list_wait_extern(command_list);

    command_list->instance_size = 0;
    command_list->ready = false;

    for(int i = 0; i < command_list->ctx->stream_indicies.size(); i++) {
        VK_CALL(vkResetCommandBuffer(command_list->cmd_buffers[i], 0));
    }
}

void command_list_append_command(struct CommandList* command_list, std::function<void(VkDevice, VkCommandBuffer, int)> func) {
    for(int i = 0; i < command_list->ctx->stream_indicies.size(); i++) {
        auto stream_indicies = command_list->ctx->stream_indicies[i];
        int device_index = stream_indicies.first;
        int stream_index = stream_indicies.second;

        VkCommandBuffer cmd_buffer = command_list->cmd_buffers[i];
        func(command_list->ctx->devices[device_index], cmd_buffer, i);
    }
}

void command_list_submit_command_and_reset(struct CommandList* command_list, int index, std::function<void(VkDevice, VkCommandBuffer, int)> func) {
    command_list_begin_extern(command_list);
    command_list_append_command(command_list, func);
    command_list_end_extern(command_list);
    command_list_submit_extern(command_list, NULL, 0, index, NULL);
    command_list_reset_extern(command_list);
}

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int index, void* signal) {
    if(command_list->ready == false) {
        set_error("Command list is not ready");
        return;
    }

    struct Context* ctx = command_list->ctx;
    
    if(index == -2) {
        if(signal != NULL) {
            set_error("Signal is not supported for all streams");
            return;
        }

        for(int i = 0; i < ctx->stream_indicies.size(); i++) {
            command_list->semaphore->addJob();
            ctx->work_queue->push(command_list, instance_buffer, instance_count, i, reinterpret_cast<Signal*>(signal));
        }
    } else {
        command_list->semaphore->addJob();
        ctx->work_queue->push(command_list, instance_buffer, instance_count, index, reinterpret_cast<Signal*>(signal));
    }
}