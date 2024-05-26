#include "internal.h"

struct CommandList* command_list_create_extern(struct Context* context) {
    struct CommandList* command_list = new struct CommandList();
    LOG_INFO("Creating command list with handle %p", command_list);

    command_list->ctx = context;
    return command_list;
}

void command_list_destroy_extern(struct CommandList* command_list) {
    LOG_INFO("Destroying command list with handle %p", command_list);

    for(int i = 0; i < command_list->stages.size(); i++) {
        free(command_list->stages[i].user_data);
    }

    delete command_list;
}

void command_list_get_instance_size_extern(struct CommandList* command_list, unsigned long long* instance_size) {
    size_t instance_data_size = 0;

    for(int i = 0; i < command_list->stages.size(); i++) {
        instance_data_size += command_list->stages[i].instance_data_size;
    }

    *instance_size = instance_data_size;

    LOG_VERBOSE("Command List (%p) instance size: %llu", command_list, *instance_size);
}

void command_list_reset_extern(struct CommandList* command_list) {
    LOG_INFO("Resetting command list with handle %p", command_list);

    for(int i = 0; i < command_list->stages.size(); i++) {
        free(command_list->stages[i].user_data);
    }

    command_list->stages.clear();
}

void command_list_wait_idle_extern(struct CommandList* command_list) {
    std::unique_lock<std::mutex> lock(command_list->ctx->mutex);

    auto check_idle = [command_list] {
        return command_list->work_info_list.size() == 0;
    };




    lock.unlock();
}

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* indicies, int count, int per_device) {
    // For now, we will just submit the command list to the first device
    //LOG_VERBOSE("Submitting command list to device %d", device);

    LOG_INFO("Submitting command list with handle %p", command_list);

    struct Context* ctx = command_list->ctx;

    std::atomic<VkFence>* fence = new std::atomic<VkFence>(VK_NULL_HANDLE);

    struct WorkInfo work_info = {
        command_list,
        indicies[0],
        (char*)instance_buffer,
        instance_count,
        fence 
    };

    LOG_INFO("Pushing work info to list for stream %d", indicies[0]);

    std::unique_lock<std::mutex> lock(ctx->mutex);

    auto check_for_room = [ctx] {
        return ctx->work_info_list.size() < ctx->stream_indicies.size() * 2;
    };

    LOG_INFO("Checking for room");

    if(!check_for_room()) {
        LOG_INFO("Waiting for room");
        ctx->cv_pop.wait(lock, check_for_room);
    }

    LOG_INFO("Adding work info to list");

    ctx->work_info_list.push_back(work_info);

    LOG_INFO("Notifying all");

    ctx->cv_push.notify_all();

    LOG_INFO("unlocking");

    lock.unlock();

/*
    char* instance_data = (char*)instance_buffer;
    char* current_instance_data = instance_data;

    VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
    };

    //int stream_index = command_list->ctx->stream_indicies[0].second;

    VkCommandBuffer cmd_buffer = command_list->ctx->streams[0][0]->begin();
    if(__error_string != NULL)
        return;

    for(size_t instance = 0; instance < instance_count; instance++) {
        LOG_VERBOSE("Recording instance %d", instance);

        for (size_t i = 0; i < command_list->stages.size(); i++) {
            LOG_VERBOSE("Recording stage %d", i);
            command_list->stages[i].record(cmd_buffer, &command_list->stages[i], current_instance_data, 0);

            if(__error_string != NULL)
                return;

            if(i < command_list->stages.size() - 1)
                vkCmdPipelineBarrier(
                    cmd_buffer, 
                    command_list->stages[i].stage, 
                    command_list->stages[i+1].stage, 
                    0, 1, 
                    &memory_barrier, 
                    0, 0, 0, 0);
            current_instance_data += command_list->stages[i].instance_data_size;
        }
    }

    command_list->ctx->streams[0][0]->submit();
    */
}