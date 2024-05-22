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

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* devices, int device_count, int* submission_thread_counts) {
    // For now, we will just submit the command list to the first device
    int device = devices[0];

    LOG_VERBOSE("Submitting command list to device %d", device);

    char* instance_data = (char*)instance_buffer;
    char* current_instance_data = instance_data;

    VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
    };

    VkCommandBuffer cmd_buffer = command_list->ctx->streams[device][0]->begin();
    if(__error_string != NULL)
        return;

    for(size_t instance = 0; instance < instance_count; instance++) {
        LOG_VERBOSE("Recording instance %d", instance);

        for (size_t i = 0; i < command_list->stages.size(); i++) {
            LOG_VERBOSE("Recording stage %d", i);
            command_list->stages[i].record(cmd_buffer, &command_list->stages[i], current_instance_data, device);

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

    command_list->ctx->streams[device][0]->submit();
}