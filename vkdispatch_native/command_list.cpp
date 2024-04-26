#include "internal.h"

struct CommandList* command_list_create_extern(struct Context* context) {
    LOG_INFO("Creating command list with context %p", context);

    struct CommandList* command_list = new struct CommandList();
    command_list->ctx = context;
    return command_list;
}

void command_list_destroy_extern(struct CommandList* command_list) {
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

    LOG_INFO("Instance size: %llu", *instance_size);
}

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* devices, int device_count, int* submission_thread_counts) {
    // For now, we will just submit the command list to the first device
    int device = devices[0];

    const char* inst_dat = (const char*)instance_buffer;

    LOG_INFO("Instance data size: %llu", command_list->stages[0].instance_data_size);
    LOG_INFO("Instance bytes %d %d %d %d", inst_dat[0], inst_dat[1], inst_dat[2], inst_dat[3]);
    LOG_INFO("Instance bytes %d %d %d %d", inst_dat[4], inst_dat[5], inst_dat[6], inst_dat[7]);
    LOG_INFO("Instance bytes %d %d %d %d", inst_dat[8], inst_dat[9], inst_dat[10], inst_dat[11]);

    LOG_INFO("Submitting command list to device %d", device);

    char* instance_data = (char*)instance_buffer;
    char* current_instance_data = instance_data;

    command_list->ctx->commandBuffers[device]->reset();

    command_list->ctx->commandBuffers[device]->begin();

    for(size_t instance = 0; instance < instance_count; instance++) {
        for (size_t i = 0; i < command_list->stages.size(); i++) {
            command_list->stages[i].record(command_list->ctx->commandBuffers[device], &command_list->stages[i], current_instance_data, device);
            current_instance_data += command_list->stages[i].instance_data_size;
        }
    }

    command_list->ctx->commandBuffers[device]->end();

    command_list->ctx->devices[device]->waitForFence(command_list->ctx->fences[device]);
    command_list->ctx->devices[device]->resetFence(command_list->ctx->fences[device]);
    command_list->ctx->queues[device]->submit(command_list->ctx->commandBuffers[device], command_list->ctx->fences[device]);
    command_list->ctx->queues[device]->waitIdle();
}