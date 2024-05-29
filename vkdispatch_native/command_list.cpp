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

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* indicies, int count, int per_device) {
    LOG_INFO("Submitting command list with handle %p", command_list);

    struct Context* ctx = command_list->ctx;

    struct WorkInfo work_info = {};
    work_info.command_list = command_list;
    work_info.instance_data = (char*)instance_buffer;
    work_info.index = indicies[0];
    work_info.instance_count = instance_count;

    LOG_INFO("Pushing work info to list for stream %d", indicies[0]);

    std::unique_lock<std::mutex> lock(ctx->mutex);

    auto check_for_room = [ctx] {
        LOG_INFO("Thread %d: Checking for room", std::this_thread::get_id());
        LOG_INFO("Work Info List Size: %d", ctx->work_info_list.size());
        LOG_INFO("Stream Indicies Size: %d", ctx->stream_indicies.size());
        return ctx->work_info_list.size() < ctx->stream_indicies.size() * 2;
    };

    LOG_INFO("Thread %d: Submitting work to context", std::this_thread::get_id());

    if(!check_for_room()) {
        LOG_INFO("Did not find room, waiting for pop");
        LOG_WARNING("Thread %d: Waiting for pop", std::this_thread::get_id());
        ctx->cv_pop.wait(lock, check_for_room);
    }

    LOG_INFO("Adding work info to list");

    ctx->work_info_list.push_back(work_info);

    LOG_INFO("Notifying all");

    ctx->cv_push.notify_all();

    LOG_INFO("unlocking");

    //context_submit_work(ctx, work_info);
}