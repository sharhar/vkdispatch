#include "internal.h"

static size_t program_id = 1;

struct CommandList* command_list_create_extern(struct Context* context) {
    struct CommandList* command_list = new struct CommandList();
    LOG_INFO("Creating command list with handle %p", command_list);

    command_list->ctx = context;
    //command_list->staging_count = context->work_queue->max_size * 3;
    command_list->instance_size = 0;
    //command_list->max_batch_count = 100;
    //command_list->staging_index = 0;

    //command_list->staging_spaces.resize(command_list->staging_count, nullptr);

    return command_list;
}

void command_list_destroy_extern(struct CommandList* command_list) {
    LOG_INFO("Destroying command list with handle %p", command_list);

    //for(int i = 0; i < command_list->stages.size(); i++) {
    //    free(command_list->stages[i].user_data);
    //}

    delete command_list;
}

/*

void command_list_record_stage(struct CommandList* command_list, struct Stage stage, bool sync) {
    if(sync)
        context_wait_idle_extern(command_list->ctx);

    command_list->stages.push_back(stage);

    command_list->instance_size += stage.instance_data_size;

    LOG_VERBOSE("Recording stage with instance data size %d", stage.instance_data_size);

    size_t new_size = command_list->instance_size * command_list->max_batch_count;

    if(stage.instance_data_size != 0)
        for(int i = 0; i < command_list->staging_spaces.size(); i++)
            if(command_list->staging_spaces[i])
                command_list->staging_spaces[i] = (char*)realloc(command_list->staging_spaces[i], new_size);
            else 
                command_list->staging_spaces[i] = (char*)malloc(new_size);
        
}

*/

void command_list_record_command(struct CommandList* command_list, struct CommandInfo command, bool sync) {
    //if(sync)
    //    context_wait_idle_extern(command_list->ctx);

    LOG_INFO("Recording command with type %d", command.type);

    command_list->program_id = program_id;
    program_id += 1;

    command_list->commands.push_back(command);

    if(command.type == COMMAND_TYPE_COMPUTE)
        command_list->instance_size += command.info.compute_info.pc_size;

    //command_list->stages.push_back(stage);

    /*
    if(command.type == COMMAND_TYPE_COMPUTE) {
        command_list->instance_size += command.info.compute_info.pc_size;

        size_t new_size = command_list->instance_size * command_list->max_batch_count;
        if(command.info.compute_info.pc_size != 0)
            for(int i = 0; i < command_list->staging_spaces.size(); i++)
                if(command_list->staging_spaces[i])
                    command_list->staging_spaces[i] = (char*)realloc(command_list->staging_spaces[i], new_size);
                else 
                    command_list->staging_spaces[i] = (char*)malloc(new_size);

        
    }
    */

    //command_list->instance_size += stage.instance_data_size;
    //LOG_VERBOSE("Recording stage with instance data size %d", stage.instance_data_size);    

    //if(stage.instance_data_size != 0)
    //    for(int i = 0; i < command_list->staging_spaces.size(); i++)
    //        if(command_list->staging_spaces[i])
    //            command_list->staging_spaces[i] = (char*)realloc(command_list->staging_spaces[i], new_size);
    //        else 
    //            command_list->staging_spaces[i] = (char*)malloc(new_size);
}

void command_list_get_instance_size_extern(struct CommandList* command_list, unsigned long long* instance_size) {
    size_t instance_data_size = 0;

    //for(int i = 0; i < command_list->stages.size(); i++) {
    //    instance_data_size += command_list->stages[i].instance_data_size;
    //}

    for(int i = 0; i < command_list->commands.size(); i++) {
        if(command_list->commands[i].type == COMMAND_TYPE_COMPUTE) {
            instance_data_size += command_list->commands[i].info.compute_info.pc_size;
        }
    }

    *instance_size = instance_data_size;

    LOG_VERBOSE("Command List (%p) instance size: %llu", command_list, *instance_size);
}

void command_list_reset_extern(struct CommandList* command_list) {
    LOG_INFO("Waiting for command list to be idle");

    //context_wait_idle_extern(command_list->ctx);

    LOG_INFO("Resetting command list with handle %p", command_list);

    //for(int i = 0; i < command_list->stages.size(); i++) {
    //    free(command_list->stages[i].user_data);
    //}

    LOG_INFO("Clearing command list stages and staging spaces");

    //command_list->stages.clear();
    command_list->commands.clear();
    command_list->instance_size = 0;

    LOG_INFO("Clearing staging spaces");

    //for(int i = 0; i < command_list->staging_spaces.size(); i++) {
    //    if(command_list->staging_spaces[i] != nullptr) {
    //        LOG_INFO("Freeing staging space %p", command_list->staging_spaces[i]);
    //        free(command_list->staging_spaces[i]);
    //        command_list->staging_spaces[i] = nullptr;
    //    }
    //}

}

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* indicies, int count, int per_device, void* signal) {
    struct Context* ctx = command_list->ctx;
    
    if(indicies[0] == -2) {
        if(signal != NULL) {
            set_error("Signal is not supported for all streams");
            return;
        }

        for(int i = 0; i < ctx->stream_indicies.size(); i++) {
            ctx->work_queue->push(command_list, instance_buffer, instance_count, i, reinterpret_cast<Signal*>(signal));
        }
    } else {
        ctx->work_queue->push(command_list, instance_buffer, instance_count, indicies[0], reinterpret_cast<Signal*>(signal));
    }

    /*
    if(instance_count > command_list->max_batch_count) {
        set_error("Instance count (%d) exceeds max batch count (%d)", instance_count, command_list->max_batch_count);
        return;
    }

    LOG_VERBOSE("Submitting command list with handle %p", command_list);

    struct Context* ctx = command_list->ctx;

    char* instance_buffer_ptr = NULL;
    if(command_list->instance_size > 0) {
        instance_buffer_ptr = command_list->staging_spaces[command_list->staging_index];
        LOG_INFO("Copying instance buffer to staging space %p with size %d and count %d", instance_buffer_ptr, command_list->instance_size, instance_count);
        command_list->staging_index = (command_list->staging_index + 1) % command_list->staging_count;
        memcpy(instance_buffer_ptr, instance_buffer, command_list->instance_size * instance_count);
    }

    if(indicies[0] == -2) {
        if(signal != NULL) {
            set_error("Signal is not supported for all streams");
            return;
        }

        for(int i = 0; i < ctx->stream_indicies.size(); i++) {
            struct WorkInfo work_info;
            work_info.command_list = command_list;
            work_info.instance_data = (char*)instance_buffer_ptr;
            work_info.index = i;
            work_info.instance_count = instance_count;
            work_info.instance_size = command_list->instance_size;
            work_info.signal = NULL;

            LOG_VERBOSE("Pushing work info to list for stream %d", i);
            ctx->work_queue->push(work_info);
        }
    } else {
        struct WorkInfo work_info;
        work_info.command_list = command_list;
        work_info.instance_data = (char*)instance_buffer_ptr;
        work_info.index = indicies[0];
        work_info.instance_count = instance_count;
        work_info.instance_size = command_list->instance_size;
        work_info.signal = reinterpret_cast<Signal*>(signal);

        LOG_VERBOSE("Pushing work info to list for stream %d", indicies[0]);
        ctx->work_queue->push(work_info);
    }
    */
}