#include "../include/internal.hh"

static size_t program_id = 1;

struct CommandList* command_list_create_extern(struct Context* context) {
    struct CommandList* command_list = new struct CommandList();
    LOG_INFO("Creating command list with handle %p", command_list);

    command_list->ctx = context;
    command_list->conditional_boolean_count = 0;
    command_list->compute_instance_size = 0;

    return command_list;
}

void command_list_destroy_extern(struct CommandList* command_list) {
    LOG_INFO("Destroying command list with handle %p", command_list);
    delete command_list;
}

void command_list_record_command(struct CommandList* command_list, struct CommandInfo command) {
    //LOG_INFO("Recording command with type %d", command.type);

    command_list->program_id = program_id;
    program_id += 1;

    command_list->commands.push_back(command);

    if(command.type == COMMAND_TYPE_COMPUTE)
        command_list->compute_instance_size += command.info.compute_info.pc_size;
}

unsigned long long command_list_get_instance_size_extern(struct CommandList* command_list) {
    return command_list->compute_instance_size + ((command_list->conditional_boolean_count + 7) / 8);
}

void command_list_reset_extern(struct CommandList* command_list) {
    LOG_INFO("Resetting command list with handle %p", command_list);
    
    command_list->commands.clear();
    command_list->compute_instance_size = 0;
    command_list->conditional_boolean_count = 0;

    LOG_INFO("Command list reset");
}

void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int* indicies, int count, void* signal) {
    struct Context* ctx = command_list->ctx;
    
    LOG_INFO("Submitting command list with handle %p to stream %d", command_list, indicies[0]);

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
}