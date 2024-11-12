#include "../include/internal.hh"

int record_conditional_extern(struct CommandList* command_list) {
    LOG_INFO("Recording conditional command list with handle %p", command_list);

    int conditional_boolean_index = command_list->conditional_boolean_count;

    struct CommandInfo command_info = {};
    command_info.type = COMMAND_TYPE_CONDITIONAL;
    command_info.pipeline_stage = VK_PIPELINE_STAGE_NONE;
    command_info.info.conditional_info.conditional_boolean_index = conditional_boolean_index;

    command_list_record_command(command_list, command_info);

    command_list->conditional_boolean_count += 1;

    return conditional_boolean_index;
}

void record_conditional_end_extern(struct CommandList* command_list) {
    struct CommandInfo command_info = {};
    command_info.type = COMMAND_TYPE_CONDITIONAL_END;
    command_info.pipeline_stage = VK_PIPELINE_STAGE_NONE;

    command_list_record_command(command_list, command_info);
}