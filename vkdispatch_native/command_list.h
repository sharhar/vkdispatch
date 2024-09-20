#ifndef SRC_COMMAND_LIST_H
#define SRC_COMMAND_LIST_H

#include "base.h"


struct CommandList* command_list_create_extern(struct Context* context);
void command_list_destroy_extern(struct CommandList* command_list);

void command_list_begin_extern(struct CommandList* command_list);
void command_list_end_extern(struct CommandList* command_list);

unsigned int command_list_register_instance_data(struct CommandList* command_list, unsigned int instance_size);

void command_list_get_instance_size_extern(struct CommandList* command_list, unsigned long long* instance_size);

void command_list_wait_extern(struct CommandList* command_list);
void command_list_reset_extern(struct CommandList* command_list);
void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int index, void* signal);

#endif // SRC_COMMAND_LIST_H