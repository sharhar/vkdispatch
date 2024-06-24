#ifndef SRC_COMMAND_LIST_H
#define SRC_COMMAND_LIST_H

#include "base.h"

struct CommandList* command_list_create_extern(struct Context* context);
void command_list_destroy_extern(struct CommandList* command_list);

void command_list_get_instance_size_extern(struct CommandList* command_list, unsigned long long* instance_size);

void command_list_reset_extern(struct CommandList* command_list);
void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int* indicies, int count, int per_device, void* signal);

#endif // SRC_COMMAND_LIST_H