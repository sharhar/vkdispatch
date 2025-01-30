#ifndef SRC_CONDITIONAL_H
#define SRC_CONDITIONAL_H

#include "base.hh"

struct ConditionalRecordInfo {
    unsigned int conditional_boolean_index;
};

int record_conditional_extern(struct CommandList* command_list);
void record_conditional_end_extern(struct CommandList* command_list);

#endif // SRC_CONDITIONAL_H