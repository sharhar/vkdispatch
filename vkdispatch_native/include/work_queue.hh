#ifndef _SRC_WORK_QUEUE_H_
#define _SRC_WORK_QUEUE_H_

#include "base.hh"

#include <mutex>
#include <condition_variable>

struct ProgramHeader {
    unsigned int command_count;
    int info_index;
    size_t array_size;
    size_t conditional_boolean_count;
};

struct ProgramInfo {
    struct ProgramHeader* header;
    int ref_count;
    size_t program_id;
};

struct WorkHeader {
    Signal* signal;
    struct ProgramHeader* program_header;
    size_t array_size;
    int info_index;
    unsigned int instance_count;
    unsigned int instance_size;
};

enum WorkState {
    //WORK_STATE_AVAILABLE = 0,
    WORK_STATE_PENDING = 1,
    WORK_STATE_ACTIVE = 2,
};

struct WorkInfo2 {
    struct WorkHeader* header;
    WorkState state;
    bool dirty;
    int stream_index;
    int program_index;
    size_t work_id;
};

class WorkQueue {
public:
    WorkQueue(int max_work_items, int max_programs);

    void stop();
    void push(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int stream_index, Signal* signal);
    bool pop(struct WorkHeader** header, int stream_index);
    void finish(struct WorkHeader* header);

    std::mutex mutex;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;

    struct WorkInfo2* work_infos;
    struct ProgramInfo* program_infos;
    int work_info_count;
    int program_info_count;

    int max_size;
    bool running;
};

#endif