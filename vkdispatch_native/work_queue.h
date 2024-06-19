#ifndef _SRC_WORK_QUEUE_H_
#define _SRC_WORK_QUEUE_H_

#include "base.h"

#include <mutex>
#include <condition_variable>
#include <vector>

struct WorkItemHead {
    Signal* signal;
    unsigned int ref_count;
    int program_index;
    int stream_index;
    unsigned int instance_count;
    unsigned int instance_size;
    size_t array_size;
};

struct ProgramArrayHead {
    unsigned int ref_count;
    unsigned int command_count;
    size_t array_size;
};

class WorkQueue {
public:
    WorkQueue(int max_work_items, int max_programs);

    void stop();
    void push(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int stream_index, Signal* signal);
    bool pop(struct WorkItemHead** elem, int stream_index);
    void finalize(struct WorkItemHead* elem);

    std::mutex mutex;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
    std::vector<struct WorkItem> data;
    int max_size;
    bool running;
};

#endif