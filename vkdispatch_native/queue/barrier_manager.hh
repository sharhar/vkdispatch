#ifndef _BARRIER_MANAGER_HH_
#define _BARRIER_MANAGER_HH_

#include "../base.hh"

#include <unordered_map>

struct BufferBarrierInfo {
    uint64_t buffer_id;
    int read;
    int write;
};

class BarrierManager {
public:
    BarrierManager(Context* context);
    void record_barriers(VkCommandBuffer cmd_buffer, struct BufferBarrierInfo* buffer_barrier_infos, int buffer_barrier_count, int queue_index);
    void reset();

    std::unordered_map<uint64_t, std::pair<int, int>> buffer_states;
    Context* ctx;
};

#endif // _BARRIER_MANAGER_HH_