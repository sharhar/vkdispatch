#ifndef SRC_DESCRIPTOR_SET_H
#define SRC_DESCRIPTOR_SET_H

#include "../base.hh"
#include "../queue/barrier_manager.hh"

class BarrierInfoManager;

struct DescriptorSet {
    struct ComputePlan* plan;
    uint64_t sets_handle;
    uint64_t pools_handle;

    BarrierInfoManager* barrier_info_manager;

    std::vector<BufferBarrierInfo> buffer_barrier_list;
};

void descriptor_set_add_buffer_info_list(struct DescriptorSet* desc_set, BufferBarrierInfo* list);

#endif // SRC_DESCRIPTOR_SET_H