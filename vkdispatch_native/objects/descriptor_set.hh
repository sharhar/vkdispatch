#ifndef SRC_DESCRIPTOR_SET_H
#define SRC_DESCRIPTOR_SET_H

#include "../base.hh"
#include "../queue/barrier_manager.hh"

#include <vector>

struct DescriptorSet {
    struct ComputePlan* plan;
    uint64_t sets_handle;
    uint64_t pools_handle;

    std::vector<BufferBarrierInfo> buffer_barriers;
};

#endif // SRC_DESCRIPTOR_SET_H