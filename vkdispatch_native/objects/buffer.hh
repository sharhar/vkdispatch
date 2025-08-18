#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

#include "../base.hh"

#include "../libs/VMA.hh"

#include "../queue/signal.hh"

#include <vector>
#include <functional>

struct Buffer {
    struct Context* ctx;
    uint64_t size;

    uint64_t signals_pointers_handle;

    uint64_t buffers_handle;
    uint64_t allocations_handle;
    uint64_t staging_buffers_handle;
    uint64_t staging_allocations_handle;

    //std::vector<VkBuffer> buffers;
    //std::vector<VmaAllocation> allocations;
    //std::vector<VkBuffer> stagingBuffers;
    //std::vector<VmaAllocation> stagingAllocations;
};

#endif // SRC_BUFFER_H_