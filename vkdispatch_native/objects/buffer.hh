#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

#include "../base.hh"

#include "../libs/VMA.h"

#include <vector>
#include <functional>

struct Buffer {
    struct Context* ctx;
    uint64_t size;
    std::vector<VkBuffer> buffers;
    std::vector<VmaAllocation> allocations;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
};

#endif // SRC_BUFFER_H_