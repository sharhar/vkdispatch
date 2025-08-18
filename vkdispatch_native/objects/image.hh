#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include "../base.hh"
#include "../libs/VMA.hh"

struct Image {
    struct Context* ctx;
    VkExtent3D extent;
    uint32_t layers;
    uint32_t mip_levels;

    std::vector<VkImage> images;
    std::vector<VmaAllocation> allocations;
    std::vector<VkImageView> imageViews;
    std::vector<VkBuffer> stagingBuffers;
    std::vector<VmaAllocation> stagingAllocations;
    
    uint32_t block_size;

    std::vector<VkImageMemoryBarrier> barriers;
};

struct Sampler {
    struct Context* ctx;
    uint64_t samplers_handle;

    //std::vector<VkSampler> samplers;
};

#endif // SRC_IMAGE_H_