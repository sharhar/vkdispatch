#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include "../base.hh"
#include "../libs/VMA.hh"

struct Image {
    struct Context* ctx;
    VkExtent3D extent;
    uint32_t layers;
    uint32_t mip_levels;

    uint64_t images_handle;
    uint64_t allocations_handle;
    uint64_t image_views_handle;
    uint64_t staging_buffers_handle;
    uint64_t staging_allocations_handle;

    uint64_t signals_pointers_handle;

    //std::vector<VkImage> images;
    //std::vector<VmaAllocation> allocations;
    //std::vector<VkImageView> imageViews;
    //std::vector<VkBuffer> stagingBuffers;
    //std::vector<VmaAllocation> stagingAllocations;
    
    uint32_t block_size;

    uint64_t barriers_handle;

    // VkImageMemoryBarrier** barriers;
    
    //std::vector<VkImageMemoryBarrier> barriers;
};

struct Sampler {
    struct Context* ctx;
    uint64_t samplers_handle;

    //std::vector<VkSampler> samplers;
};

#endif // SRC_IMAGE_H_