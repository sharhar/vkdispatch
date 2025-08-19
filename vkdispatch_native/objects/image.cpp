#include "image.hh"
#include "../context/context.hh"
#include "../queue/signal.hh"
#include "objects_extern.hh"
#include "command_list.hh"

#include <vulkan/utility/vk_format_utils.h>
#include <vulkan/vk_enum_string_helper.h>

#include <math.h>

struct Image* image_create_extern(struct Context* context, VkExtent3D a_extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type, unsigned int generate_mips) {
    struct Context* ctx = context;
    VkExtent3D extent = a_extent;

    LOG_INFO("Creating image with extent (%d, %d, %d), layers %d, format %s, type %s, view_type %s, generate_mips %d",
        extent.width, extent.height, extent.depth, layers,
        string_VkFormat((VkFormat)format),
        string_VkImageType((VkImageType)type),
        string_VkImageViewType((VkImageViewType)view_type),
        generate_mips);
    
    
    struct Image* image = new struct Image();
    image->ctx = ctx;
    image->extent = extent;
    image->layers = layers;
    image->mip_levels = 1;
    
    image->images_handle = ctx->handle_manager->register_queue_handle("Image");
    image->allocations_handle = ctx->handle_manager->register_queue_handle("ImageAllocations");
    image->image_views_handle = ctx->handle_manager->register_queue_handle("ImageViews");
    image->staging_buffers_handle = ctx->handle_manager->register_queue_handle("StagingBuffers");
    image->staging_allocations_handle = ctx->handle_manager->register_queue_handle("StagingAllocations");
    image->barriers_handle = ctx->handle_manager->register_queue_handle("Image Barriers");
    image->signals_pointers_handle = ctx->handle_manager->register_queue_handle("Image Signals");

    for(int queue_index = 0; queue_index < ctx->queues.size(); queue_index++) {
        Signal* signal = new Signal();

        LOG_INFO("Creating signal for image with handle %p for queue %d", signal, queue_index);

        ctx->handle_manager->set_handle(queue_index, image->signals_pointers_handle, (uint64_t)signal);

        VkImageMemoryBarrier* barrier = new VkImageMemoryBarrier();
        barrier->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier->pNext = NULL;
        barrier->srcAccessMask = 0;
        barrier->dstAccessMask = 0;
        barrier->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier->newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier->subresourceRange.baseMipLevel = 0;
        barrier->subresourceRange.levelCount = image->mip_levels;
        barrier->subresourceRange.baseArrayLayer = 0;
        barrier->subresourceRange.layerCount = layers;

        ctx->handle_manager->set_handle(queue_index, image->barriers_handle, (uint64_t)barrier);
    }
    
    image->block_size = image_format_block_size_extern(format);

    for(int i = 0; i < ctx->deviceCount; i++) {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(ctx->physicalDevices[i], (VkFormat)format, &formatProperties);

        VkFormatFeatureFlags featureFlagsMask = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                                VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                                VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                                VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

        const auto feats = formatProperties.optimalTilingFeatures;
        if ( (feats & featureFlagsMask) != featureFlagsMask ) {
            const auto missing = featureFlagsMask & ~feats;
            set_error("Format missing features: %s", string_VkFormatFeatureFlags(missing).c_str());
            delete image;            // avoid leak
            return nullptr;
        }
    }

    if(generate_mips) {
        image->mip_levels = (uint32_t)floor(log2(std::max(extent.width, std::max(extent.height, extent.depth))) + 1);
    }

    uint32_t block_size = image->block_size;
    uint32_t mip_levels = image->mip_levels;

    uint64_t images_handle = image->images_handle;
    uint64_t allocations_handle = image->allocations_handle;
    uint64_t image_views_handle = image->image_views_handle;
    uint64_t staging_buffers_handle = image->staging_buffers_handle;
    uint64_t staging_allocations_handle = image->staging_allocations_handle;
    uint64_t barriers_handle = image->barriers_handle;
    uint64_t signals_pointers_handle = image->signals_pointers_handle;
    
    context_submit_command(ctx, "image-init", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, extent, mip_levels, block_size, layers,
            format, type, view_type, images_handle, allocations_handle,
            image_views_handle, staging_buffers_handle, staging_allocations_handle,
            barriers_handle, signals_pointers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            LOG_INFO("Creating image with extent (%d, %d, %d), layers %d, format %s, type %s, view_type %s, generate_mips %d",
                extent.width, extent.height, extent.depth, layers,
                string_VkFormat((VkFormat)format),
                string_VkImageType((VkImageType)type),
                string_VkImageViewType((VkImageViewType)view_type),
                mip_levels);

            VkImageCreateInfo imageCreateInfo;
            memset(&imageCreateInfo, 0, sizeof(VkImageCreateInfo));
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = (VkImageType)type;
            imageCreateInfo.extent = extent;
            imageCreateInfo.mipLevels = mip_levels;
            imageCreateInfo.arrayLayers = layers;
            imageCreateInfo.format = (VkFormat)format;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageCreateInfo.usage = usage;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
            vmaAllocationCreateInfo.flags = 0;
            vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            
            VkImage h_image;
            VmaAllocation h_allocation;
            
            {
                std::unique_lock lock(ctx->vma_mutex);
                VK_CALL(vmaCreateImage(ctx->allocators[indicies.device_index], &imageCreateInfo, &vmaAllocationCreateInfo, &h_image, &h_allocation, NULL));
            }

            // VkImageViewCreateInfo imageViewCreateInfo;
            // memset(&imageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
            // imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            // imageViewCreateInfo.image = h_image;
            // imageViewCreateInfo.viewType = (VkImageViewType)view_type;
            // imageViewCreateInfo.format = (VkFormat)format;
            // imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            // imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
            // imageViewCreateInfo.subresourceRange.levelCount = mip_levels;
            // imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
            // imageViewCreateInfo.subresourceRange.layerCount = layers;
            
            // VkImageView h_image_view;
            // VK_CALL_RETNULL(vkCreateImageView(ctx->devices[indicies.device_index], &imageViewCreateInfo, NULL, &h_image_view));

            // VkBufferCreateInfo bufferCreateInfo;
            // memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
            // bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            // bufferCreateInfo.size = (VkDeviceSize)layers * 
            //                         (VkDeviceSize) extent.width * 
            //                         (VkDeviceSize) extent.height * 
            //                         (VkDeviceSize) extent.depth * 
            //                         (VkDeviceSize) block_size;
            // bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            // bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            // vmaAllocationCreateInfo = {};
            // vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            // vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

            // VkBuffer h_staging_buffer;
            // VmaAllocation h_staging_allocation;

            // {
            //     std::unique_lock lock(ctx->vma_mutex);
            //     VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[indicies.device_index], &bufferCreateInfo, &vmaAllocationCreateInfo, &h_staging_buffer, &h_staging_allocation, NULL));
            // }

            // VkImageMemoryBarrier* barrier = (VkImageMemoryBarrier*)ctx->handle_manager->get_handle(indicies.queue_index, barriers_handle, 0);
            // barrier->image = h_image;

            // ctx->handle_manager->set_handle(indicies.queue_index, images_handle, (uint64_t)h_image);
            // ctx->handle_manager->set_handle(indicies.queue_index, allocations_handle, (uint64_t)h_allocation);
            // ctx->handle_manager->set_handle(indicies.queue_index, image_views_handle, (uint64_t)h_image_view);
            // ctx->handle_manager->set_handle(indicies.queue_index, staging_buffers_handle, (uint64_t)h_staging_buffer);
            // ctx->handle_manager->set_handle(indicies.queue_index, staging_allocations_handle, (uint64_t)h_staging_allocation);

            // Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
            // signal->notify();
        }
    );
    
    return image;
}

void image_destroy_extern(struct Image* image) {

    struct Context* ctx = image->ctx;

    //uint64_t fences_handle = plan->fences_handle;
    //uint64_t vkfft_applications_handle = plan->vkfft_applications_handle;

    // context_submit_command(ctx, "image-destroy", -2, NULL, RECORD_TYPE_SYNC,
    //     [ctx]
    //     (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
    //         //int app_index = indicies.queue_index * recorder_count + j;

    //         //uint64_t vkfft_timestamp = ctx->handle_manager->get_handle_timestamp(app_index, vkfft_applications_handle);
    //         //ctx->queues[indicies.queue_index]->wait_for_timestamp(vkfft_timestamp);

    //         //VkFence fence = (VkFence)ctx->handle_manager->get_handle(app_index, fences_handle, 0);
    //         //vkDestroyFence(ctx->devices[indicies.device_index], fence, NULL);

    //         //VkFFTApplication* application = (VkFFTApplication*)ctx->handle_manager->get_handle(app_index, vkfft_applications_handle, 0);
    //         //deleteVkFFT(application);


    //         //ctx->handle_manager->destroy_handle(app_index, fences_handle);
    //         //ctx->handle_manager->destroy_handle(app_index, vkfft_applications_handle);
    //     }
    // );

    //delete image;

    // LOG_WARNING("Destroying image with handle %p", image);
    
    // for (int i = 0; i < image->images.size(); i++) {
    //     int device_index = image->ctx->queues[i]->device_index;

    //     vkDestroyImageView(image->ctx->devices[device_index], image->imageViews[i], NULL);
    //     vmaDestroyImage(image->ctx->allocators[device_index], image->images[i], image->allocations[i]);
    //     vmaDestroyBuffer(image->ctx->allocators[device_index], image->stagingBuffers[i], image->stagingAllocations[i]);
    // }

    // delete image;
}

struct Sampler* image_create_sampler_extern(struct Context* ctx, 
        unsigned int mag_filter, 
        unsigned int min_filter, 
        unsigned int mip_mode, 
        unsigned int address_mode,
        float mip_lod_bias, 
        float min_lod, 
        float max_lod,
        unsigned int border_color) {

    struct Sampler* sampler = new struct Sampler();
    sampler->ctx = ctx;
    sampler->samplers_handle = ctx->handle_manager->register_queue_handle("Sampler");

    uint64_t samplers_handle = sampler->samplers_handle;
    
    context_submit_command(ctx, "sampler-init", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, samplers_handle, mag_filter, mip_mode, address_mode, mip_lod_bias, min_lod, max_lod, border_color]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
        VkSamplerCreateInfo samplerCreateInfo;
        memset(&samplerCreateInfo, 0, sizeof(VkSamplerCreateInfo));
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = (VkFilter)mag_filter;
        samplerCreateInfo.minFilter = (VkFilter)mag_filter;
        samplerCreateInfo.mipmapMode = (VkSamplerMipmapMode)mip_mode;
        samplerCreateInfo.addressModeU = (VkSamplerAddressMode)address_mode;
        samplerCreateInfo.addressModeV = (VkSamplerAddressMode)address_mode;
        samplerCreateInfo.addressModeW = (VkSamplerAddressMode)address_mode;
        samplerCreateInfo.mipLodBias = mip_lod_bias;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerCreateInfo.minLod = min_lod;
        samplerCreateInfo.maxLod = max_lod;
        samplerCreateInfo.borderColor = (VkBorderColor)border_color;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler sampler;
        VK_CALL(vkCreateSampler(ctx->devices[indicies.device_index], &samplerCreateInfo, NULL, &sampler));
        ctx->handle_manager->set_handle(indicies.queue_index, samplers_handle, (uint64_t)sampler);
    });

    return sampler;
}

void image_destroy_sampler_extern(struct Sampler* sampler) {
    struct Context* ctx = sampler->ctx;
    uint64_t samplers_handle = sampler->samplers_handle;

    context_submit_command(ctx, "sampler-destroy", -2, NULL, RECORD_TYPE_SYNC,
        [ctx, samplers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            uint64_t sampler_timestamp = ctx->handle_manager->get_handle_timestamp(indicies.queue_index, samplers_handle);
            ctx->queues[indicies.queue_index]->wait_for_timestamp(sampler_timestamp);

            VkSampler sampler = (VkSampler)ctx->handle_manager->get_handle(indicies.queue_index, samplers_handle, 0);
            vkDestroySampler(ctx->devices[indicies.device_index], sampler, NULL);
            
            ctx->handle_manager->destroy_handle(indicies.queue_index, samplers_handle);
        }
    );

    delete sampler;
}

static void insert_barrier(VkCommandBuffer cmd_buffer, VkImageMemoryBarrier* barrier, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask) {
    vkCmdPipelineBarrier(cmd_buffer, srcStageMask, dstStageMask, 0, 0, NULL, 0, NULL, 1, barrier);
            
    barrier->srcAccessMask = barrier->dstAccessMask;
    barrier->oldLayout = barrier->newLayout;
}

unsigned int image_format_block_size_extern(unsigned int format) {
    struct VKU_FORMAT_INFO formatInfo = vkuGetFormatInfo((VkFormat)format);
    return formatInfo.block_size;
}

void write_to_image(struct Context* ctx, struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int queue_index) {
    int device_index = ctx->queues[queue_index]->device_index;

    size_t data_size = extent.width * extent.height * extent.depth * layerCount * image->block_size;

    uint64_t signals_pointers_handle = image->signals_pointers_handle;
    Signal* signal = (Signal*)ctx->handle_manager->get_handle(queue_index, signals_pointers_handle, 0);

    LOG_INFO("waiting for recording thread to finish for image %p signal %p queue %d", image, signal, queue_index);
    
    // wait for the recording thread to finish
    signal->wait();
    signal->reset();

    LOG_INFO(
        "Writing data to image %p at offset (%d, %d, %d) with extent (%d, %d, %d), baseLayer %d, layerCount %d",
        image, offset.x, offset.y, offset.z, extent.width, extent.height, extent.depth, baseLayer, layerCount
    );

    // wait for the staging buffer to be ready
    uint64_t staging_buffer_timestamp = ctx->handle_manager->get_handle_timestamp(queue_index, image->staging_buffers_handle);
    ctx->queues[queue_index]->wait_for_timestamp(staging_buffer_timestamp);
    
    LOG_INFO("Staging buffer ready for image %p", image);

    if(data != NULL) {
        VmaAllocation staging_allocation = (VmaAllocation)ctx->handle_manager->get_handle(queue_index, image->staging_allocations_handle, 0);
    
        void* mapped;
        VK_CALL(vmaMapMemory(ctx->allocators[device_index], staging_allocation, &mapped));
        memcpy(mapped, data, data_size);
        vmaUnmapMemory(ctx->allocators[device_index], staging_allocation);
    }

    LOG_INFO("Data written to staging buffer for image %p", image);

    uint64_t images_handle = image->images_handle;
    uint64_t staging_buffers_handle = image->staging_buffers_handle;
    uint64_t barriers_handle = image->barriers_handle;

    uint32_t mip_levels = image->mip_levels;
    uint32_t layers = image->layers;

    context_submit_command(ctx, "image-write", queue_index, NULL, RECORD_TYPE_SYNC,
        [ctx, images_handle, staging_buffers_handle, barriers_handle, offset, extent, baseLayer, layerCount, mip_levels, layers, signals_pointers_handle]
        (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
            LOG_INFO(
                "Writing data to image (%p) at offset (%d, %d, %d) with extent (%d, %d, %d)", 
                images_handle, offset.x, offset.y, 
                offset.z, extent.width, 
                extent.height, extent.depth
            );

            VkImageMemoryBarrier* barrier = (VkImageMemoryBarrier*)ctx->handle_manager->get_handle(indicies.queue_index, barriers_handle, 0);

            VkImage h_image = (VkImage)ctx->handle_manager->get_handle(indicies.queue_index, images_handle, timestamp);
            VkBuffer h_staging_buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, staging_buffers_handle, timestamp);

            VkBufferImageCopy bufferImageCopy;
            memset(&bufferImageCopy, 0, sizeof(VkBufferImageCopy));
            bufferImageCopy.bufferOffset = 0;
            bufferImageCopy.bufferRowLength = 0;
            bufferImageCopy.bufferImageHeight = 0;
            bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferImageCopy.imageSubresource.baseArrayLayer = baseLayer;
            bufferImageCopy.imageSubresource.layerCount = layerCount;
            bufferImageCopy.imageOffset = offset;
            bufferImageCopy.imageExtent = extent;

            if(layerCount > 0) {
                barrier->dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier->newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                vkCmdCopyBufferToImage(cmd_buffer, h_staging_buffer, h_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);
            }

            barrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier->newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            if(mip_levels > 1) {
                barrier->dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                barrier->newLayout = VK_IMAGE_LAYOUT_GENERAL;
                insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                int32_t mipWidth = extent.width;
                int32_t mipHeight = extent.height;
                int32_t mipDepth = extent.depth;

                for(int i = 1; i < mip_levels; i++) {
                    LOG_VERBOSE("Generating mip %d for image %p", i, images_handle);

                    VkImageBlit imageBlit;
                    imageBlit.srcOffsets[0] = { 0, 0, 0 };
                    imageBlit.srcOffsets[1] = { mipWidth, mipHeight, mipDepth };
                    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    imageBlit.srcSubresource.mipLevel = i - 1;
                    imageBlit.srcSubresource.baseArrayLayer = 0;
                    imageBlit.srcSubresource.layerCount = layers;
                    imageBlit.dstOffsets[0] = { 0, 0, 0 };
                    imageBlit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, mipDepth > 1 ? mipDepth / 2 : 1 };
                    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    imageBlit.dstSubresource.mipLevel = i;
                    imageBlit.dstSubresource.baseArrayLayer = 0;
                    imageBlit.dstSubresource.layerCount = layers;

                    vkCmdBlitImage(
                        cmd_buffer,
                        h_image, VK_IMAGE_LAYOUT_GENERAL,
                        h_image, VK_IMAGE_LAYOUT_GENERAL,
                        1, &imageBlit,
                        VK_FILTER_LINEAR
                    );

                    if (mipWidth > 1) mipWidth /= 2;
                    if (mipHeight > 1) mipHeight /= 2;
                    if (mipDepth > 1) mipDepth /= 2;
                }

                barrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier->newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }

            Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
            signal->notify();
        }
    );
}

void image_write_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int queue_index) {
    // LOG_INFO("Writing data to image (%p) at offset (%d, %d, %d) with extent (%d, %d, %d)", image, offset.x, offset.y, offset.z, extent.width, extent.height, extent.depth);

    // struct Context* ctx = image->ctx;

    // if(queue_index != -1) {
    //     write_to_image(ctx, image, data, offset, extent, baseLayer, layerCount, queue_index);
    //     return;
    // }

    // for(int i = 0; i < ctx->queues.size(); i++) {
    //     write_to_image(ctx, image, data, offset, extent, baseLayer, layerCount, i);
    // }
}

void image_read_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int queue_index) {
    // LOG_INFO("Reading data from image (%p) at offset (%d, %d, %d) with extent (%d, %d, %d)", image, offset.x, offset.y, offset.z, extent.width, extent.height, extent.depth);

    // struct Context* ctx = image->ctx;

    // uint64_t signals_pointers_handle = image->signals_pointers_handle;
    // Signal* signal = (Signal*)ctx->handle_manager->get_handle(queue_index, signals_pointers_handle, 0);

    // // wait for the recording thread to finish
    // signal->wait();
    // signal->reset();

    // uint64_t images_handle = image->images_handle;
    // uint64_t staging_buffers_handle = image->staging_buffers_handle;
    // uint64_t barriers_handle = image->barriers_handle;

    // size_t data_size = extent.width * extent.height * extent.depth * layerCount * image->block_size;

    // context_submit_command(ctx, "image-read", queue_index, NULL, RECORD_TYPE_SYNC,
    //     [ctx, offset, staging_buffers_handle, signals_pointers_handle, images_handle, barriers_handle, data_size, baseLayer, layerCount, extent]
    //     (VkCommandBuffer cmd_buffer, ExecIndicies indicies, void* pc_data, BarrierManager* barrier_manager, uint64_t timestamp) {
    //         VkImageMemoryBarrier* barrier = (VkImageMemoryBarrier*)ctx->handle_manager->get_handle(indicies.queue_index, barriers_handle, 0);

    //         VkImage h_image = (VkImage)ctx->handle_manager->get_handle(indicies.queue_index, images_handle, timestamp);
    //         VkBuffer h_staging_buffer = (VkBuffer)ctx->handle_manager->get_handle(indicies.queue_index, staging_buffers_handle, timestamp);
            
    //         VkBufferImageCopy bufferImageCopy;
    //         memset(&bufferImageCopy, 0, sizeof(VkBufferImageCopy));
    //         bufferImageCopy.bufferOffset = 0;
    //         bufferImageCopy.bufferRowLength = 0;
    //         bufferImageCopy.bufferImageHeight = 0;
    //         bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    //         bufferImageCopy.imageSubresource.baseArrayLayer = baseLayer;
    //         bufferImageCopy.imageSubresource.layerCount = layerCount;
    //         bufferImageCopy.imageOffset = offset;
    //         bufferImageCopy.imageExtent = extent;

    //         barrier->dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    //         barrier->newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    //         insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    //         vkCmdCopyImageToBuffer(cmd_buffer, h_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, h_staging_buffer, 1, &bufferImageCopy);

    //         barrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    //         barrier->newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    //         insert_barrier(cmd_buffer, barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    //         Signal* signal = (Signal*)ctx->handle_manager->get_handle(indicies.queue_index, signals_pointers_handle, 0);
    //         signal->notify();
    //     }
    // );

    // signal->wait();

    // // wait for the staging buffer to be ready
    // uint64_t staging_buffer_timestamp = ctx->handle_manager->get_handle_timestamp(queue_index, image->staging_buffers_handle);
    // ctx->queues[queue_index]->wait_for_timestamp(staging_buffer_timestamp);
    
    // int device_index = ctx->queues[queue_index]->device_index;

    // VmaAllocation staging_allocation = (VmaAllocation)ctx->handle_manager->get_handle(queue_index, image->staging_allocations_handle, 0);
    
    // void* mapped;
    // VK_CALL(vmaMapMemory(ctx->allocators[device_index], staging_allocation, &mapped));
    // memcpy(data, mapped, data_size);
    // vmaUnmapMemory(ctx->allocators[device_index], staging_allocation);
}