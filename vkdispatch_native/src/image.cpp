#include "../include/internal.hh"

#include <vulkan/utility/vk_format_utils.h>
#include <vulkan/vk_enum_string_helper.h>

#include <math.h>

struct Image* image_create_extern(struct Context* ctx, VkExtent3D extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type, unsigned int generate_mips) {
    struct Image* image = new struct Image();
    image->ctx = ctx;
    image->extent = extent;
    image->layers = layers;
    image->mip_levels = 1;
    image->images.resize(ctx->stream_indicies.size());
    image->allocations.resize(ctx->stream_indicies.size());
    image->imageViews.resize(ctx->stream_indicies.size());
    image->stagingBuffers.resize(ctx->stream_indicies.size());
    image->stagingAllocations.resize(ctx->stream_indicies.size());
    image->barriers.resize(ctx->stream_indicies.size());
    image->block_size = image_format_block_size_extern(format);

    for(int i = 0; i < ctx->deviceCount; i++) {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(ctx->physicalDevices[0], (VkFormat)format, &formatProperties);

        VkFormatFeatureFlags featureFlagsMask = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                                VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                                VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                                VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;


        if(formatProperties.optimalTilingFeatures & featureFlagsMask == 0) {
            VkFormatFeatureFlags unsupportedFeatures = (formatProperties.optimalTilingFeatures ^ featureFlagsMask) & featureFlagsMask;
            set_error("Format does not support required features: %s", string_VkFormatFeatureFlags(unsupportedFeatures).c_str());
            return VK_NULL_HANDLE;
        }
    }


    if(generate_mips) {
        image->mip_levels = (uint32_t)floor(log2(std::max(extent.width, std::max(extent.height, extent.depth))) + 1);
    }

    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    for (int i = 0; i < ctx->stream_indicies.size(); i++) {
        VkImageCreateInfo imageCreateInfo;
        memset(&imageCreateInfo, 0, sizeof(VkImageCreateInfo));
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = (VkImageType)type;
        imageCreateInfo.extent = extent;
        imageCreateInfo.mipLevels = image->mip_levels;
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
        VK_CALL_RETNULL(vmaCreateImage(ctx->allocators[ctx->stream_indicies[i].first], &imageCreateInfo, &vmaAllocationCreateInfo, &image->images[i], &image->allocations[i], NULL));

        VkImageViewCreateInfo imageViewCreateInfo;
        memset(&imageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.image = image->images[i];
        imageViewCreateInfo.viewType = (VkImageViewType)view_type;
        imageViewCreateInfo.format = (VkFormat)format;
        imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = image->mip_levels;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = layers;

        VK_CALL_RETNULL(vkCreateImageView(ctx->devices[ctx->stream_indicies[i].first], &imageViewCreateInfo, NULL, &image->imageViews[i]));

        VkBufferCreateInfo bufferCreateInfo;
        memset(&bufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = layers * extent.width * extent.height * extent.depth * image->block_size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        vmaAllocationCreateInfo = {};
		vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        VK_CALL_RETNULL(vmaCreateBuffer(ctx->allocators[ctx->stream_indicies[i].first], &bufferCreateInfo, &vmaAllocationCreateInfo, &image->stagingBuffers[i], &image->stagingAllocations[i], NULL));
        
        memset(&image->barriers[i], 0, sizeof(VkImageMemoryBarrier));
        image->barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        image->barriers[i].pNext = NULL;
        image->barriers[i].srcAccessMask = 0;
        image->barriers[i].dstAccessMask = 0;
        image->barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image->barriers[i].newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image->barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image->barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image->barriers[i].image = image->images[i];
        image->barriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image->barriers[i].subresourceRange.baseMipLevel = 0;
        image->barriers[i].subresourceRange.levelCount = image->mip_levels;
        image->barriers[i].subresourceRange.baseArrayLayer = 0;
        image->barriers[i].subresourceRange.layerCount = layers;
    }

    image_write_extern(image, NULL, {0, 0, 0}, extent, 0, 0, -1);
    
    return image;
}

void image_destroy_extern(struct Image* image) {
    for (int i = 0; i < image->images.size(); i++) {
        vkDestroyImageView(image->ctx->devices[image->ctx->stream_indicies[i].first], image->imageViews[i], NULL);
        vmaDestroyImage(image->ctx->allocators[image->ctx->stream_indicies[i].first], image->images[i], image->allocations[i]);
        vmaDestroyBuffer(image->ctx->allocators[image->ctx->stream_indicies[i].first], image->stagingBuffers[i], image->stagingAllocations[i]);
    }

    delete image;
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
    sampler->samplers.resize(ctx->stream_indicies.size());

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

    for (int i = 0; i < ctx->stream_indicies.size(); i++) {
        VK_CALL_RETNULL(vkCreateSampler(ctx->devices[ctx->stream_indicies[i].first], &samplerCreateInfo, NULL, &sampler->samplers[i]));
    }

    return sampler;
}

void image_destroy_sampler_extern(struct Sampler* sampler) {
    for (int i = 0; i < sampler->samplers.size(); i++) {
        vkDestroySampler(sampler->ctx->devices[sampler->ctx->stream_indicies[i].first], sampler->samplers[i], NULL);
    }

    delete sampler;
}

unsigned int image_format_block_size_extern(unsigned int format) {
    struct VKU_FORMAT_INFO formatInfo = vkuGetFormatInfo((VkFormat)format);
    return formatInfo.block_size;
}

void image_memory_barrier_internal(struct Image* image, int stream_index, VkCommandBuffer command_buffer, VkAccessFlags accessMask, VkImageLayout layout, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask) {
	image->barriers[stream_index].dstAccessMask = accessMask;
	image->barriers[stream_index].newLayout = layout;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, NULL, 0, NULL, 1, &image->barriers[stream_index]);
	
	image->barriers[stream_index].srcAccessMask = image->barriers[stream_index].dstAccessMask;
	image->barriers[stream_index].oldLayout = image->barriers[stream_index].newLayout;
}

void image_write_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int index) {
    LOG_INFO("Writing data to image (%p) at offset (%d, %d, %d) with extent (%d, %d, %d)", image, offset.x, offset.y, offset.z, extent.width, extent.height, extent.depth);

    struct Context* ctx = image->ctx;

    int enum_count = index == -1 ? image->images.size() : 1;
    int start_index = index == -1 ? 0 : index;

    size_t data_size = extent.width * extent.height * extent.depth * layerCount * image->block_size;

    Signal* signals = new Signal[enum_count];

    for (int i = 0; i < enum_count; i++) {
        int buffer_index = start_index + i;

        LOG_INFO("Writing data to buffer %d", buffer_index);

        int device_index = 0;

        auto stream_index = ctx->stream_indicies[buffer_index];
        device_index = stream_index.first;

        LOG_INFO("Writing data to buffer %d in device %d", buffer_index, device_index);

        if(data != NULL) {
            void* mapped;
            VK_CALL(vmaMapMemory(ctx->allocators[device_index], image->stagingAllocations[buffer_index], &mapped));
            memcpy(mapped, data, data_size);
            vmaUnmapMemory(ctx->allocators[device_index], image->stagingAllocations[buffer_index]);
        }

        struct CommandInfo command = {};
        command.type = COMMAND_TYPE_IMAGE_WRITE;
        command.pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        command.info.image_write_info.image = image;
        command.info.image_write_info.offset = offset;
        command.info.image_write_info.extent = extent;
        command.info.image_write_info.baseLayer = baseLayer;
        command.info.image_write_info.layerCount = layerCount;

        command_list_record_command(ctx->command_list, command);
        

        if(image->mip_levels > 1) {
            struct CommandInfo command = {};
            command.type = COMMAND_TYPE_IMAGE_MIP_MAP;
            command.pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            command.info.image_mip_map_info.image = image;
            command.info.image_mip_map_info.mip_count = image->mip_levels;

            command_list_record_command(ctx->command_list, command);
        }

        command_list_submit_extern(ctx->command_list, NULL, 1, &buffer_index, 1, &signals[i]);
        command_list_reset_extern(ctx->command_list);
        RETURN_ON_ERROR(;)
    }

    for (int i = 0; i < enum_count; i++) {
        signals[i].wait();
    }

    delete[] signals;
}

void image_write_exec_internal(VkCommandBuffer cmd_buffer, const struct ImageWriteInfo& info, int device_index, int stream_index) {
    VkBufferImageCopy bufferImageCopy;
    memset(&bufferImageCopy, 0, sizeof(VkBufferImageCopy));
    bufferImageCopy.bufferOffset = 0;
    bufferImageCopy.bufferRowLength = 0;
    bufferImageCopy.bufferImageHeight = 0;
    bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferImageCopy.imageSubresource.baseArrayLayer = info.baseLayer;
    bufferImageCopy.imageSubresource.layerCount = info.layerCount;
    bufferImageCopy.imageOffset = info.offset;
    bufferImageCopy.imageExtent = info.extent;

    if(info.layerCount > 0) {
        image_memory_barrier_internal(
            info.image,
            stream_index, 
            cmd_buffer, 
            VK_ACCESS_TRANSFER_WRITE_BIT, 
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
            VK_PIPELINE_STAGE_TRANSFER_BIT
        );

        vkCmdCopyBufferToImage(cmd_buffer, info.image->stagingBuffers[stream_index], info.image->images[stream_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);
    }

    image_memory_barrier_internal(
        info.image,
        stream_index, 
        cmd_buffer, 
        VK_ACCESS_SHADER_READ_BIT, 
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
        VK_PIPELINE_STAGE_TRANSFER_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
}

void image_generate_mipmaps_internal(VkCommandBuffer cmd_buffer, const struct ImageMipMapInfo& info, int device_index, int stream_index) {
    LOG_VERBOSE("Generating mipmaps for image %p", info.image);

    image_memory_barrier_internal(
        info.image,
        stream_index, 
        cmd_buffer, 
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, 
        VK_IMAGE_LAYOUT_GENERAL, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        VK_PIPELINE_STAGE_TRANSFER_BIT
    );

    LOG_VERBOSE("Did barrier for image %p", info.image);

    int32_t mipWidth = info.image->extent.width;
    int32_t mipHeight = info.image->extent.height;
    int32_t mipDepth = info.image->extent.depth;

    for(int i = 1; i < info.mip_count; i++) {
        LOG_VERBOSE("Generating mip %d for image %p", i, info.image);

        image_memory_barrier_internal(
            info.image,
            stream_index, 
            cmd_buffer, 
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, 
            VK_IMAGE_LAYOUT_GENERAL, 
            VK_PIPELINE_STAGE_TRANSFER_BIT, 
            VK_PIPELINE_STAGE_TRANSFER_BIT
        );

        LOG_VERBOSE("Did barrier for mip %d image %p", i, info.image);

        VkImageBlit imageBlit;
        imageBlit.srcOffsets[0] = { 0, 0, 0 };
        imageBlit.srcOffsets[1] = { mipWidth, mipHeight, mipDepth };
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = i - 1;
        imageBlit.srcSubresource.baseArrayLayer = 0;
        imageBlit.srcSubresource.layerCount = info.image->layers;
        imageBlit.dstOffsets[0] = { 0, 0, 0 };
        imageBlit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, mipDepth > 1 ? mipDepth / 2 : 1 };
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = i;
        imageBlit.dstSubresource.baseArrayLayer = 0;
        imageBlit.dstSubresource.layerCount = info.image->layers;

        vkCmdBlitImage(
            cmd_buffer,
            info.image->images[stream_index], VK_IMAGE_LAYOUT_GENERAL,
            info.image->images[stream_index], VK_IMAGE_LAYOUT_GENERAL,
            1, &imageBlit,
            VK_FILTER_LINEAR
        );

        LOG_VERBOSE("Did blit for mip %d image %p", i, info.image);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
        if (mipDepth > 1) mipDepth /= 2;
    }
    
    image_memory_barrier_internal(
        info.image,
        stream_index, 
        cmd_buffer, 
        VK_ACCESS_SHADER_READ_BIT, 
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
        VK_PIPELINE_STAGE_TRANSFER_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    LOG_VERBOSE("Finish image %p", info.image);
}

void image_read_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int index) {
    LOG_INFO("Reading data from image (%p) at offset (%d, %d, %d) with extent (%d, %d, %d)", image, offset.x, offset.y, offset.z, extent.width, extent.height, extent.depth);

    struct Context* ctx = image->ctx;

    int device_index = 0;

    auto stream_index = ctx->stream_indicies[index];
    device_index = stream_index.first;

    size_t data_size = extent.width * extent.height * extent.depth * layerCount * image->block_size;

    struct CommandInfo command = {};
    command.type = COMMAND_TYPE_IMAGE_READ;
    command.pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    command.info.image_read_info.image = image;
    command.info.image_read_info.offset = offset;
    command.info.image_read_info.extent = extent;
    command.info.image_read_info.baseLayer = baseLayer;
    command.info.image_read_info.layerCount = layerCount;

    command_list_record_command(ctx->command_list, command);
    
    Signal signal;
    command_list_submit_extern(ctx->command_list, NULL, 1, &index, 1, &signal);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(;)
    
    LOG_INFO("Waiting for signal");

    signal.wait();

    void* mapped;
    VK_CALL(vmaMapMemory(ctx->allocators[device_index], image->stagingAllocations[index], &mapped));
    memcpy(data, mapped, data_size);
    vmaUnmapMemory(ctx->allocators[device_index], image->stagingAllocations[index]);
}

void image_read_exec_internal(VkCommandBuffer cmd_buffer, const struct ImageReadInfo& info, int device_index, int stream_index) {
    VkBufferImageCopy bufferImageCopy;
    memset(&bufferImageCopy, 0, sizeof(VkBufferImageCopy));
    bufferImageCopy.bufferOffset = 0;
    bufferImageCopy.bufferRowLength = 0;
    bufferImageCopy.bufferImageHeight = 0;
    bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferImageCopy.imageSubresource.baseArrayLayer = info.baseLayer;
    bufferImageCopy.imageSubresource.layerCount = info.layerCount;
    bufferImageCopy.imageOffset = info.offset;
    bufferImageCopy.imageExtent = info.extent;

    image_memory_barrier_internal(
        info.image,
        stream_index, 
        cmd_buffer, 
        VK_ACCESS_TRANSFER_READ_BIT, 
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        VK_PIPELINE_STAGE_TRANSFER_BIT
    );

    vkCmdCopyImageToBuffer(cmd_buffer, info.image->images[stream_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, info.image->stagingBuffers[stream_index], 1, &bufferImageCopy);

    image_memory_barrier_internal(
        info.image,
        stream_index, 
        cmd_buffer, 
        VK_ACCESS_SHADER_READ_BIT, 
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
        VK_PIPELINE_STAGE_TRANSFER_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
}

void image_copy_extern(struct Image* src, struct Image* dst, VkOffset3D src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, 
                                                             VkOffset3D dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, VkExtent3D extent, int device_index) {
    /*
    assert(src->ctx == dst->ctx);

    int enum_count = device_index == -1 ? src->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        VkImageCopy imageCopy;
        memset(&imageCopy, 0, sizeof(VkImageCopy));
        imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopy.srcSubresource.baseArrayLayer = src_baseLayer;
        imageCopy.srcSubresource.layerCount = src_layerCount;
        imageCopy.srcOffset = src_offset;
        imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopy.dstSubresource.baseArrayLayer = dst_baseLayer;
        imageCopy.dstSubresource.layerCount = dst_layerCount;
        imageCopy.dstOffset = dst_offset;
        imageCopy.extent = extent;

        dst->images[dev_index]->copyFrom(src->images[dev_index], src->ctx->queues[dev_index], imageCopy);
    } 
*/
}