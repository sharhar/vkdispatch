#include "internal.h"

#include <vulkan/utility/vk_format_utils.h>
#include <vulkan/vk_enum_string_helper.h>

struct Image* image_create_extern(struct Context* context, VkExtent3D extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type) {
    /*
    
    struct Image* image = new struct Image();
    image->ctx = context;
    image->images = (VKLImage**)malloc(sizeof(VKLImage*) * context->deviceCount);
    image->stagingBuffers = (VKLBuffer**)malloc(sizeof(VKLBuffer*) * context->deviceCount);

    VkFormatProperties formatProperties = context->devices[0]->physical()->getFormatProperties((VkFormat)format);

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
        LOG_ERROR("Format does not support required features: %s", string_VkFormatFeatureFlags(unsupportedFeatures).c_str());
        return VK_NULL_HANDLE;
    }

    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    struct VKU_FORMAT_INFO formatInfo = vkuGetFormatInfo((VkFormat)format);

    image->block_size = formatInfo.block_size;

    for (int i = 0; i < context->deviceCount; i++) {
        VkFormatProperties formatProperties = context->devices[i]->physical()->getFormatProperties((VkFormat)format);

        image->images[i] = new VKLImage(
            VKLImageCreateInfo()
            .device(context->devices[i])
            .extent(extent.width, extent.height, extent.depth)
            .arrayLayers(layers)
            .format((VkFormat)format)
            .imageType((VkImageType)type)
            .usage(usage)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
        );

        image->imageViews[i] = new VKLImageView(
            VKLImageViewCreateInfo()
            .image(image->images[i])
            .type((VkImageViewType)view_type)
        );

        image->stagingBuffers[i] = new VKLBuffer(
            VKLBufferCreateInfo()
            .device(context->devices[i])
            .size(layers * extent.width * extent.height * extent.depth * formatInfo.block_size)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
		    .flagsVMA(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
            .usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        );
    }

    return image;
    */

    return NULL;
}

void image_destroy_extern(struct Image* image) {
    /*
    for (int i = 0; i < image->ctx->deviceCount; i++) {
        image->images[i]->destroy();
        delete image->images[i];
        image->stagingBuffers[i]->destroy();
        delete image->stagingBuffers[i];
    }

    free((void*)image->images);
    free((void*)image->stagingBuffers);
    delete image;
    */
}

unsigned int image_format_block_size_extern(unsigned int format) {
    //struct VKU_FORMAT_INFO formatInfo = vkuGetFormatInfo((VkFormat)format);
    //return formatInfo.block_size;

    return 0;
}

void image_write_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index) {
    /*
    
    int enum_count = device_index == -1 ? image->ctx->deviceCount : 1;
    int start_index = device_index == -1 ? 0 : device_index;

    unsigned long long size = extent.width * extent.height * extent.depth * image->block_size;

    for (int i = 0; i < enum_count; i++) {
        int dev_index = start_index + i;

        image->stagingBuffers[dev_index]->setData(data, size, 0);

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

        VKLCommandBuffer* cmdBuffer = image->ctx->queues[dev_index]->getCmdBuffer();

        cmdBuffer->begin();

        image->images[dev_index]->cmdTransitionBarrier(cmdBuffer,
                                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        image->ctx->devices[dev_index]->vk.CmdCopyBufferToImage(
            cmdBuffer->handle(),
            image->stagingBuffers[dev_index]->handle(),
            image->images[dev_index]->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferImageCopy
        );

        cmdBuffer->end();
        image->ctx->queues[dev_index]->submitAndWait(cmdBuffer);
    }
    */
}

void image_read_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index) {
    /*
    int dev_index = device_index == -1 ? 0 : device_index;

    unsigned long long size = extent.width * extent.height * extent.depth * image->block_size;

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
    
    VKLCommandBuffer* cmdBuffer = image->ctx->queues[dev_index]->getCmdBuffer();

    cmdBuffer->begin();

    image->images[dev_index]->cmdTransitionBarrier(cmdBuffer,
                                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    
    image->ctx->devices[dev_index]->vk.CmdCopyImageToBuffer(
        cmdBuffer->handle(),
        image->images[dev_index]->handle(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        image->stagingBuffers[dev_index]->handle(),
        1,
        &bufferImageCopy
    );

    cmdBuffer->end();
    image->ctx->queues[dev_index]->submitAndWait(cmdBuffer);

    image->stagingBuffers[dev_index]->getData(data, size, 0);
    */
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