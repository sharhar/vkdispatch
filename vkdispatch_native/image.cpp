#include "internal.h"

struct Image* image_create_extern(struct Context* context, unsigned int width, unsigned int height, unsigned int depth, unsigned int format, unsigned int type) {
    struct Image* image = new struct Image();
    image->ctx = context;
    image->images = (VKLImage**)malloc(sizeof(VKLImage*) * context->deviceCount);
    image->stagingImages = (VKLImage**)malloc(sizeof(VKLImage*) * context->deviceCount);

    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    for (int i = 0; i < context->deviceCount; i++) {
        image->images[i] = new VKLImage(
            VKLImageCreateInfo()
            .device(context->devices[i])
            .extent(width, height, depth)
            .format((VkFormat)format)
            .imageType((VkImageType)type)
            .usage(usage)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
        );

        image->stagingImages[i] = new VKLImage(
            VKLImageCreateInfo()
            .device(context->devices[i])
            .extent(width, height, depth)
            .format((VkFormat)format)
            .imageType((VkImageType)type)
            .usage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .usageVMA(VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
		    .flagsVMA(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
        );
    }

    return image;
}