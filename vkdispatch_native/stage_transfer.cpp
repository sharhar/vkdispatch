#include "internal.h"

void stage_transfer_record_copy_buffer_extern(struct CommandList* command_list, struct BufferCopyInfo* copy_info) {
    command_list_append_command(command_list, [copy_info](VkDevice device, VkCommandBuffer cmd_buffer, int stream_index) {
        VkBufferCopy bufferCopy = {};
        bufferCopy.srcOffset = copy_info->src_offset;
        bufferCopy.dstOffset = copy_info->dst_offset;
        bufferCopy.size = copy_info->size;

        vkCmdCopyBuffer(
            cmd_buffer, 
            copy_info->src->buffers[stream_index], 
            copy_info->dst->buffers[stream_index], 
            1, &bufferCopy
        );
    });
}

void stage_transfer_record_copy_image_extern(struct CommandList* command_list, struct ImageCopyInfo* copy_info) {

    /*
    struct ImageCopyInfo* my_copy_info = (struct ImageCopyInfo*)malloc(sizeof(*my_copy_info));
    memcpy(my_copy_info, copy_info, sizeof(*my_copy_info));

    command_list->stages.push_back({
        [](VKLCommandBuffer* cmd_buffer, struct Stage* stage, void* instanceData, int device) {
            struct ImageCopyInfo* copy_info = (struct ImageCopyInfo*)stage->user_data;

            VkImageCopy imageCopy = {};
            imageCopy.srcOffset = copy_info->src_offset;
            imageCopy.dstOffset = copy_info->dst_offset;
            imageCopy.extent = copy_info->extent;
            imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopy.srcSubresource.baseArrayLayer = copy_info->src_baseLayer;
            imageCopy.srcSubresource.layerCount = copy_info->src_layerCount;
            imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopy.dstSubresource.baseArrayLayer = copy_info->dst_baseLayer;
            imageCopy.dstSubresource.layerCount = copy_info->dst_layerCount;

            copy_info->src->images[device]->cmdTransitionBarrier(cmd_buffer,
                                            VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            
            copy_info->dst->images[device]->cmdTransitionBarrier(cmd_buffer,
                                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            cmd_buffer->copyImage(copy_info->src->images[device], copy_info->dst->images[device], imageCopy);
        },
        my_copy_info,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT
    });
    */
}

void stage_transfer_record_copy_buffer_to_image_extern(struct CommandList* command_list, struct ImageBufferCopyInfo* copy_info) {
    /*
    struct ImageBufferCopyInfo* my_copy_info = (struct ImageBufferCopyInfo*)malloc(sizeof(*my_copy_info));
    memcpy(my_copy_info, copy_info, sizeof(*my_copy_info));

    command_list->stages.push_back({
        [](VKLCommandBuffer* cmd_buffer, struct Stage* stage, void* instanceData, int device) {
            struct ImageBufferCopyInfo* copy_info = (struct ImageBufferCopyInfo*)stage->user_data;

            VkBufferImageCopy bufferImageCopy = {};
            bufferImageCopy.bufferOffset = copy_info->buffer_offset;
            bufferImageCopy.bufferRowLength = copy_info->buffer_row_length;
            bufferImageCopy.bufferImageHeight = copy_info->buffer_image_height;
            bufferImageCopy.imageOffset = copy_info->image_offset;
            bufferImageCopy.imageExtent = copy_info->extent;
            bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferImageCopy.imageSubresource.baseArrayLayer = copy_info->image_baseLayer;
            bufferImageCopy.imageSubresource.layerCount = copy_info->image_layerCount;

            copy_info->image->images[device]->cmdTransitionBarrier(cmd_buffer,
                                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            vkCmdCopyBufferToImage(
                cmd_buffer->handle(),
                copy_info->buffer->buffers[device]->handle(),
                copy_info->image->images[device]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &bufferImageCopy
            );
        },
        my_copy_info,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT
    });
    */
}

void stage_transfer_record_copy_image_to_buffer_extern(struct CommandList* command_list, struct ImageBufferCopyInfo* copy_info) {
    /*

    struct ImageBufferCopyInfo* my_copy_info = (struct ImageBufferCopyInfo*)malloc(sizeof(*my_copy_info));
    memcpy(my_copy_info, copy_info, sizeof(*my_copy_info));

    command_list->stages.push_back({
        [](VKLCommandBuffer* cmd_buffer, struct Stage* stage, void* instanceData, int device) {
            struct ImageBufferCopyInfo* copy_info = (struct ImageBufferCopyInfo*)stage->user_data;

            VkBufferImageCopy bufferImageCopy = {};
            bufferImageCopy.bufferOffset = copy_info->buffer_offset;
            bufferImageCopy.bufferRowLength = copy_info->buffer_row_length;
            bufferImageCopy.bufferImageHeight = copy_info->buffer_image_height;
            bufferImageCopy.imageOffset = copy_info->image_offset;
            bufferImageCopy.imageExtent = copy_info->extent;
            bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferImageCopy.imageSubresource.baseArrayLayer = copy_info->image_baseLayer;
            bufferImageCopy.imageSubresource.layerCount = copy_info->image_layerCount;

            copy_info->image->images[device]->cmdTransitionBarrier(cmd_buffer,
                                            VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            vkCmdCopyImageToBuffer(
                cmd_buffer->handle(),
                copy_info->image->images[device]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                copy_info->buffer->buffers[device]->handle(),
                1,
                &bufferImageCopy
            );
        },
        my_copy_info,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT
    });
    */
}