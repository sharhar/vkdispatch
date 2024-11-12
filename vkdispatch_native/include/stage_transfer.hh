#ifndef SRC_STAGE_TRANSFER_H_
#define SRC_STAGE_TRANSFER_H_

#include "base.hh"

struct BufferCopyInfo {
    struct Buffer* src;
    struct Buffer* dst;
    unsigned long long src_offset;
    unsigned long long dst_offset;
    unsigned long long size;
};

struct ImageCopyInfo {
    struct Image* src;
    struct Image* dst;
    VkOffset3D src_offset;
    VkOffset3D dst_offset;
    VkExtent3D extent;
    unsigned int src_baseLayer;
    unsigned int src_layerCount;
    unsigned int dst_baseLayer;
    unsigned int dst_layerCount;
};

struct ImageBufferCopyInfo {
    struct Image* image;
    struct Buffer* buffer;
    VkOffset3D image_offset;
    unsigned long long buffer_offset;
    unsigned long long buffer_row_length;
    unsigned long long buffer_image_height;
    VkExtent3D extent;
    unsigned int image_baseLayer;
    unsigned int image_layerCount;
};

void stage_transfer_record_copy_buffer_extern(struct CommandList* command_list, struct BufferCopyInfo* copy_info);
void stage_transfer_record_copy_image_extern(struct CommandList* command_list, struct ImageCopyInfo* copy_info);
void stage_transfer_record_copy_buffer_to_image_extern(struct CommandList* command_list, struct ImageBufferCopyInfo* copy_info);
void stage_transfer_record_copy_image_to_buffer_extern(struct CommandList* command_list, struct ImageBufferCopyInfo* copy_info);

void stage_transfer_copy_buffer_exec_internal(VkCommandBuffer cmd_buffer, const struct BufferCopyInfo& info, int device_index, int stream_index);

#endif // SRC_STAGE_TRANSFER_H_