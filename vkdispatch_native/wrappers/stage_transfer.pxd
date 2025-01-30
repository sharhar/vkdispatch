# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "../include/stage_transfer.hh":
    struct Context
    struct Buffer
    struct Image
    struct CommandList

    struct VkOffset3D:
        int x
        int y
        int z
    
    struct VkExtent3D:
        int width
        int height
        int depth

    struct BufferCopyInfo:
        Buffer* src
        Buffer* dst
        unsigned long long src_offset
        unsigned long long dst_offset
        unsigned long long size

    struct ImageCopyInfo:
        Image* src
        Image* dst
        VkOffset3D src_offset
        VkOffset3D dst_offset
        VkExtent3D extent
        unsigned int src_baseLayer
        unsigned int src_layerCount
        unsigned int dst_baseLayer
        unsigned int dst_layerCount

    struct ImageBufferCopyInfo:
        Image* image
        Buffer* buffer
        VkOffset3D image_offset
        unsigned long long buffer_offset
        unsigned long long buffer_row_length
        unsigned long long buffer_image_height
        VkExtent3D extent
        unsigned int image_baseLayer
        unsigned int image_layerCount

    void stage_transfer_record_copy_buffer_extern(CommandList* command_list, BufferCopyInfo* copy_info)
    void stage_transfer_record_copy_image_extern(CommandList* command_list, ImageCopyInfo* copy_info)
    void stage_transfer_record_copy_buffer_to_image_extern(CommandList* command_list, ImageBufferCopyInfo* copy_info)
    void stage_transfer_record_copy_image_to_buffer_extern(CommandList* command_list, ImageBufferCopyInfo* copy_info)


cpdef inline stage_transfer_record_copy_buffer(unsigned long long command_list, unsigned long long src, unsigned long long dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size):
    cdef BufferCopyInfo copy_info
    copy_info.src = <Buffer*>src
    copy_info.dst = <Buffer*>dst
    copy_info.src_offset = src_offset
    copy_info.dst_offset = dst_offset
    copy_info.size = size

    stage_transfer_record_copy_buffer_extern(<CommandList*>command_list, &copy_info)

cpdef inline stage_transfer_record_copy_image(unsigned long long command_list, unsigned long long src, unsigned long long dst, tuple[int, int, int] src_offset, tuple[int, int, int] dst_offset, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int src_baseLayer, unsigned int src_layerCount, unsigned int dst_baseLayer, unsigned int dst_layerCount):
    cdef ImageCopyInfo copy_info
    copy_info.src = <Image*>src
    copy_info.dst = <Image*>dst
    copy_info.src_offset.x = src_offset[0]
    copy_info.src_offset.y = src_offset[1]
    copy_info.src_offset.z = src_offset[2]
    copy_info.dst_offset.x = dst_offset[0]
    copy_info.dst_offset.y = dst_offset[1]
    copy_info.dst_offset.z = dst_offset[2]
    copy_info.extent.width = extent[0]
    copy_info.extent.height = extent[1]
    copy_info.extent.depth = extent[2]
    copy_info.src_baseLayer = src_baseLayer
    copy_info.src_layerCount = src_layerCount
    copy_info.dst_baseLayer = dst_baseLayer
    copy_info.dst_layerCount = dst_layerCount

    stage_transfer_record_copy_image_extern(<CommandList*>command_list, &copy_info)

cpdef inline stage_transfer_record_copy_buffer_to_image(unsigned long long command_list, unsigned long long image, unsigned long long buffer, tuple[int, int, int] image_offset, unsigned long long buffer_offset, unsigned long long buffer_row_length, unsigned long long buffer_image_height, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int image_baseLayer, unsigned int image_layerCount):
    cdef ImageBufferCopyInfo copy_info
    copy_info.image = <Image*>image
    copy_info.buffer = <Buffer*>buffer
    copy_info.image_offset.x = image_offset[0]
    copy_info.image_offset.y = image_offset[1]
    copy_info.image_offset.z = image_offset[2]
    copy_info.buffer_offset = buffer_offset
    copy_info.buffer_row_length = buffer_row_length
    copy_info.buffer_image_height = buffer_image_height
    copy_info.extent.width = extent[0]
    copy_info.extent.height = extent[1]
    copy_info.extent.depth = extent[2]
    copy_info.image_baseLayer = image_baseLayer
    copy_info.image_layerCount = image_layerCount

    stage_transfer_record_copy_buffer_to_image_extern(<CommandList*>command_list, &copy_info)

cpdef inline stage_transfer_record_copy_image_to_buffer(unsigned long long command_list, unsigned long long image, unsigned long long buffer, tuple[int, int, int] image_offset, unsigned long long buffer_offset, unsigned long long buffer_row_length, unsigned long long buffer_image_height, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int image_baseLayer, unsigned int image_layerCount):
    cdef ImageBufferCopyInfo copy_info
    copy_info.image = <Image*>image
    copy_info.buffer = <Buffer*>buffer
    copy_info.image_offset.x = image_offset[0]
    copy_info.image_offset.y = image_offset[1]
    copy_info.image_offset.z = image_offset[2]
    copy_info.buffer_offset = buffer_offset
    copy_info.buffer_row_length = buffer_row_length
    copy_info.buffer_image_height = buffer_image_height
    copy_info.extent.width = extent[0]
    copy_info.extent.height = extent[1]
    copy_info.extent.depth = extent[2]
    copy_info.image_baseLayer = image_baseLayer
    copy_info.image_layerCount = image_layerCount

    stage_transfer_record_copy_image_to_buffer_extern(<CommandList*>command_list, &copy_info)

