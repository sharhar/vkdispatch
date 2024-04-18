import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "image.h":
    struct Context
    struct Image
    struct VkOffset3D:
        int x
        int y
        int z
    
    struct VkExtent3D:
        int width
        int height
        int depth

    Image* image_create_extern(Context* context, VkExtent3D extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type)
    void image_destroy_extern(Image* image)

    unsigned int image_format_block_size_extern(unsigned int format)

    void image_write_extern(Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index)
    void image_read_extern(Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index)

    #void image_copy_extern(Image* src, Image* dst, VkOffset3D src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, 
    #                                               VkOffset3D dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, VkExtent3D extent, int device_index)

cpdef inline image_create(unsigned long long context, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type):
    assert len(extent) == 3

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    return <unsigned long long>image_create_extern(<Context*>context, VkExtent3D(width, height, depth), layers, format, type, view_type)

cpdef inline image_destroy(unsigned long long image):
    image_destroy_extern(<Image*>image)

cpdef inline image_write(unsigned long long image, cnp.ndarray data, tuple[int, int, int] offset, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int baseLayer, unsigned int layerCount, int device_index):
    assert len(offset) == 3
    assert len(extent) == 3

    cdef int x = offset[0]
    cdef int y = offset[1]
    cdef int z = offset[2]

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    image_write_extern(<Image*>image, <void*>data.data, VkOffset3D(x, y, z), VkExtent3D(width, height, depth), baseLayer, layerCount, device_index)

cpdef inline unsigned int image_format_block_size(unsigned int format):
    return image_format_block_size_extern(format)

cpdef inline image_read(unsigned long long image, cnp.ndarray data, tuple[int, int, int] offset, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int baseLayer, unsigned int layerCount, int device_index):
    assert len(offset) == 3
    assert len(extent) == 3

    cdef int x = offset[0]
    cdef int y = offset[1]
    cdef int z = offset[2]

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    image_read_extern(<Image*>image, <void*>data.data, VkOffset3D(x, y, z), VkExtent3D(width, height, depth), baseLayer, layerCount, device_index)

#cpdef inline image_copy(unsigned long long src, unsigned long long dst, list[int] src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, list[int] dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, list[unsigned int] extent, int device_index):
#    assert len(src_offset) == 3
#    assert len(dst_offset) == 3
#    assert len(extent) == 3

#    cdef int src_x = src_offset[0]
#    cdef int src_y = src_offset[1]
#    cdef int src_z = src_offset[2]

#    cdef int dst_x = dst_offset[0]
#    cdef int dst_y = dst_offset[1]
#    cdef int dst_z = dst_offset[2]

#    cdef unsigned int width = extent[0]
#    cdef unsigned int height = extent[1]
#    cdef unsigned int depth = extent[2]

#    image_copy_extern(<Image*>src, <Image*>dst, VkOffset3D(src_x, src_y, src_z), src_baseLayer, src_layerCount, VkOffset3D(dst_x, dst_y, dst_z), dst_baseLayer, dst_layerCount, VkExtent3D(width, height, depth), device_index)
