# distutils: language=c++
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "objects/objects_extern.hh":
    struct Context
    struct Buffer
    struct CommandList
    struct ComputePlan
    struct DescriptorSet
    struct Image
    struct Sampler
    struct VkOffset3D:
        int x
        int y
        int z
    
    struct VkExtent3D:
        int width
        int height
        int depth


    Buffer* buffer_create_extern(Context* context, unsigned long long size, int per_device)
    void buffer_destroy_extern(Buffer* buffer)

    void buffer_write_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index)
    void buffer_read_extern(Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index)

    CommandList* command_list_create_extern(Context* context)
    void command_list_destroy_extern(CommandList* command_list)
    unsigned long long command_list_get_instance_size_extern(CommandList* command_list) 
    void command_list_reset_extern(CommandList* command_list)
    void command_list_submit_extern(CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int index, void* signal, int recordType)

    DescriptorSet* descriptor_set_create_extern(ComputePlan* plan)
    void descriptor_set_destroy_extern(DescriptorSet* descriptor_set)

    void descriptor_set_write_buffer_extern(DescriptorSet* descriptor_set, unsigned int binding, void* object, unsigned long long offset, unsigned long long range, int uniform, int read_access, int write_access)
    void descriptor_set_write_image_extern(DescriptorSet* descriptor_set, unsigned int binding, void* object, void* sampler_obj, int read_access, int write_access)

    Image* image_create_extern(Context* context, VkExtent3D extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type, unsigned int generate_mips)
    void image_destroy_extern(Image* image)

    Sampler* image_create_sampler_extern(Context* context, 
        unsigned int mag_filter, 
        unsigned int min_filter, 
        unsigned int mip_mode, 
        unsigned int address_mode,
        float mip_lod_bias, 
        float min_lod, 
        float max_lod,
        unsigned int border_color)
    
    void image_destroy_sampler_extern(Sampler* sampler)

    unsigned int image_format_block_size_extern(unsigned int format)

    void image_write_extern(Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index)
    void image_read_extern(Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index)

    #void image_copy_extern(Image* src, Image* dst, VkOffset3D src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, 
    #                                               VkOffset3D dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, VkExtent3D extent, int device_index)


cpdef inline buffer_create(unsigned long long context, unsigned long long size, int per_device):
    return <unsigned long long>buffer_create_extern(<Context*>context, size, per_device)

cpdef inline buffer_destroy(unsigned long long buffer):
    buffer_destroy_extern(<Buffer*>buffer)

cpdef inline buffer_write(unsigned long long buffer, bytes data, unsigned long long offset, unsigned long long size, int index):
    cdef const char* data_view = data
    buffer_write_extern(<Buffer*>buffer, <void*>data_view, offset, size, index)

cpdef inline buffer_read(unsigned long long buffer, unsigned long long offset, unsigned long long size, int index):
    cdef bytes data = bytes(size)
    cdef char* data_view = data

    buffer_read_extern(<Buffer*>buffer, <void*>data_view, offset, size, index)

    return data

cpdef inline command_list_create(unsigned long long context):
    return <unsigned long long>command_list_create_extern(<Context*>context)

cpdef inline command_list_destroy(unsigned long long command_list):
    command_list_destroy_extern(<CommandList*>command_list)

cpdef inline command_list_get_instance_size(unsigned long long command_list):
    return command_list_get_instance_size_extern(<CommandList*>command_list)

cpdef inline command_list_reset(unsigned long long command_list):
    command_list_reset_extern(<CommandList*>command_list)

cpdef inline command_list_submit(unsigned long long command_list, bytes data, unsigned int instance_count, int index):
    cdef const char* data_view = NULL
    if data is not None:
        data_view = data

    command_list_submit_extern(<CommandList*>command_list, <void*>data_view, instance_count, index, <void*>0, 0)

cpdef inline descriptor_set_create(unsigned long long plan):
    cdef ComputePlan* p = <ComputePlan*>plan
    return <unsigned long long>descriptor_set_create_extern(p)

cpdef inline descriptor_set_destroy(unsigned long long descriptor_set):
    descriptor_set_destroy_extern(<DescriptorSet*>descriptor_set)

cpdef inline descriptor_set_write_buffer(
    unsigned long long descriptor_set,
    unsigned int binding,
    unsigned long long object,
    unsigned long long offset,
    unsigned long long range,
    int uniform,
    int read_access,
    int write_access):

    cdef DescriptorSet* ds = <DescriptorSet*>descriptor_set
    descriptor_set_write_buffer_extern(ds, binding, <void*>object, offset, range, uniform, read_access, write_access)

cpdef inline descriptor_set_write_image(
    unsigned long long descriptor_set,
    unsigned int binding,
    unsigned long long object,
    unsigned long long sampler_obj,
    int read_access,
    int write_access):

    cdef DescriptorSet* ds = <DescriptorSet*>descriptor_set
    descriptor_set_write_image_extern(ds, binding, <void*>object, <void*>sampler_obj, read_access, write_access)

cpdef inline image_create(unsigned long long context, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type, unsigned int generate_mips):
    assert len(extent) == 3

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    return <unsigned long long>image_create_extern(<Context*>context, VkExtent3D(width, height, depth), layers, format, type, view_type, generate_mips)

cpdef inline image_destroy(unsigned long long image):
    image_destroy_extern(<Image*>image)

cpdef inline image_create_sampler(unsigned long long context, unsigned int mag_filter, unsigned int min_filter, unsigned int mip_mode, unsigned int address_mode, float mip_lod_bias, float min_lod, float max_lod, unsigned int border_color):
    return <unsigned long long>image_create_sampler_extern(<Context*>context, mag_filter, min_filter, mip_mode, address_mode, mip_lod_bias, min_lod, max_lod, border_color)

cpdef inline image_destroy_sampler(unsigned long long sampler):
    image_destroy_sampler_extern(<Sampler*>sampler)

cpdef inline image_write(unsigned long long image, bytes data, tuple[int, int, int] offset, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int baseLayer, unsigned int layerCount, int device_index):
    assert len(offset) == 3
    assert len(extent) == 3

    cdef int x = offset[0]
    cdef int y = offset[1]
    cdef int z = offset[2]

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    cdef char* data_view = data

    image_write_extern(<Image*>image, <void*>data_view, VkOffset3D(x, y, z), VkExtent3D(width, height, depth), baseLayer, layerCount, device_index)

cpdef inline unsigned int image_format_block_size(unsigned int format):
    return image_format_block_size_extern(format)

cpdef inline image_read(unsigned long long image, unsigned long long out_size, tuple[int, int, int] offset, tuple[unsigned int, unsigned int, unsigned int] extent, unsigned int baseLayer, unsigned int layerCount, int device_index):
    assert len(offset) == 3
    assert len(extent) == 3

    cdef int x = offset[0]
    cdef int y = offset[1]
    cdef int z = offset[2]

    cdef unsigned int width = extent[0]
    cdef unsigned int height = extent[1]
    cdef unsigned int depth = extent[2]

    cdef bytes data = bytes(out_size)
    cdef char* data_view = data

    image_read_extern(<Image*>image, <void*>data_view, VkOffset3D(x, y, z), VkExtent3D(width, height, depth), baseLayer, layerCount, device_index)

    return data