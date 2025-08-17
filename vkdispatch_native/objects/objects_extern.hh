#ifndef SRC_OBJECTS_EXTERN_H_
#define SRC_OBJECTS_EXTERN_H_

#include "../base.hh"

struct BufferWriteInfo {
    struct Buffer* buffer;
    unsigned long long offset;
    unsigned long long size;
};

struct BufferReadInfo {
    struct Buffer* buffer;
    unsigned long long offset;
    unsigned long long size;
};

struct ImageWriteInfo {
    struct Image* image;
    VkOffset3D offset;
    VkExtent3D extent;
    unsigned int baseLayer;
    unsigned int layerCount;
};

struct ImageMipMapInfo {
    struct Image* image;
    unsigned int mip_count;   
};

struct ImageReadInfo {
    struct Image* image;
    VkOffset3D offset;
    VkExtent3D extent;
    unsigned int baseLayer;
    unsigned int layerCount;
};

struct Buffer* buffer_create_extern(struct Context* context, unsigned long long size, int per_device);
void buffer_destroy_extern(struct Buffer* buffer);

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index);
void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int index);

struct CommandList* command_list_create_extern(struct Context* context);
void command_list_destroy_extern(struct CommandList* command_list);

unsigned long long command_list_get_instance_size_extern(struct CommandList* command_list);

void command_list_reset_extern(struct CommandList* command_list);
void command_list_submit_extern(struct CommandList* command_list, void* instance_buffer, unsigned int instanceCount, int index, void* signal, int recordType);

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan);
void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set);

void descriptor_set_write_buffer_extern(
    struct DescriptorSet* descriptor_set,
    unsigned int binding,
    void* object,
    unsigned long long offset,
    unsigned long long range,
    int uniform,
    int read_access,
    int write_access
);
void descriptor_set_write_image_extern(
    struct DescriptorSet* descriptor_set,
    unsigned int binding,
    void* object,
    void* sampler_obj,
    int read_access,
    int write_access
);

struct Image* image_create_extern(struct Context* context, VkExtent3D extent, unsigned int layers, unsigned int format, unsigned int type, unsigned int view_type, unsigned int generate_mips);
void image_destroy_extern(struct Image* image);

struct Sampler* image_create_sampler_extern(struct Context* context, 
    unsigned int mag_filter, 
    unsigned int min_filter, 
    unsigned int mip_mode, 
    unsigned int address_mode,
    float mip_lod_bias, 
    float min_lod, 
    float max_lod,
    unsigned int border_color);
void image_destroy_sampler_extern(struct Sampler* sampler);

unsigned int image_format_block_size_extern(unsigned int format);

void image_write_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index);
void image_read_extern(struct Image* image, void* data, VkOffset3D offset, VkExtent3D extent, unsigned int baseLayer, unsigned int layerCount, int device_index);

//void image_copy_extern(struct Image* src, struct Image* dst, VkOffset3D src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, 
//                                                             VkOffset3D dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, VkExtent3D extent, int device_index);



#endif // SRC_OBJECTS_EXTERN_H_