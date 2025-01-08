#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include "base.hh"

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

void image_write_exec_internal(VkCommandBuffer cmd_buffer, const struct ImageWriteInfo& info, int device_index, int stream_index);
void image_generate_mipmaps_internal(VkCommandBuffer cmd_buffer, const struct ImageMipMapInfo& info, int device_index, int stream_index);
void image_read_exec_internal(VkCommandBuffer cmd_buffer, const struct ImageReadInfo& info, int device_index, int stream_index);

//void image_copy_extern(struct Image* src, struct Image* dst, VkOffset3D src_offset, unsigned int src_baseLayer, unsigned int src_layerCount, 
//                                                             VkOffset3D dst_offset, unsigned int dst_baseLayer, unsigned int dst_layerCount, VkExtent3D extent, int device_index);


#endif // SRC_IMAGE_H_