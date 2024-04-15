#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include "base.h"

struct Image* image_create_extern(struct Context* context, unsigned int width, unsigned int height, unsigned int depth, unsigned int format, unsigned int type);
void image_destroy_extern(struct Image* image);

void image_write_extern(struct Image* image, void* data, unsigned long long offset, unsigned long long size, int device_index);
void image_read_extern(struct Image* image, void* data, unsigned long long offset, unsigned long long size, int device_index);

void image_copy_extern(struct Image* src, struct Image* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index);

#endif // SRC_IMAGE_H_