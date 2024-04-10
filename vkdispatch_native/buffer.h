#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

#include "base.h"

struct Buffer* create_buffer_extern(struct DeviceContext* device_context, unsigned long long size);
void destroy_buffer_extern(struct Buffer* buffer);

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index);
void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index);

void buffer_copy_extern(struct Buffer* src, struct Buffer* dst, unsigned long long src_offset, unsigned long long dst_offset, unsigned long long size, int device_index);

#endif // SRC_BUFFER_H_