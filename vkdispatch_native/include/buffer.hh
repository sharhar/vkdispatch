#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

#include "base.hh"

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

struct Buffer* buffer_create_extern(struct Context* context, unsigned long long size, int per_device);
void buffer_destroy_extern(struct Buffer* buffer);

void buffer_write_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index);
void buffer_read_extern(struct Buffer* buffer, void* data, unsigned long long offset, unsigned long long size, int device_index);

void buffer_write_exec_internal(VkCommandBuffer cmd_buffer, const struct BufferWriteInfo& info, int device_index, int stream_index);
void buffer_read_exec_internal(VkCommandBuffer cmd_buffer, const struct BufferReadInfo& info, int device_index, int stream_index);

#endif // SRC_BUFFER_H_