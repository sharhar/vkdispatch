#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

#include "base.h"

struct Buffer* create_buffer_extern(struct DeviceContext* device_context, unsigned long long size);
void destroy_buffer_extern(struct Buffer* buffer);

#endif // SRC_BUFFER_H_