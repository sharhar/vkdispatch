#ifndef SRC_DEVICE_CONTEXT_H_
#define SRC_DEVICE_CONTEXT_H_

#include "base.h"

DeviceContext* create_device_context_extern(int* device_indicies, int* submission_thread_couts, int device_count);
void destroy_device_context_extern(DeviceContext* device_context);

#endif  // SRC_DEVICE_CONTEXT_H_