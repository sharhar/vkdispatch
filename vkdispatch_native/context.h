#ifndef SRC_DEVICE_CONTEXT_H_
#define SRC_DEVICE_CONTEXT_H_

#include "base.h"

struct Context* context_create_extern(int* device_indicies, int* submission_thread_couts, int device_count);
void context_destroy_extern(struct Context* device_context);

#endif  // SRC_DEVICE_CONTEXT_H_