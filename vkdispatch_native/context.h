#ifndef SRC_DEVICE_CONTEXT_H_
#define SRC_DEVICE_CONTEXT_H_

#include "base.h"

struct Context* context_create_extern(int* device_indicies, int* queue_counts, int* queue_families, int device_count);
void context_wait_idle_extern(struct Context* context);
void context_destroy_extern(struct Context* context);

#endif  // SRC_DEVICE_CONTEXT_H_