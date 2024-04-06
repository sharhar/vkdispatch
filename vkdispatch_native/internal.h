#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include <VKL/VKL.h>
#include <vkFFT.h>
#include <vector>

#include "base.h"
#include "init.h"
#include "device_context.h"
#include "buffer.h"

typedef struct {
    VKLInstance instance;
    struct PhysicalDeviceProperties* devices;
} Context;

extern Context _ctx;

struct DeviceContext {
    uint32_t deviceCount;
    VKLDevice* devices;
    const VKLQueue** queues;
    uint32_t* submissionThreadCounts;
};

struct Buffer {
    struct DeviceContext* ctx;
    VKLBuffer* buffers;
    VKLBuffer* stagingBuffers;
};

#endif // INTERNAL_H