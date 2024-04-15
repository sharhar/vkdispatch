#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include <VKL/VKL.h>
#include <vkFFT.h>
#include <vector>

#include "base.h"
#include "init.h"
#include "context.h"
#include "buffer.h"
#include "image.h"

typedef struct {
    VKLInstance instance;
    struct PhysicalDeviceProperties* devices;
} MyInstance;

extern MyInstance _instance;

struct Context {
    uint32_t deviceCount;
    VKLDevice** devices;
    const VKLQueue** queues;
    uint32_t* submissionThreadCounts;
};

struct Buffer {
    struct Context* ctx;
    VKLBuffer** buffers;
    VKLBuffer** stagingBuffers;
};

struct Image {
    struct Context* ctx;
    VKLImage** images;
    VKLImage** stagingImages;
};

#endif // INTERNAL_H