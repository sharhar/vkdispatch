#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include <VKL/VKL.h>
#include <vkFFT.h>
#include <vector>

#include "base.h"
#include "init.h"

typedef struct {
    VKLInstance instance;
    struct MyPhysicalDeviceProperties* devices;
} MyContext;

extern MyContext _ctx;

struct MyDeviceContext {
    uint32_t deviceCount;
    uint32_t submissionThreadCount;
    VKLDevice* devices;
    const VKLQueue** queues;
};

#endif // BASE_H