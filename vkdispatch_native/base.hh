#ifndef SRC_BASE_H
#define SRC_BASE_H

#ifndef VKDISPATCH_USE_VOLK
#include <vulkan/vulkan.h>
#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk/volk.h>

#endif

#include <stdio.h>
#include <cstring>
#include <vector>

#include "log.hh"
#include "errors.hh"

enum RecordType {
    RECORD_TYPE_ASYNC = 0,
    RECORD_TYPE_SYNC = 1,
    RECORD_TYPE_SYNC_GPU = 2
};

struct Context;
struct Buffer;
struct Image;
struct Sampler;
struct Stage;
struct CommandList;
struct FFTPlan;
struct ComputePlan;
struct DescriptorSet;

class Signal;

#endif // BASE_H