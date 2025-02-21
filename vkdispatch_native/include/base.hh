#ifndef SRC_BASE_H
#define SRC_BASE_H

#ifndef VKDISPATCH_USE_VOLK
#include <vulkan/vulkan.h>
#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk/volk.h>

#endif

enum LogLevel {
    LOG_LEVEL_VERBOSE = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3
};

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