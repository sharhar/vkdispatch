#ifndef SRC_BASE_H
#define SRC_BASE_H

#ifndef VKDISPATCH_USE_VOLK
#include <vulkan/vulkan.h>
#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk/volk.h>

#endif

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the printing system
void init_print_system(void);

// Cleanup the printing system
void cleanup_print_system(void);

// Thread-safe printing function that can be called from any thread
void thread_safe_print(const char* message);

#ifdef __cplusplus
}
#endif

enum LogLevel {
    LOG_LEVEL_VERBOSE = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3
};

struct Context;
struct Buffer;
struct Image;
struct Stage;
struct CommandList;
struct FFTPlan;
struct ComputePlan;
struct DescriptorSet;

class Signal;

#endif // BASE_H