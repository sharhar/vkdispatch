#ifndef SRC_INIT_H
#define SRC_INIT_H

#include "../base.hh"
#include "context_extern.hh"

#include <vector>

/**
 * @brief A struct that contains information about the Vulkan instance.
 * 
 * This struct contains the handle to the:
 * - Vulkan instance (VkInstance)
 * - Debug messenger (VkDebugUtilsMessengerEXT)
 * - Physical devices (VkPhysicalDevice)
 * - Features of the physical devices (VkPhysicalDeviceFeatures2)
 * - Shader atomic float features (VkPhysicalDeviceShaderAtomicFloatFeaturesEXT)
 * - Shader float16 and int8 features (VkPhysicalDeviceShaderFloat16Int8Features)
 * - 16-bit storage features (VkPhysicalDevice16BitStorageFeatures)
 * - Physical device properties (VkPhysicalDeviceProperties2)
 * - Subgroup properties (VkPhysicalDeviceSubgroupProperties)
 * - Device details (PhysicalDeviceDetails)
 * - Queue family properties (VkQueueFamilyProperties)
 * - Timeline semaphore features (VkPhysicalDeviceTimelineSemaphoreFeatures)
 * 
 * 
 * These handles are primarily used for iterating over the physical devices and their properties
 * so that the program can adapt to the capabilities of the available hardware.
 */
typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::vector<VkPhysicalDeviceFeatures2> features;
    std::vector<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT> atomicFloatFeatures;
    std::vector<VkPhysicalDeviceShaderFloat16Int8Features> float16int8Features;
    std::vector<VkPhysicalDevice16BitStorageFeatures> storage16bit;
    std::vector<VkPhysicalDeviceProperties2> properties;
    std::vector<VkPhysicalDeviceSubgroupProperties> subgroup_properties;
    std::vector<struct PhysicalDeviceDetails> device_details;
    std::vector<std::vector<VkQueueFamilyProperties>> queue_family_properties;
    std::vector<VkPhysicalDeviceTimelineSemaphoreFeatures> timeline_semaphore_features;

    bool debug;

    Context* context;
} Instance;

/**
 * @brief Global instance of MyInstance.
 * 
 * This instance is used to store the Vulkan instance and its related properties.
 * It is initialized during the program startup and used throughout the program.
 */
extern Instance _instance;

#endif // INIT_H