#include "internal.h"
#include <vector>

struct Context* context_create_extern(int* device_indicies, int* submission_thread_couts, int device_count) {
    LOG_INFO("Creating context with %d devices", device_count);

    struct Context* ctx = new struct Context();
    ctx->deviceCount = device_count;
    ctx->physicalDevices.resize(device_count);
    ctx->devices.resize(device_count);
    ctx->streams.resize(device_count);
    ctx->allocators.resize(device_count);

    LOG_INFO("Enumerating physical devices...");

    uint32_t true_device_count;
    VK_CALL(vkEnumeratePhysicalDevices(_instance.instance, &true_device_count, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices(true_device_count);
    VK_CALL(vkEnumeratePhysicalDevices(_instance.instance, &true_device_count, physicalDevices.data()));

    for(int i = 0; i < device_count; i++) {
        ctx->physicalDevices[i] = physicalDevices[device_indicies[i]];

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = {};
        atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &atomicFloatFeatures;
        vkGetPhysicalDeviceFeatures2(ctx->physicalDevices[i], &features2);

        if(!atomicFloatFeatures.shaderBufferFloat32AtomicAdd) {
            LOG_ERROR("Device does not support shaderBufferFloat32AtomicAdd");
            return nullptr;
        }

        LOG_INFO("Creating physical device %p...", static_cast<VkPhysicalDevice>(ctx->physicalDevices[i]));

        uint32_t queueFamilyCount;
        std::vector<VkQueueFamilyProperties> queue_family_properties;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevices[i], &queueFamilyCount, nullptr);
        queue_family_properties.resize(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevices[i], &queueFamilyCount, queue_family_properties.data());

        int foundIndex = -1;

        for(int queueIndex = 0; queueIndex < queue_family_properties.size(); queueIndex++) {
            LOG_INFO("Queue Index: %d", queueIndex);
            LOG_INFO("Queue Count: %d", queue_family_properties[queueIndex].queueCount);
            LOG_INFO("Queue Flags:  %p", queue_family_properties[queueIndex].queueFlags);

            if(queue_family_properties[queueIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                foundIndex = queueIndex;
                break;
            }
        }

        if(foundIndex == -1) {
            LOG_ERROR("Failed to find queue family index");
            return nullptr;
        }

        LOG_INFO("Queue Family Index: %d", foundIndex);

        float queue_priority = 1.0f;

        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = foundIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queue_priority;

        std::vector<const char*> desiredExtensions =  {
            "VK_KHR_shader_non_semantic_info",
            "VK_EXT_shader_atomic_float"
        };

#ifdef __APPLE__
        desiredExtensions.push_back("VK_KHR_portability_subset");
#endif
        
        uint32_t extensionCount;
        std::vector<VkExtensionProperties> deviceExtensions;
        vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, nullptr);
        deviceExtensions.resize(extensionCount);
        vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, deviceExtensions.data());

        // Check if all desired extensions are supported
        for(auto& desiredExtension : desiredExtensions) {
            bool found = false;

            for(auto& extension : deviceExtensions) {
                if(strcmp(extension.extensionName, desiredExtension) == 0) {
                    found = true;
                    break;
                }
            }

            if(!found) {
                LOG_ERROR("Device does not support extension: %s", desiredExtension);
                return nullptr;
            }
        }

        for(auto& extension : deviceExtensions) {
            LOG_INFO("Device Extension: %s", extension.extensionName);
        }

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeaturesEnableStruct = {};
        atomicFloatFeaturesEnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomicFloatFeaturesEnableStruct.shaderBufferFloat32AtomicAdd = VK_TRUE;

        VkPhysicalDeviceFeatures2 features2EnableStruct = {};
        features2EnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2EnableStruct.pNext = &atomicFloatFeaturesEnableStruct;

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pNext = &features2EnableStruct;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = desiredExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = desiredExtensions.data();

        VK_CALL(vkCreateDevice(ctx->physicalDevices[i], &deviceCreateInfo, nullptr, &ctx->devices[i]));

        LOG_INFO("Created device %p", static_cast<VkDevice>(ctx->devices[i]));

        VkQueue queue;
        vkGetDeviceQueue(ctx->devices[i], foundIndex, 0, &queue);
        ctx->streams[i] = new Stream(ctx->devices[i], queue, foundIndex, 2);

        VmaVulkanFunctions vmaVulkanFunctions = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorCreateInfo.physicalDevice = ctx->physicalDevices[i];
        allocatorCreateInfo.device = ctx->devices[i];
        allocatorCreateInfo.instance = _instance.instance;
        allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;
        VK_CALL(vmaCreateAllocator(&allocatorCreateInfo, &ctx->allocators[i]));

        LOG_INFO("Created allocator %p", ctx->allocators[i]);
    }

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    return ctx;
}

void context_destroy_extern(struct Context* context) {
    for(int i = 0; i < context->deviceCount; i++) {
        context->streams[i]->destroy();
        delete context->streams[i];

        vmaDestroyAllocator(context->allocators[i]);
        vkDestroyDevice(context->devices[i], nullptr);
    }

    delete context;
}