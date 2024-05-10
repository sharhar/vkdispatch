#include "internal.h"
#include <vector>

struct Context* context_create_extern(int* device_indicies, int* submission_thread_couts, int device_count) {
    LOG_INFO("Creating context with %d devices", device_count);

    struct Context* ctx = new struct Context();
    ctx->deviceCount = device_count;

    LOG_INFO("Enumerating physical devices...");

    auto physicalDevices = _instance.instance.enumeratePhysicalDevices();

    for(int i = 0; i < device_count; i++) {
        auto physical = physicalDevices[device_indicies[i]];

        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> featuresChainQuery = {
            vk::PhysicalDeviceFeatures2(),
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT()
        };

        physical.getFeatures2(&featuresChainQuery.get<vk::PhysicalDeviceFeatures2>());

        auto atomicFeatures = featuresChainQuery.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();

        if(!atomicFeatures.shaderBufferFloat32AtomicAdd) {
            LOG_ERROR("Device does not support shaderBufferFloat32AtomicAdd");
            return nullptr;
        }

        ctx->physicalDevices.push_back(physical);

        LOG_INFO("Creating physical device %p...", static_cast<VkPhysicalDevice>(ctx->physicalDevices[i]));

        auto queue_family_properties = ctx->physicalDevices[i].getQueueFamilyProperties();

        int foundIndex = -1;

        for(int queueIndex = 0; queueIndex < queue_family_properties.size(); queueIndex++) {
            LOG_INFO("Queue Index: %d", queueIndex);
            LOG_INFO("Queue Count: %d", queue_family_properties[queueIndex].queueCount);
            LOG_INFO("Queue Flags:  %p", queue_family_properties[queueIndex].queueFlags);

            if(queue_family_properties[queueIndex].queueFlags & vk::QueueFlagBits::eCompute) {
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

        

        vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(foundIndex)
            .setQueueCount(1)
            .setPQueuePriorities(&queue_priority);

        std::vector<const char*> desiredExtensions =  {
            "VK_KHR_shader_non_semantic_info",
            "VK_EXT_shader_atomic_float"
        };

#ifdef __APPLE__
        desiredExtensions.push_back("VK_KHR_portability_subset");
#endif
        
        auto deviceExtensions = ctx->physicalDevices[i].enumerateDeviceExtensionProperties();

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

        vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> deviceCreateChain = {
            vk::DeviceCreateInfo()
                .setQueueCreateInfoCount(1)
                .setPQueueCreateInfos(&queueCreateInfo)
                .setPEnabledExtensionNames(desiredExtensions),
            vk::PhysicalDeviceFeatures2(),
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT()
                .setShaderBufferFloat32AtomicAdd(VK_TRUE)
        };

        ctx->devices.push_back(ctx->physicalDevices[i].createDevice(deviceCreateChain.get<vk::DeviceCreateInfo>()));

        LOG_INFO("Created device %p", static_cast<VkDevice>(ctx->devices[i]));

        ctx->streams.push_back(new Stream(ctx->devices[i], ctx->devices[i].getQueue(foundIndex, 0), foundIndex, 2));
        ctx->submissionThreadCounts.push_back(submission_thread_couts[i]);

        VmaVulkanFunctions vmaVulkanFunctions = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            _instance.instance.getProcAddr("vkGetInstanceProcAddr")
        );
        vmaVulkanFunctions.vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            _instance.instance.getProcAddr("vkGetDeviceProcAddr")
        );

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorCreateInfo.physicalDevice = static_cast<VkPhysicalDevice>(ctx->physicalDevices[i]);
        allocatorCreateInfo.device = static_cast<VkDevice>(ctx->devices[i]);
        allocatorCreateInfo.instance = static_cast<VkInstance>(_instance.instance);
        allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;
        
        VmaAllocator allocator;
        VK_CALL(vmaCreateAllocator(&allocatorCreateInfo, &allocator));
        ctx->allocators.push_back(allocator);

        LOG_INFO("Created allocator %p", ctx->allocators[i]);
    }

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    return ctx;
}

void context_destroy_extern(struct Context* ctx) {
    for (int i = 0; i < ctx->deviceCount; i++) {
        vmaDestroyAllocator(ctx->allocators[i]);
        ctx->streams[i]->destroy();
        delete ctx->streams[i];
        ctx->devices[i].destroy();
    }

    ctx->devices.clear();
    ctx->streams.clear();
    ctx->submissionThreadCounts.clear();
    ctx->allocators.clear();
    
    delete ctx;
}