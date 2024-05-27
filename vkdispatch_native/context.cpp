#include "internal.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>

struct Context* context_create_extern(int* device_indicies, int* queue_counts, int* queue_families, int device_count) {
    LOG_INFO("Creating context with %d devices", device_count);

    struct Context* ctx = new struct Context();
    ctx->deviceCount = device_count;
    ctx->physicalDevices.resize(device_count);
    ctx->devices.resize(device_count);
    ctx->streams.resize(device_count);
    ctx->allocators.resize(device_count);

    LOG_INFO("Enumerating physical devices...");

    int queue_index_offset = 0;

    for(int i = 0; i < device_count; i++) {
        LOG_INFO("Device %d Index: %d", i, device_indicies[i]);

        ctx->physicalDevices[i] = _instance.physicalDevices[device_indicies[i]];

        //if(!_instance.atomicFloatFeatures[device_indicies[i]].shaderBufferFloat32AtomicAdd) {
        //    set_error("Device does not support shaderBufferFloat32AtomicAdd");
        //    return nullptr;
        //}

        LOG_INFO("Creating physical device %p...", static_cast<VkPhysicalDevice>(ctx->physicalDevices[i]));

        std::vector<const char*> desiredExtensions =  {
            "VK_KHR_shader_non_semantic_info",
            "VK_EXT_shader_atomic_float"
        };

#ifdef __APPLE__
        desiredExtensions.push_back("VK_KHR_portability_subset");
#endif
        
        uint32_t extensionCount;
        std::vector<VkExtensionProperties> deviceExtensions;
        VK_CALL_RETNULL(vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, nullptr));
        deviceExtensions.resize(extensionCount);
        VK_CALL_RETNULL(vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, deviceExtensions.data()));

        // Check if all desired extensions are supported
        std::vector<const char*> supportedExtensions;
        for(auto& desiredExtension : desiredExtensions) {
            auto it = std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [&](const VkExtensionProperties& prop) {
                return strcmp(desiredExtension, prop.extensionName) == 0;
            });
            if (it != deviceExtensions.end()) {
                LOG_INFO("Device Extension '%s' is supported", desiredExtension);
                supportedExtensions.push_back(desiredExtension);
            } else {
                LOG_WARNING("Extension '%s' is not supported", desiredExtension);
            }
        }

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeaturesEnableStruct = {};
        atomicFloatFeaturesEnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomicFloatFeaturesEnableStruct.shaderBufferFloat32AtomicAdd = VK_TRUE;

        VkPhysicalDeviceFeatures2 features2EnableStruct = {};
        features2EnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2EnableStruct.pNext = &atomicFloatFeaturesEnableStruct;

        float* queue_priorities = (float*)malloc(sizeof(float) * queue_counts[i]);
        for(int j = 0; j < queue_counts[i]; j++) {
            queue_priorities[j] = 1.0f;
        }

        std::unordered_map<int, int> frequencyMap;
        for(int j = 0; j < queue_counts[i]; j++)
            frequencyMap[queue_families[queue_index_offset + j]]++;

        LOG_INFO("Queue Family Count: %d", frequencyMap.size());

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        queueCreateInfos.resize(frequencyMap.size());

        int index = 0;
        for (const auto& queueFamily : frequencyMap) {
            LOG_INFO("Queue Family %d Count: %d", queueFamily.first, queueFamily.second);

            queueCreateInfos[index] = {};
            queueCreateInfos[index].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfos[index].queueFamilyIndex = queueFamily.first;
            queueCreateInfos[index].queueCount = queueFamily.second;
            queueCreateInfos[index].pQueuePriorities = queue_priorities;
            index++;
        }

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pNext = NULL; //&features2EnableStruct;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.enabledExtensionCount = supportedExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = supportedExtensions.data();

        VK_CALL_RETNULL(vkCreateDevice(ctx->physicalDevices[i], &deviceCreateInfo, nullptr, &ctx->devices[i]));

        free(queue_priorities);

        LOG_INFO("Created device %p", static_cast<VkDevice>(ctx->devices[i]));

        ctx->streams[i] = {};
        for(int j = 0; j < queueCreateInfos.size(); j++) {
            LOG_INFO("Creating %d queues for family %d", queueCreateInfos[j].queueCount, queueCreateInfos[j].queueFamilyIndex);
            for(int k = 0; k < queueCreateInfos[j].queueCount; k++) {
                VkQueue queue;
                vkGetDeviceQueue(ctx->devices[i], queueCreateInfos[j].queueFamilyIndex, k, &queue);
                LOG_INFO("Creating queue %d with handle %p", k, queue);

                ctx->stream_indicies.push_back(std::make_pair(i, ctx->streams[i].size()));

                Stream* stream = new Stream(ctx, ctx->devices[i], queue, queueCreateInfos[j].queueFamilyIndex, 2, ctx->stream_indicies.size() - 1);
                ctx->streams[i].push_back(stream);
                stream->start_thread();
            }            
        }

        queue_index_offset += queue_counts[i];

        VmaVulkanFunctions vmaVulkanFunctions = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorCreateInfo.physicalDevice = ctx->physicalDevices[i];
        allocatorCreateInfo.device = ctx->devices[i];
        allocatorCreateInfo.instance = _instance.instance;
        allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;
        VK_CALL_RETNULL(vmaCreateAllocator(&allocatorCreateInfo, &ctx->allocators[i]));

        LOG_INFO("Created allocator %p", ctx->allocators[i]);
    }

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    ctx->command_list = command_list_create_extern(ctx);

    return ctx;
}

VkFence context_submit_work(struct Context* ctx, struct WorkInfo work_info) {
    std::unique_lock<std::mutex> lock(ctx->mutex);

    auto check_for_room = [ctx] {
        return ctx->work_info_list.size() < ctx->stream_indicies.size() * 2;
    };

    LOG_INFO("Checking for room");

    if(!check_for_room()) {
        LOG_INFO("Waiting for room");
        ctx->cv_pop.wait(lock, check_for_room);
    }

    LOG_INFO("Adding work info to list");

    ctx->work_info_list.push_back(work_info);

    LOG_INFO("Notifying all");

    ctx->cv_push.notify_all();

    LOG_INFO("unlocking");

    lock.unlock();
}

void context_wait_idle_extern(struct Context* context) {
    for(int i = 0; i < context->deviceCount; i++) {
        for(int j = 0; j < context->streams[i].size(); j++) {
            context->streams[i][j]->wait_idle();
        }
    }
}

void context_destroy_extern(struct Context* context) {
    for(int i = 0; i < context->deviceCount; i++) {
        for(int j = 0; j < context->streams[i].size(); j++) {
            context->streams[i][j]->destroy();
            delete context->streams[i][j];
        }
        context->streams[i].clear();

        vmaDestroyAllocator(context->allocators[i]);
        vkDestroyDevice(context->devices[i], nullptr);
    }

    delete context;
}