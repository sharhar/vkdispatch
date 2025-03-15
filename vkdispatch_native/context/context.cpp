#include "../internal.hh"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>

#include <climits>

#include <glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

void inplace_min(int* a, int b) {
    if(b < *a) {
        *a = b;
    }
}

struct Context* context_create_extern(int* device_indicies, int* queue_counts, int* queue_families, int device_count) {
    LOG_INFO("Creating context with %d devices", device_count);

    struct Context* ctx = new struct Context();
    ctx->deviceCount = device_count;
    ctx->physicalDevices.resize(device_count);
    ctx->devices.resize(device_count);
    //ctx->streams.resize(device_count);
    ctx->stream_index_map.resize(device_count);
    ctx->allocators.resize(device_count);
    ctx->glslang_resource_limits = new glslang_resource_t();
    memcpy(ctx->glslang_resource_limits, glslang_default_resource(), sizeof(glslang_resource_t));

    glslang_resource_t* resource = reinterpret_cast<glslang_resource_t*>(ctx->glslang_resource_limits);

    resource->max_compute_work_group_size_x = INT_MAX;
    resource->max_compute_work_group_size_y = INT_MAX;
    resource->max_compute_work_group_size_z = INT_MAX;

    resource->max_compute_work_group_count_x = INT_MAX;
    resource->max_compute_work_group_count_y = INT_MAX;
    resource->max_compute_work_group_count_z = INT_MAX;
    
    ctx->work_queue = new WorkQueue(device_count * 4, 4);
    ctx->command_list = command_list_create_extern(ctx);
    
    LOG_INFO("Enumerating physical devices...");

    int queue_index_offset = 0;

    for(int i = 0; i < device_count; i++) {
        LOG_INFO("Device %d Index: %d", i, device_indicies[i]);

        struct PhysicalDeviceDetails* details = &_instance.device_details[device_indicies[i]];

        inplace_min(&resource->max_compute_work_group_size_x, details->max_workgroup_size_x);
        inplace_min(&resource->max_compute_work_group_size_y, details->max_workgroup_size_y);
        inplace_min(&resource->max_compute_work_group_size_z, details->max_workgroup_size_z);

        inplace_min(&resource->max_compute_work_group_count_x, details->max_workgroup_count_x);
        inplace_min(&resource->max_compute_work_group_count_y, details->max_workgroup_count_y);
        inplace_min(&resource->max_compute_work_group_count_z, details->max_workgroup_count_z);

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

        //VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeaturesEnableStruct = {};
        //atomicFloatFeaturesEnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        //atomicFloatFeaturesEnableStruct.shaderBufferFloat32AtomicAdd = VK_TRUE;

        //VkPhysicalDeviceFeatures2 features2EnableStruct = {};
        //features2EnableStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        //features2EnableStruct.pNext = &atomicFloatFeaturesEnableStruct;

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
        deviceCreateInfo.pNext = &_instance.features[device_indicies[i]];
        deviceCreateInfo.flags = 0;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.enabledLayerCount = 0;
        deviceCreateInfo.ppEnabledLayerNames = nullptr;
        deviceCreateInfo.enabledExtensionCount = supportedExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = supportedExtensions.data();
        deviceCreateInfo.pEnabledFeatures = nullptr;

        VK_CALL_RETNULL(vkCreateDevice(ctx->physicalDevices[i], &deviceCreateInfo, nullptr, &ctx->devices[i]));

        free(queue_priorities);

        LOG_INFO("Created device %p", static_cast<VkDevice>(ctx->devices[i]));

        VmaVulkanFunctions vmaVulkanFunctions = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        vmaVulkanFunctions.vkAllocateMemory = vkAllocateMemory;
        vmaVulkanFunctions.vkFreeMemory = vkFreeMemory;
        vmaVulkanFunctions.vkMapMemory = vkMapMemory;
        vmaVulkanFunctions.vkUnmapMemory = vkUnmapMemory;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorCreateInfo.physicalDevice = ctx->physicalDevices[i];
        allocatorCreateInfo.device = ctx->devices[i];
        allocatorCreateInfo.instance = _instance.instance;
        allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;
        VK_CALL_RETNULL(vmaCreateAllocator(&allocatorCreateInfo, &ctx->allocators[i]));

        LOG_INFO("Created allocator %p", ctx->allocators[i]);

        ctx->stream_index_map[i] = {};

        for(int j = 0; j < queueCreateInfos.size(); j++) {
            LOG_INFO("Creating %d queues for family %d", queueCreateInfos[j].queueCount, queueCreateInfos[j].queueFamilyIndex);
            for(int k = 0; k < queueCreateInfos[j].queueCount; k++) {
                VkQueue queue;
                vkGetDeviceQueue(ctx->devices[i], queueCreateInfos[j].queueFamilyIndex, k, &queue);
                LOG_INFO("Creating queue %d with handle %p", k, queue);

                ctx->stream_index_map[i].push_back(ctx->streams.size());
                ctx->streams.push_back(new Stream(ctx, ctx->devices[i], queue, queueCreateInfos[j].queueFamilyIndex, i, ctx->streams.size()));
            }            
        }

        queue_index_offset += queue_counts[i];
    }

    LOG_INFO("Created a context with the following glsl_resource properties:");
    LOG_INFO("  Max Compute Work Group Size: (%d, %d, %d)", resource->max_compute_work_group_size_x, resource->max_compute_work_group_size_y, resource->max_compute_work_group_size_z);
    LOG_INFO("  Max Compute Work Group Count: (%d, %d, %d)", resource->max_compute_work_group_count_x, resource->max_compute_work_group_count_y, resource->max_compute_work_group_count_z);

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    for(int i = 0; i < ctx->streams.size(); i++) {
        command_list_record_command(ctx->command_list, 
            "noop-on-init",
            0,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            [](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
                // Do nothing
            }
        );

        Signal signal;
        command_list_submit_extern(ctx->command_list, NULL, 1, &i, 1, &signal, RECORD_TYPE_SYNC);
        command_list_reset_extern(ctx->command_list);
        RETURN_ON_ERROR(NULL)

        signal.wait();
    }

    ctx->handle_manager = new HandleManager(ctx);

    return ctx;
}

void context_destroy_extern(struct Context* context) {
    for(int i = 0; i < context->streams.size(); i++) {
        context->streams[i]->destroy();
        delete context->streams[i];
    }

    context->streams.clear();

    for(int i = 0; i < context->deviceCount; i++) {
        vmaDestroyAllocator(context->allocators[i]);
        vkDestroyDevice(context->devices[i], nullptr);
    }

    delete context;
}