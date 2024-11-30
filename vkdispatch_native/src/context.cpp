#include "../include/internal.hh"
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
    
    ctx->work_queue = new WorkQueue(device_count * 4, 4);
    ctx->command_list = command_list_create_extern(ctx);
    
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
            "VK_EXT_shader_atomic_float",
            "VK_EXT_memory_budget"
        };

#ifdef __APPLE__
        desiredExtensions.push_back("VK_KHR_portability_subset");
#endif
        
        uint32_t extensionCount;
        std::vector<VkExtensionProperties> deviceExtensions;
        VK_CALL_RETNULL(vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, nullptr));
        deviceExtensions.resize(extensionCount);
        VK_CALL_RETNULL(vkEnumerateDeviceExtensionProperties(ctx->physicalDevices[i], nullptr, &extensionCount, deviceExtensions.data()));

        for(uint32_t i = 0; i < extensionCount; i++) {
            LOG_VERBOSE("Device Extension '%s' is supported by device", deviceExtensions[i].extensionName);
        }

        bool memory_budget_supported = false;

        // Check if all desired extensions are supported
        std::vector<const char*> supportedExtensions;
        for(uint32_t j = 0; j < desiredExtensions.size(); j++) {
            
            bool found_extension = false;

            for(uint32_t i = 0; i < deviceExtensions.size(); i++) {
                if(strcmp(desiredExtensions[j], deviceExtensions[i].extensionName) == 0) {
                    found_extension = true;
                }
            }

            if(!found_extension) {
                LOG_WARNING("Device Extension '%s' is desired but not found", desiredExtensions[j]);
                continue;
            }

            if(strcmp(desiredExtensions[j], "VK_EXT_memory_budget") == 0) {
                memory_budget_supported = true;
            }

            LOG_INFO("Device Extension '%s' is supported", desiredExtensions[j]);

            supportedExtensions.push_back(desiredExtensions[j]);
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

        VmaVulkanFunctions vmaVulkanFunctions = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorCreateInfo.physicalDevice = ctx->physicalDevices[i];
        allocatorCreateInfo.device = ctx->devices[i];
        allocatorCreateInfo.instance = _instance.instance;
        allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;
        
        if (memory_budget_supported)
            allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        
        VK_CALL_RETNULL(vmaCreateAllocator(&allocatorCreateInfo, &ctx->allocators[i]));

        LOG_INFO("Created allocator %p", ctx->allocators[i]);

        ctx->streams[i] = {};
        for(int j = 0; j < queueCreateInfos.size(); j++) {
            LOG_INFO("Creating %d queues for family %d", queueCreateInfos[j].queueCount, queueCreateInfos[j].queueFamilyIndex);
            for(int k = 0; k < queueCreateInfos[j].queueCount; k++) {
                VkQueue queue;
                vkGetDeviceQueue(ctx->devices[i], queueCreateInfos[j].queueFamilyIndex, k, &queue);
                LOG_INFO("Creating queue %d with handle %p", k, queue);

                ctx->stream_indicies.push_back(std::make_pair(i, ctx->streams[i].size()));
                ctx->streams[i].push_back(new Stream(ctx, ctx->devices[i], queue, queueCreateInfos[j].queueFamilyIndex, ctx->stream_indicies.size() - 1));
            }            
        }

        queue_index_offset += queue_counts[i];
    }

    LOG_INFO("Created context at %p with %d devices", ctx, device_count);

    for(int i = 0; i < ctx->stream_indicies.size(); i++) {
        struct CommandInfo command = {};
        command.type = COMMAND_TYPE_NOOP;
        command.pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        command_list_record_command(ctx->command_list, command);

        Signal signal;
        command_list_submit_extern(ctx->command_list, NULL, 1, &i, 1, &signal);
        command_list_reset_extern(ctx->command_list);
        RETURN_ON_ERROR(NULL)

        signal.wait();
    }

    return ctx;
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

void log_memory_extern(Context* ctx) {
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    

    for(int dev = 0; dev < ctx->allocators.size(); dev++) {
        memset(budgets, 0, sizeof(budgets));
        vmaGetHeapBudgets(ctx->allocators[dev], budgets);

        printf("Memory Budgets for Device %d:\n", dev);

        for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
            if(budgets[i].budget == 0)
                continue;

            printf("\tHeap %d: %llu MB used out of %llu MB total.\n", i, budgets[i].usage / (1024 * 1024), budgets[i].budget / (1024 * 1024));
            
            printf("\t\tAllocation Bytes: %llu\n", budgets[i].statistics.allocationBytes);
            printf("\t\tAllocation Count: %llu\n", budgets[i].statistics.allocationCount);
            printf("\t\tBlock Bytes: %llu\n", budgets[i].statistics.blockBytes);
            printf("\t\tBlock Count: %llu\n", budgets[i].statistics.blockCount);
        }
    }

    

}