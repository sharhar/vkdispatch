#include "internal.h"

#include <stdio.h>
#include <vector>

MyInstance _instance;

void init_extern(bool debug) {
    #ifdef VKDISPATCH_USE_MVK
    setenv("MVK_CONFIG_LOG_LEVEL", "2", 0);
    #else
    VK_CALL(volkInitialize());
    #endif
	

    LOG_INFO("Initializing glslang...");

    if(!glslang_initialize_process()) {
        LOG_ERROR("Failed to initialize glslang process");
        return;
    }

    LOG_INFO("Initializing Vulkan Instance...");

    std::vector<const char*> extention_names;

    vk::ApplicationInfo appInfo = vk::ApplicationInfo()
        .setPApplicationName("vkdispatch")
        .setApplicationVersion(1)
        .setPEngineName("vkdispatch")
        .setEngineVersion(1)
        .setApiVersion(VK_API_VERSION_1_2);

    _instance.instance = vk::createInstance(
        vk::InstanceCreateInfo()
            .setPApplicationInfo(&appInfo),
        nullptr
    );

    #ifndef VKDISPATCH_USE_MVK
	volkLoadInstance(_instance.instance);
	#endif

    LOG_INFO("Initializing Vulkan Devices...");

    auto physicalDevices = _instance.instance.enumeratePhysicalDevices();

    _instance.devices.reserve(physicalDevices.size());

    for(int i = 0; i < physicalDevices.size(); i++) {
        vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties> propertiesChain = {
            vk::PhysicalDeviceProperties2(),
            vk::PhysicalDeviceSubgroupProperties()
        };

        physicalDevices[i].getProperties2(&propertiesChain.get<vk::PhysicalDeviceProperties2>());

        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> featuresChain = {
            vk::PhysicalDeviceFeatures2(),
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT()
        };

        physicalDevices[i].getFeatures2(&featuresChain.get<vk::PhysicalDeviceFeatures2>());

        vk::PhysicalDeviceProperties properties = propertiesChain.get<vk::PhysicalDeviceProperties2>().properties;
        vk::PhysicalDeviceFeatures features = featuresChain.get<vk::PhysicalDeviceFeatures2>().features;
        vk::PhysicalDeviceSubgroupProperties subgroupProperties = propertiesChain.get<vk::PhysicalDeviceSubgroupProperties>();
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = featuresChain.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();

        _instance.devices[i].version_variant = VK_API_VERSION_VARIANT(properties.apiVersion);
        _instance.devices[i].version_major = VK_API_VERSION_MAJOR(properties.apiVersion);
        _instance.devices[i].version_minor = VK_API_VERSION_MINOR(properties.apiVersion);
        _instance.devices[i].version_patch = VK_API_VERSION_PATCH(properties.apiVersion);

        _instance.devices[i].driver_version = properties.driverVersion;
        _instance.devices[i].vendor_id = properties.vendorID;
        _instance.devices[i].device_id = properties.deviceID;

        _instance.devices[i].device_type = static_cast<VkPhysicalDeviceType>(properties.deviceType);

        size_t deviceNameLength = strlen(properties.deviceName) + 1;
        _instance.devices[i].device_name = new char[deviceNameLength];
        strcpy(const_cast<char*>(_instance.devices[i].device_name), properties.deviceName);

        _instance.devices[i].float_64_support = features.shaderFloat64;
        _instance.devices[i].int_64_support = features.shaderInt64;
        _instance.devices[i].int_16_support = features.shaderInt16;

        _instance.devices[i].max_workgroup_size_x = properties.limits.maxComputeWorkGroupSize[0];
        _instance.devices[i].max_workgroup_size_y = properties.limits.maxComputeWorkGroupSize[1];
        _instance.devices[i].max_workgroup_size_z = properties.limits.maxComputeWorkGroupSize[2];

        _instance.devices[i].max_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;

        _instance.devices[i].max_workgroup_count_x = properties.limits.maxComputeWorkGroupCount[0];
        _instance.devices[i].max_workgroup_count_y = properties.limits.maxComputeWorkGroupCount[1];
        _instance.devices[i].max_workgroup_count_z = properties.limits.maxComputeWorkGroupCount[2];

        _instance.devices[i].max_descriptor_set_count = properties.limits.maxBoundDescriptorSets;

        _instance.devices[i].max_push_constant_size = properties.limits.maxPushConstantsSize;
        _instance.devices[i].max_storage_buffer_range = properties.limits.maxStorageBufferRange;
        _instance.devices[i].max_uniform_buffer_range = properties.limits.maxUniformBufferRange;

        _instance.devices[i].subgroup_size = subgroupProperties.subgroupSize;
        _instance.devices[i].supported_stages = static_cast<VkShaderStageFlags>(subgroupProperties.supportedStages);
        _instance.devices[i].supported_operations = static_cast<VkSubgroupFeatureFlags>(subgroupProperties.supportedOperations);
        _instance.devices[i].quad_operations_in_all_stages = subgroupProperties.quadOperationsInAllStages;

        _instance.devices[i].max_compute_shared_memory_size = properties.limits.maxComputeSharedMemorySize;

        //printf("Device %d: %s\n", i, _instance.devices[i].device_name);
        //printf("Atomics: %d\n", atomicFloatFeatures.shaderBufferFloat32Atomics);
        //printf("Atomics Add: %d\n", atomicFloatFeatures.shaderBufferFloat32AtomicAdd);
    }

    LOG_INFO("Initialized Vulkan Devices");
}

struct PhysicalDeviceProperties* get_devices_extern(int* count) {
    *count = _instance.instance.enumeratePhysicalDevices().size();

    LOG_INFO("Returning %d devices", *count);

    return _instance.devices.data();
}