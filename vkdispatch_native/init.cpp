#include "internal.h"

#include <stdio.h>

MyInstance _instance;

void init_extern(bool debug) {
    LOG_INFO("Initializing glslang...");

    if(!glslang_initialize_process()) {
        LOG_ERROR("Failed to initialize glslang process");
        return;
    }

    LOG_INFO("Initializing Vulkan Instance...");
    
    _instance.instance.create(
        VKLInstanceCreateInfo()
        .procAddr(vkGetInstanceProcAddr)
        .debug(VK_TRUE)
    );

    const std::vector<VKLPhysicalDevice*>& physicalDevices = _instance.instance.getPhysicalDevices();

    _instance.devices = new PhysicalDeviceProperties[physicalDevices.size()];

    for(int i = 0; i < physicalDevices.size(); i++) {
        VkPhysicalDeviceProperties properties = physicalDevices[i]->getProperties();
        VkPhysicalDeviceFeatures features = physicalDevices[i]->getFeatures();

        _instance.devices[i].version_variant = VK_API_VERSION_VARIANT(properties.apiVersion);
        _instance.devices[i].version_major = VK_API_VERSION_MAJOR(properties.apiVersion);
        _instance.devices[i].version_minor = VK_API_VERSION_MINOR(properties.apiVersion);
        _instance.devices[i].version_patch = VK_API_VERSION_PATCH(properties.apiVersion);

        _instance.devices[i].driver_version = properties.driverVersion;
        _instance.devices[i].vendor_id = properties.vendorID;
        _instance.devices[i].device_id = properties.deviceID;

        _instance.devices[i].device_type = properties.deviceType;

        size_t deviceNameLength = strlen(properties.deviceName) + 1;
        _instance.devices[i].device_name = new char[deviceNameLength];
        strcpy((char*)_instance.devices[i].device_name, properties.deviceName);

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
    }
}

struct PhysicalDeviceProperties* get_devices_extern(int* count) {
    *count = _instance.instance.getPhysicalDevices().size();
    return _instance.devices;
}