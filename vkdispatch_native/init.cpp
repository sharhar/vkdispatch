#include "internal.h"

#include <stdio.h>

Context _ctx;

void init_extern(bool debug) {
    LOG_INFO("Initializing glslang...");

    if(!glslang_initialize_process()) {
        LOG_ERROR("Failed to initialize glslang process");
        return;
    }

    LOG_INFO("Initializing Vulkan Instance...");
    
    _ctx.instance.create(VKLInstanceCreateInfo()
                            .procAddr(vkGetInstanceProcAddr)
                            .debug(VK_TRUE));

    const std::vector<VKLPhysicalDevice>& physicalDevices = _ctx.instance.getPhysicalDevices();

    _ctx.devices = new PhysicalDeviceProperties[physicalDevices.size()];

    for(int i = 0; i < physicalDevices.size(); i++) {
        VkPhysicalDeviceProperties properties = physicalDevices[i].getProperties();

        _ctx.devices[i].version_variant = VK_API_VERSION_VARIANT(properties.apiVersion);
        _ctx.devices[i].version_major = VK_API_VERSION_MAJOR(properties.apiVersion);
        _ctx.devices[i].version_minor = VK_API_VERSION_MINOR(properties.apiVersion);
        _ctx.devices[i].version_patch = VK_API_VERSION_PATCH(properties.apiVersion);

        _ctx.devices[i].driver_version = properties.driverVersion;
        _ctx.devices[i].vendor_id = properties.vendorID;
        _ctx.devices[i].device_id = properties.deviceID;

        _ctx.devices[i].device_type = properties.deviceType;

        size_t deviceNameLength = strlen(properties.deviceName) + 1;
        _ctx.devices[i].device_name = new char[deviceNameLength];
        strcpy((char*)_ctx.devices[i].device_name, properties.deviceName);
    }
}

struct PhysicalDeviceProperties* get_devices_extern(int* count) {
    *count = _ctx.instance.getPhysicalDevices().size();
    return _ctx.devices;
}