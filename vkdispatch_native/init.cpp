#include "internal.h"

#include <iostream>
#include <stdio.h>
#include <vector>

#include <algorithm>
#include <string>
#include <glslang_c_interface.h>

MyInstance _instance;

const char* concat(const std::string& str,  const char** out, const char* cstr) {
    size_t total_length = str.length() + strlen(cstr) + 1;
    char* result = new char[total_length];

    strcpy(result, str.c_str());
    strcat(result, cstr);


    if(out != nullptr) {
        if(*out != nullptr)
            delete[] *out;

        *out = result;
    }

    return result;
}

// Some platform specific code borrowed from volk for loading Vulkan
#ifdef _WIN32

#include <Windows.h>

//__declspec(dllimport) HMODULE __stdcall LoadLibraryA(LPCSTR);
//__declspec(dllimport) FARPROC __stdcall GetProcAddress(HMODULE, LPCSTR);
//__declspec(dllimport) int __stdcall FreeLibrary(HMODULE);

#endif

#if defined(__GNUC__)
#    define VOLK_DISABLE_GCC_PEDANTIC_WARNINGS \
		_Pragma("GCC diagnostic push") \
		_Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#    define VOLK_RESTORE_GCC_PEDANTIC_WARNINGS \
		_Pragma("GCC diagnostic pop")
#else
#    define VOLK_DISABLE_GCC_PEDANTIC_WARNINGS
#    define VOLK_RESTORE_GCC_PEDANTIC_WARNINGS
#endif


// This is a modified version of volkInitialize
static PFN_vkGetInstanceProcAddr load_vulkan(std::string path) {
    PFN_vkGetInstanceProcAddr load_func = nullptr;
    const char* res_str = nullptr;

#if defined(_WIN32)
	HMODULE module = LoadLibraryA(concat(path, &res_str, "vulkan-1.dll"));
	if (!module)
		return nullptr;

	// note: function pointer is cast through void function pointer to silence cast-function-type warning on gcc8
	load_func = (PFN_vkGetInstanceProcAddr)(void(*)(void))GetProcAddress(module, "vkGetInstanceProcAddr");
#elif defined(__APPLE__)
	void* module = dlopen(concat(path, &res_str, "libvulkan.dylib"), RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen(concat(path, &res_str, "libvulkan.1.dylib"), RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen(concat(path, &res_str, "libMoltenVK.dylib"), RTLD_NOW | RTLD_LOCAL);
    if (!module)
        module = dlopen(concat(path, &res_str, "vulkan.framework/vulkan"), RTLD_NOW | RTLD_LOCAL);
    if (!module)
        module = dlopen(concat(path, &res_str, "MoltenVK.framework/MoltenVK"), RTLD_NOW | RTLD_LOCAL);
	if (!module)
		return nullptr;

	load_func = (PFN_vkGetInstanceProcAddr)dlsym(module, "vkGetInstanceProcAddr");
#else
	void* module = dlopen(concat(path, &res_str, "libvulkan.so.1"), RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen(concat(path, &res_str, "libvulkan.so"), RTLD_NOW | RTLD_LOCAL);
	if (!module)
		return nullptr;
	VOLK_DISABLE_GCC_PEDANTIC_WARNINGS
	load_func = (PFN_vkGetInstanceProcAddr)dlsym(module, "vkGetInstanceProcAddr");
	VOLK_RESTORE_GCC_PEDANTIC_WARNINGS
#endif

    delete[] res_str;

	return load_func;
}

VkBool32 VKAPI_PTR mystdOutLogger(
    VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
    void*                                            pUserData) {
    printf("VKL: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

#ifdef _WIN32

void setenv(const char* arg1, const char* arg2, int arg3) {

}

#endif

void init_extern(bool debug) {
    #ifdef VKDISPATCH_LOADER_PATH
    const char* layer_path = concat(VKDISPATCH_LOADER_PATH, nullptr, "share/vulkan/explicit_layer.d/");
    LOG_INFO("Layer path: %s", layer_path);
    setenv("VK_LAYER_PATH", layer_path, 0);
    delete[] layer_path;
    #endif

    #ifndef VKDISPATCH_USE_VOLK    

    setenv("MVK_CONFIG_LOG_LEVEL", "2", 0);

    #else

    #ifdef VKDISPATCH_LOADER_PATH

    LOG_INFO("Loading Vulkan with loader path: %s", VKDISPATCH_LOADER_PATH);
    
    const char* vulk_path = concat(VKDISPATCH_LOADER_PATH, nullptr, "lib/");
    LOG_INFO("Vulkan path: %s", vulk_path);
    PFN_vkGetInstanceProcAddr load_func = load_vulkan(vulk_path);
    delete[] vulk_path;

    if(load_func) {
        LOG_INFO("Loading Vulkan using custom loader");

        volkInitializeCustom(load_func);
    } else {
        LOG_INFO("Failed to load Vulkan using custom loader, trying normally with volk...");

        if (volkInitialize() != VK_SUCCESS) {
            LOG_ERROR("Failed to load Vulkan using volk!");
            return;
        }
    }

    #else

    LOG_INFO("Loading Vulkan using volk");

    VK_CALL(volkInitialize());
    
    #endif
    #endif
	

    LOG_INFO("Initializing glslang...");

    if(!glslang_initialize_process()) {
        LOG_ERROR("Failed to initialize glslang process");
        return;
    }

    LOG_INFO("Initializing Vulkan Instance...");

    auto instanceVersion = vk::enumerateInstanceVersion();

    LOG_INFO("Instance API Version: %d.%d.%d", VK_API_VERSION_MAJOR(instanceVersion), VK_API_VERSION_MINOR(instanceVersion), VK_API_VERSION_PATCH(instanceVersion));

    vk::ApplicationInfo appInfo = vk::ApplicationInfo()
        .setPApplicationName("vkdispatch")
        .setApplicationVersion(1)
        .setPEngineName("vkdispatch")
        .setEngineVersion(1)
        .setApiVersion(VK_API_VERSION_1_2);

    vk::InstanceCreateFlags flags = vk::InstanceCreateFlags();

    std::vector<const char *> extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    std::vector<const char *> layers = {
        "VK_LAYER_KHRONOS_validation"
    };


#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    auto instance_layers = vk::enumerateInstanceLayerProperties();
    auto instance_extensions = vk::enumerateInstanceExtensionProperties();

    // Check if layers are supported and remove the ones that aren't
    std::vector<const char*> supportedLayers;
    for (const char* layer : layers) {
        auto it = std::find_if(instance_layers.begin(), instance_layers.end(), [&](const vk::LayerProperties& prop) {
            return strcmp(layer, prop.layerName) == 0;
        });
        if (it != instance_layers.end()) {
            LOG_INFO("Layer '%s' is supported", layer);
            supportedLayers.push_back(layer);
        } else {
            LOG_INFO("Layer '%s' is not supported", layer);
        }
    }

    // Check if extensions are supported and remove the ones that aren't
    std::vector<const char*> supportedExtensions;
    for (const char* extension : extensions) {
        auto it = std::find_if(instance_extensions.begin(), instance_extensions.end(), [&](const vk::ExtensionProperties& prop) {
            return strcmp(extension, prop.extensionName) == 0;
        });
        if (it != instance_extensions.end()) {
            LOG_INFO("Extension '%s' is supported", extension);
            supportedExtensions.push_back(extension);
        } else {
            LOG_INFO("Extension '%s' is not supported", extension);
        }
    }

    auto validationFeatures = vk::ValidationFeatureEnableEXT::eDebugPrintf;

    vk::StructureChain<vk::InstanceCreateInfo, vk::ValidationFeaturesEXT> instanceCreateChain = { 
        vk::InstanceCreateInfo()
            .setPApplicationInfo(&appInfo)
            .setFlags(flags)
            .setPEnabledExtensionNames(supportedExtensions)
            .setPEnabledLayerNames(supportedLayers),
        vk::ValidationFeaturesEXT()
            .setEnabledValidationFeatures(validationFeatures)
    };

    _instance.instance = vk::createInstance(instanceCreateChain.get<vk::InstanceCreateInfo>());

    #ifdef VKDISPATCH_USE_VOLK
	volkLoadInstance(_instance.instance);
	#endif

    _instance.instance.createDebugUtilsMessengerEXT(
        vk::DebugUtilsMessengerCreateInfoEXT()
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(mystdOutLogger)
    );

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