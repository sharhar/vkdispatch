#ifndef SRC_INIT_H
#define SRC_INIT_H

#include "base.hh"

struct QueueFamilyProperties {
    unsigned int queueCount;
    unsigned int queueFlags;
};

struct PhysicalDeviceDetails {
    int version_variant;
    int version_major;
    int version_minor;
    int version_patch;

    int driver_version;
    int vendor_id;
    int device_id;

    int device_type;

    const char* device_name;

    int shader_buffer_float32_atomics;
    int shader_buffer_float32_atomic_add;

    int float_64_support;
    int int_64_support;
    int int_16_support;

    unsigned int max_workgroup_size_x;
    unsigned int max_workgroup_size_y;
    unsigned int max_workgroup_size_z;

    unsigned int max_workgroup_invocations;

    unsigned int max_workgroup_count_x;
    unsigned int max_workgroup_count_y;
    unsigned int max_workgroup_count_z;

    unsigned int max_descriptor_set_count;
    unsigned int max_push_constant_size;
    unsigned int max_storage_buffer_range;
    unsigned int max_uniform_buffer_range;
    unsigned int uniform_buffer_alignment;

    unsigned int subgroup_size;
    unsigned int supported_stages;
    unsigned int supported_operations;
    unsigned int quad_operations_in_all_stages;

    unsigned int max_compute_shared_memory_size;

    unsigned int queue_family_count;
    struct QueueFamilyProperties* queue_family_properties;
};

void init_extern(bool debug, LogLevel log_level);
struct PhysicalDeviceDetails* get_devices_extern(int* count);

void log_extern(LogLevel log_level, const char* text, const char* file_str, int line_str);

#endif // INIT_H