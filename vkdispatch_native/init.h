#ifndef SRC_INIT_H
#define SRC_INIT_H

struct PhysicalDeviceProperties {
    int version_variant;
    int version_major;
    int version_minor;
    int version_patch;

    int driver_version;
    int vendor_id;
    int device_id;

    int device_type;

    const char* device_name;

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
};

void init_extern(bool debug);
struct PhysicalDeviceProperties* get_devices_extern(int* count);

#endif // INIT_H