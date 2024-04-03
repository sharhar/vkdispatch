#ifndef SRC_INIT_H
#define SRC_INIT_H

struct MyPhysicalDeviceProperties {
    int version_variant;
    int version_major;
    int version_minor;
    int version_patch;

    int driver_version;
    int vendor_id;
    int device_id;

    int device_type;

    const char* device_name;
};

void init_extern(bool debug);
struct MyPhysicalDeviceProperties* get_devices_extern(int* count);

#endif // INIT_H