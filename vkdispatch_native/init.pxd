# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
import sys

cdef extern from "init.h":
    struct PhysicalDeviceProperties:
        int version_variant;
        int version_major;
        int version_minor;
        int version_patch;

        int driver_version;
        int vendor_id;
        int device_id;

        int device_type;

        const char* device_name

        int float_64_support
        int int_64_support
        int int_16_support
    
    void init_extern(bool debug)
    PhysicalDeviceProperties* get_devices_extern(int* count)

cpdef inline init(bool debug):
    init_extern(debug)

cpdef inline get_devices():
    cdef int count = 0
    cdef PhysicalDeviceProperties* devices = get_devices_extern(&count)

    if not devices:
        return []
    
    device_list = []

    for i in range(count):
        device = devices[i]
        device_info = (
            device.version_variant,
            device.version_major,
            device.version_minor,
            device.version_patch,
            device.driver_version,
            device.vendor_id,
            device.device_id,
            device.device_type,
            device.device_name.decode('utf-8') if device.device_name is not None else None,  # Convert C string to Python string, handling null pointers
            device.float_64_support,
            device.int_64_support,
            device.int_16_support
        )
        device_list.append(device_info)

    return device_list