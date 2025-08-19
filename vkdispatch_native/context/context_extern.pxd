# distutils: language=c++
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import sys

cdef extern from "context/context_extern.hh":
    enum LogLevel:
        LOG_LEVEL_VERBOSE = 0
        LOG_LEVEL_INFO = 1
        LOG_LEVEL_WARNING = 2
        LOG_LEVEL_ERROR = 3

    struct QueueFamilyProperties:
        unsigned int queueCount
        unsigned int queueFlags

    struct PhysicalDeviceDetails:
        int version_variant;
        int version_major;
        int version_minor;
        int version_patch;

        int driver_version;
        int vendor_id;
        int device_id;

        int device_type;

        const char* device_name

        int shader_buffer_float32_atomics;
        int shader_buffer_float32_atomic_add;

        int float_64_support
        int float_16_support
        int int_64_support
        int int_16_support

        int storage_buffer_16_bit_access
        int uniform_and_storage_buffer_16_bit_access
        int storage_push_constant_16
        int storage_input_output_16

        unsigned int max_workgroup_size_x
        unsigned int max_workgroup_size_y
        unsigned int max_workgroup_size_z

        unsigned int max_workgroup_invocations

        unsigned int max_workgroup_count_x
        unsigned int max_workgroup_count_y
        unsigned int max_workgroup_count_z

        unsigned int max_descriptor_set_count
        unsigned int max_push_constant_size
        unsigned int max_storage_buffer_range
        unsigned int max_uniform_buffer_range
        unsigned int uniform_buffer_alignment

        unsigned int subgroup_size
        unsigned int supported_stages
        unsigned int supported_operations
        unsigned int quad_operations_in_all_stages

        unsigned int max_compute_shared_memory_size

        unsigned int queue_family_count
        QueueFamilyProperties* queue_family_properties
    
    void init_extern(bool debug, LogLevel log_level)
    PhysicalDeviceDetails* get_devices_extern(int* count)
    void log_extern(LogLevel log_level, const char* text, const char* file_str, int line_str)
    void set_log_level_extern(LogLevel log_level)

    struct Context

    Context* context_create_extern(int* device_indicies, int* queue_counts, int* queue_families, int device_count)
    void context_queue_wait_idle_extern(Context* context, int queue_index);
    void context_destroy_extern(Context* device_context);

    const char* get_error_string_extern()

cpdef inline init(bool debug, int log_level):
    init_extern(debug, <LogLevel>(log_level))

cpdef inline log(int log_level, bytes text, bytes file_str, int line_str):
    cdef const char* text_c = text
    cdef const char* file_str_c = file_str

    log_extern(<LogLevel>(log_level), text_c, file_str_c, line_str)

cpdef inline set_log_level(int log_level):
    set_log_level_extern(<LogLevel>(log_level))

cpdef inline get_devices():
    cdef int count = 0
    cdef PhysicalDeviceDetails* devices = get_devices_extern(&count)

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
            device.shader_buffer_float32_atomics,
            device.shader_buffer_float32_atomic_add,
            device.float_64_support,
            device.float_16_support,
            device.int_64_support,
            device.int_16_support,
            device.storage_buffer_16_bit_access,
            device.uniform_and_storage_buffer_16_bit_access,
            device.storage_push_constant_16,
            device.storage_input_output_16,
            (device.max_workgroup_size_x, device.max_workgroup_size_y, device.max_workgroup_size_z),
            device.max_workgroup_invocations,
            (device.max_workgroup_count_x, device.max_workgroup_count_y, device.max_workgroup_count_z),
            device.max_descriptor_set_count,
            device.max_push_constant_size,
            device.max_storage_buffer_range,
            device.max_uniform_buffer_range,
            device.uniform_buffer_alignment,
            device.subgroup_size,
            device.supported_stages,
            device.supported_operations,
            device.quad_operations_in_all_stages,
            device.max_compute_shared_memory_size,
            [(device.queue_family_properties[j].queueCount, device.queue_family_properties[j].queueFlags) for j in range(device.queue_family_count)]
        )
        device_list.append(device_info)

    return device_list

cpdef inline context_create(list[int] device_indicies, list[list[int]] queue_families):
    assert len(device_indicies) == len(queue_families)

    cdef int len_device_indicies = len(device_indicies)
    cdef int* device_indicies_c = <int*>malloc(len_device_indicies * sizeof(int))
    cdef int* queue_counts_c    = <int*>malloc(len_device_indicies * sizeof(int))

    cdef int total_queue_count = 0

    for i in range(len_device_indicies):
        device_indicies_c[i] = device_indicies[i]
        queue_counts_c[i] = len(queue_families[i]) #submission_thread_counts[i]
        total_queue_count += queue_counts_c[i]

    cdef int* queue_families_c = <int*>malloc(total_queue_count * sizeof(int))

    cdef int current_index = 0
    
    for i in range(len_device_indicies):
        for j in range(queue_counts_c[i]):
            queue_families_c[current_index] = queue_families[i][j]
            current_index += 1
    
    assert current_index == total_queue_count

    cdef unsigned long long result = <unsigned long long>context_create_extern(device_indicies_c, queue_counts_c, queue_families_c, len_device_indicies)

    free(device_indicies_c)
    free(queue_counts_c)
    free(queue_families_c)

    return result

cpdef inline void context_queue_wait_idle(unsigned long long context, int queue_index):
    context_queue_wait_idle_extern(<Context*>context, queue_index)

cpdef inline context_destroy(unsigned long long context):
    context_destroy_extern(<Context*>context)

cpdef inline get_error_string():
    cdef const char* error_string = get_error_string_extern()
    if error_string is NULL:
        return 0
    else:
        return error_string.decode('utf-8')