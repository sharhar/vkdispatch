import typing
from enum import Enum
import os

import inspect

from .errors import check_for_errors

import vkdispatch_native

# string representations of device types
device_type_id_to_str_dict = {
    0: "Other",
    1: "Integrated GPU",
    2: "Discrete GPU",
    3: "Virtual GPU",
    4: "CPU",
}

# device type ranking for sorting, higher is better:
# 4: Discrete GPU
# 3: Virtual GPU
# 2: Integrated GPU
# 1: CPU
# 0: Other
device_type_ranks_dict = {
    0: 0,
    1: 2,
    2: 4,
    3: 3,
    4: 1
}

def get_queue_type_strings(queue_type: int, verbose: bool) -> typing.List[str]:
    """
    A function which returns a list of strings representing the queue's supported operations.

    Args:
        queue_type (`int`): The queue type, a combination of the following flags:
            0x001: Graphics
            0x002: Compute
            0x004: Transfer
            0x008: Sparse Binding
            0x010: Protected
            0x020: Video Decode
            0x040: Video Encode
            0x100: Optical Flow (NV)
        verbose (`bool`): A flag that controls whether to include verbose output. By Default, this
            flag is set to False. Meaning only the Graphics and Compute flags will be included.
    """

    result = []

    if queue_type & 0x001:
        result.append("Graphics")
    if queue_type & 0x002:
        result.append("Compute")

    if verbose:
        if queue_type & 0x004:
            result.append("Transfer")
        if queue_type & 0x008:
            result.append("Sparse Binding")
        if queue_type & 0x010:
            result.append("Protected")
        if queue_type & 0x020:
            result.append("Video Decode")
        if queue_type & 0x040:
            result.append("Video Encode")
        if queue_type & 0x100:
            result.append("Optical Flow (NV)")

    return result

class LogLevel(Enum):
    """
    An enumeration which represents the log levels.

    Attributes:
        VERBOSE (`int`): All possible logs are printed. Module must be compiled with debug mode enabled to see these logs.
        INFO (`int`): All release mode logs are printed. Useful for debugging the publicly available module.
        WARNING (`int`): Only warnings and errors are printed. Default log level.
        ERROR (`int`): Only errors are printed. Useful for muting annoying warnings that you *know* are harmless.
    """
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

class DeviceInfo:
    """
    A class which represents all the features and properties of a Vulkan device.
    
    NOTE: This class is not meant to be instantiated by the user. Instead, the
    user should call the get_devices() function to get a list of DeviceInfo
    instances.

    Attributes:
        dev_index (`int`): The index of the device.
        version_variant (`int`): The Vulkan variant version.
        version_major (`int`): The Vulkan major version.
        version_minor (`int`): The Vulkan minor version.
        version_patch (`int`): The Vulkan patch version.
        driver_version (`int`): The version of the driver.
        vendor_id (`int`): The vendor ID of the device.
        device_id (`int`): The device ID of the device.
        device_type (`int`): The device type, which is one of the following:
            0: Other
            1: Integrated GPU
            2: Discrete GPU
            3: Virtual GPU
            4: CPU
        device_name (`str`): The name of the device.
        shader_buffer_float32_atomics (`int`): float32 atomics support.
        shader_buffer_float32_atomic_add (`int`): float32 atomic add support.
        float_64_support (`int`): 64-bit float support.
        int_64_support (`int`): 64-bit integer support.
        int_16_support (`int`): 16-bit integer support.
        max_workgroup_size (`Tuple[int, int, int]`): Maximum workgroup size.
        max_workgroup_invocations (`int`): Maximum workgroup invocations.
        max_workgroup_count (`Tuple[int, int, int]`): Maximum workgroup count.
        max_bound_descriptor_sets (`int`): Maximum bound descriptor sets.
        max_push_constant_size (`int`): Maximum push constant size.
        max_storage_buffer_range (`int`): Maximum storage buffer range.
        max_uniform_buffer_range (`int`): Maximum uniform buffer range.
        uniform_buffer_alignment (`int`): Uniform buffer alignment.
        sub_group_size (`int`): Subgroup size.
        supported_stages (`int`): Supported subgroup stages.
        supported_operations (`int`): Supported subgroup operations.
        quad_operations_in_all_stages (`int`): Quad operations in all stages.
        max_compute_shared_memory_size (`int`): Maximum compute shared memory size.
        queue_properties (`List[Tuple[int, int]]`): Queue properties.
    """

    def __init__(
        self,
        dev_index: int,
        version_variant: int,
        version_major: int,
        version_minor: int,
        version_patch: int,
        driver_version: int,
        vendor_id: int,
        device_id: int,
        device_type: int,
        device_name: str,
        shader_buffer_float32_atomics: int,
        shader_buffer_float32_atomic_add: int,
        float_64_support: int,
        int_64_support: int,
        int_16_support: int,
        max_workgroup_size: typing.Tuple[int, int, int],
        max_workgroup_invocations: int,
        max_workgroup_count: typing.Tuple[int, int, int],
        max_bound_descriptor_sets: int,
        max_push_constant_size: int,
        max_storage_buffer_range: int,
        max_uniform_buffer_range: int,
        uniform_buffer_alignment: int,
        sub_group_size: int,
        supported_stages: int,
        supported_operations: int,
        quad_operations_in_all_stages: int,
        max_compute_shared_memory_size: int,
        queue_properties: typing.List[typing.Tuple[int, int]]
    ):
        self.dev_index = dev_index

        self.version_variant = version_variant
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_patch = version_patch

        self.driver_version = driver_version
        self.vendor_id = vendor_id
        self.device_id = device_id

        self.device_type = device_type

        self.device_name = device_name

        self.shader_buffer_float32_atomics = shader_buffer_float32_atomics
        self.shader_buffer_float32_atomic_add = shader_buffer_float32_atomic_add

        self.float_64_support = float_64_support
        self.int_64_support = int_64_support
        self.int_16_support = int_16_support

        self.max_workgroup_size = max_workgroup_size
        self.max_workgroup_invocations = max_workgroup_invocations
        self.max_workgroup_count = max_workgroup_count

        self.max_bound_descriptor_sets = max_bound_descriptor_sets
        self.max_push_constant_size = max_push_constant_size
        self.max_storage_buffer_range = max_storage_buffer_range
        self.max_uniform_buffer_range = max_uniform_buffer_range
        self.uniform_buffer_alignment = uniform_buffer_alignment

        self.sub_group_size = sub_group_size
        self.supported_stages = supported_stages
        self.supported_operations = supported_operations
        self.quad_operations_in_all_stages = quad_operations_in_all_stages

        self.max_compute_shared_memory_size = max_compute_shared_memory_size

        self.queue_properties = queue_properties

    def get_info_string(self, verbose: bool = False) -> str:
        """
        A method which returns a string representation of the device information.

        Args:
            verbose (`bool`): A flag to enable verbose output.
        
        Returns:
            str: A string representation of the device information.
        """

        result = f"Device {self.dev_index}: {self.device_name}\n"

        result += f"\tVulkan Version: {self.version_major}.{self.version_minor}.{self.version_patch}\n"
        result += f"\tDevice Type: {device_type_id_to_str_dict[self.device_type]}\n"

        if self.version_variant != 0:
            result += f"\tVariant: {self.version_variant}\n"

        if verbose:
            result += f"\tDriver Version={self.driver_version}\n"
            result += f"\tVendor ID={self.vendor_id}\n"
            result += f"\tDevice ID={self.device_id}\n"

        result += "\n\tFeatures:\n"

        if verbose:
            result += f"\t\tFloat32 Atomics: {self.shader_buffer_float32_atomics == 1}\n"
        
        result += f"\t\tFloat32 Atomic Add: {self.shader_buffer_float32_atomic_add == 1}\n"

        result += "\n\tProperties:\n"

        result += f"\t\t64-bit Float Support: {self.float_64_support == 1}\n"
        result += f"\t\t64-bit Int Support: {self.int_64_support == 1}\n"
        result += f"\t\t16-bit Int Suppor: {self.int_16_support == 1}\n"
        
        if verbose:
            result += f"\t\tMax Workgroup Sizes: {self.max_workgroup_size}\n"
            result += f"\t\tMax Workgroup Invocations: {self.max_workgroup_invocations}\n"
            result += f"\t\tMax Workgroup Counts: {self.max_workgroup_count}\n"
            result += f"\t\tMax Bound Descriptor Sets={self.max_bound_descriptor_sets}\n"
        
        result += f"\t\tMax Push Constant Size: {self.max_push_constant_size} bytes\n"
        
        if verbose:
            result += f"\t\tMax Storage Buffer Range: {self.max_storage_buffer_range} bytes\n"
            result += f"\t\tMax Uniform Buffer Range: {self.max_uniform_buffer_range} bytes\n"
            result += f"\t\tUniform Buffer Alignment: {self.uniform_buffer_alignment}\n"
        
        result += f"\t\tSubgroup Size: {self.sub_group_size}\n"

        if verbose:
            result += f"\t\tSubgroup Operations Supported Stages: {hex(self.supported_stages)}\n"
            result += f"\t\tSubgroup Operations Supported Operations: {hex(self.supported_operations)}\n"

        result += f"\t\tMax Compute Shared Memory Size: {self.max_compute_shared_memory_size}\n"
        
        result += f"\n\tQueues:\n"
        for ii, queue in enumerate(self.queue_properties):
            queue_types = get_queue_type_strings(queue[1], verbose)

            if len(queue_types) != 0:
                result += f"\t\t{ii} (count={queue[0]}, flags={hex(queue[1])}): "
                result += " | ".join(queue_types) + "\n"

        return result
    
    def __repr__(self) -> str:
        return self.get_info_string()

__initilized_instance: bool = False


def is_initialized() -> bool:
    """
    A function which checks if the Vulkan dispatch library has been initialized.

    Returns:
        `bool`: A flag indicating whether the Vulkan dispatch library has been initialized.
    """

    global __initilized_instance

    return __initilized_instance

def initialize(debug_mode: bool = False, log_level: LogLevel = LogLevel.WARNING, loader_debug_logs: bool = False):
    """
    A function which initializes the Vulkan dispatch library.

    Args:
        debug_mode (`bool`): A flag to enable debug mode.
        log_level (`LogLevel`): The log level, which is one of the following:
            LogLevel.VERBOSE
            LogLevel.INFO
            LogLevel.WARNING
            LogLevel.ERROR
        loader_debug_logs (bool): A flag to enable vulkan loader debug logs.
    """

    global __initilized_instance

    if __initilized_instance:
        return
    
    if loader_debug_logs:
        os.environ["VK_LOADER_DEBUG"] = "all"

    vkdispatch_native.init(debug_mode, log_level.value)
    check_for_errors()
    
    __initilized_instance = True


def get_devices() -> typing.List[DeviceInfo]:
    """
    Get a list of DeviceInfo instances representing all the Vulkan devices on the system.

    Returns:
        `List[DeviceInfo]`: A list of DeviceInfo instances.
    """

    initialize()

    return [
        DeviceInfo(ii, *dev_obj)
        for ii, dev_obj in enumerate(vkdispatch_native.get_devices())
    ]

def log(text: str, end: str = '\n', level: LogLevel = LogLevel.ERROR, stack_offset: int = 1):
    """
    A function which logs a message at the specified log level.

    Args:
        level (`LogLevel`): The log level.
        message (`str`): The message to log.
    """

    initialize()

    frame = inspect.stack()[stack_offset]
    vkdispatch_native.log(
        level.value, 
        (text + end).encode(), 
        os.path.relpath(frame.filename, os.getcwd()).encode(), 
        frame.lineno
    )

def log_error(text: str, end: str = '\n'):
    """
    A function which logs an error message.

    Args:
        message (`str`): The message to log.
    """

    log(text, end, LogLevel.ERROR, 2)

def log_warning(text: str, end: str = '\n'):
    """
    A function which logs a warning message.

    Args:
        message (`str`): The message to log.
    """

    log(text, end, LogLevel.WARNING, 2)

def log_info(text: str, end: str = '\n'):
    """
    A function which logs an info message.

    Args:
        message (`str`): The message to log.
    """

    log(text, end, LogLevel.INFO, 2)

def log_verbose(text: str, end: str = '\n'):
    """
    A function which logs a verbose message.

    Args:
        message (`str`): The message to log.
    """

    log(text, end, LogLevel.VERBOSE, 2)
