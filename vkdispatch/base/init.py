
from enum import Enum
import os
from typing import Tuple, List, Optional

import inspect

from .errors import check_for_errors
from .backend import (
    BACKEND_PYCUDA,
    BACKEND_VULKAN,
    BackendUnavailableError,
    clear_active_backend,
    get_active_backend_name,
    get_backend_module,
    native,
    normalize_backend_name,
    set_active_backend,
)

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

def get_queue_type_strings(queue_type: int, verbose: bool) -> List[str]:
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
        float_16_support: int,
        int_64_support: int,
        int_16_support: int,
        storage_buffer_16_bit_access: int, 
        uniform_and_storage_buffer_16_bit_access: int,
        storage_push_constant_16: int,
        storage_input_output_16: int,
        max_workgroup_size: Tuple[int, int, int],
        max_workgroup_invocations: int,
        max_workgroup_count: Tuple[int, int, int],
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
        queue_properties: List[Tuple[int, int]],
        scalar_block_layout: int,
        timeline_semaphores: int,
        uuid: Optional[bytes],
    ):
        self.dev_index = dev_index
        self.sorted_index = -1  # to be set later

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
        self.float_16_support = float_16_support
        self.int_64_support = int_64_support
        self.int_16_support = int_16_support

        self.storage_buffer_16_bit_access = storage_buffer_16_bit_access
        self.uniform_and_storage_buffer_16_bit_access = uniform_and_storage_buffer_16_bit_access
        self.storage_push_constant_16 = storage_push_constant_16
        self.storage_input_output_16 = storage_input_output_16

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

        self.scalar_block_layout = scalar_block_layout
        self.timeline_semaphores = timeline_semaphores
        self.uuid = uuid

    def is_nvidia(self) -> bool:
        """
        A method which checks if the device is an NVIDIA device.

        Returns:
            `bool`: A flag indicating whether the device is an NVIDIA device.
        """
        return "NVIDIA" in self.device_name
    
    def is_apple(self) -> bool:
        """
        A method which checks if the device is an Apple device.

        Returns:
            `bool`: A flag indicating whether the device is an Apple device.
        """
        return "Apple" in self.device_name

    def get_info_string(self, verbose: bool = False) -> str:
        """
        A method which returns a string representation of the device information.

        Args:
            verbose (`bool`): A flag to enable verbose output.
        
        Returns:
            str: A string representation of the device information.
        """

        result = f"Device {self.sorted_index}: {self.device_name}\n"

        result += f"\tVulkan Version: {self.version_major}.{self.version_minor}.{self.version_patch}\n"
        result += f"\tDevice Type: {device_type_id_to_str_dict[self.device_type]}\n"

        if self.version_variant != 0:
            result += f"\tVariant: {self.version_variant}\n"

        if verbose:
            result += f"\tDriver Version={self.driver_version}\n"
            result += f"\tVendor ID={self.vendor_id}\n"
            result += f"\tDevice ID={self.device_id}\n"


            if self.uuid is not None:
                uuid_str = '-'.join([
                    self.uuid[0:4].hex(),
                    self.uuid[4:6].hex(),
                    self.uuid[6:8].hex(),
                    self.uuid[8:10].hex(),
                    self.uuid[10:16].hex(),
                ])
                result += f"\tUUID: {uuid_str}\n"

        result += "\n\tFeatures:\n"

        if verbose:
            result += f"\t\tFloat32 Atomics: {self.shader_buffer_float32_atomics == 1}\n"
            result += f"\t\tScalar Block Layout: {self.scalar_block_layout == 1}\n"
            result += f"\t\tTimeline Semaphores: {self.timeline_semaphores == 1}\n"
        
        result += f"\t\tFloat32 Atomic Add: {self.shader_buffer_float32_atomic_add == 1}\n"

        result += "\n\tProperties:\n"

        result += f"\t\t64-bit Float Support: {self.float_64_support == 1}\n"
        result += f"\t\t16-bit Float Support: {self.float_16_support == 1}\n"
        result += f"\t\t64-bit Int Support: {self.int_64_support == 1}\n"
        result += f"\t\t16-bit Int Support: {self.int_16_support == 1}\n"
        
        if verbose:
            result += f"\t\tStorage Buffer 16-bit Access: {self.storage_buffer_16_bit_access == 1}\n"
            result += f"\t\tUniform and Storage Buffer 16-bit Access: {self.uniform_and_storage_buffer_16_bit_access == 1}\n"
            result += f"\t\tStorage Push Constant 16: {self.storage_push_constant_16 == 1}\n"
            result += f"\t\tStorage Input Output 16: {self.storage_input_output_16 == 1}\n"

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
__device_infos: List[DeviceInfo] = None
__backend_name: str = BACKEND_VULKAN

def is_initialized() -> bool:
    """
    A function which checks if the Vulkan dispatch library has been initialized.

    Returns:
        `bool`: A flag indicating whether the Vulkan dispatch library has been initialized.
    """

    global __initilized_instance

    return __initilized_instance

def get_cuda_device_map():
    """
    Returns a dict mapping CUDA device index -> UUID (bytes).
    Format: { 0: b'\x00...', 1: b'\x01...' }

    If the CUDA driver bindings are not available, returns None.
    """
    try:
        from cuda.bindings import driver
    except (ImportError, ModuleNotFoundError):
        __log_noinit("'cuda-python' not installed, skipping CUDA device matching", level=LogLevel.WARNING)
        return None

    try:
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to initialize CUDA Driver API")

        err, count = driver.cuDeviceGetCount()
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to get CUDA device count")

        uuid_map = {}

        for i in range(count):
            err, device = driver.cuDeviceGet(i)
            if err != driver.CUresult.CUDA_SUCCESS:
                continue

            err, uuid_bytes = driver.cuDeviceGetUuid(device)
            if err == driver.CUresult.CUDA_SUCCESS:
                assert len(uuid_bytes.bytes) == 16
                uuid_map[i] = uuid_bytes.bytes
    except Exception as e:
        __log_noinit(f"Error while querying CUDA devices: {e}", level=LogLevel.WARNING)
        return None

    return uuid_map


def _set_initialized_state(backend_name: str, devices: List[DeviceInfo]) -> None:
    global __initilized_instance
    global __backend_name
    global __device_infos

    __initilized_instance = True
    __backend_name = backend_name
    __device_infos = devices

    for ii, dev in enumerate(__device_infos):
        dev.sorted_index = ii


def _build_no_gpu_backend_error(vulkan_error: Exception, pycuda_error: Exception) -> RuntimeError:
    return RuntimeError(
        "vkdispatch could not find an available GPU backend.\n"
        f"Vulkan backend unavailable: {vulkan_error}\n"
        f"PyCUDA backend unavailable: {pycuda_error}\n"
        "Install the Vulkan backend with `pip install vkdispatch`, or install PyCUDA support "
        "(`pip install pycuda numpy`), or explicitly use `vd.initialize(backend='dummy')` "
        "for codegen-only workflows."
    )


def _build_vulkan_backend_error(vulkan_error: Exception) -> RuntimeError:
    return RuntimeError(
        "vkdispatch could not load the Vulkan backend.\n"
        f"Vulkan backend unavailable: {vulkan_error}\n"
        "Install the Vulkan backend with `pip install vkdispatch`, use the PyCUDA backend "
        "(`pip install pycuda numpy`, or explicitly use `vd.initialize(backend='dummy')` "
        "for codegen-only workflows."
    )


def _initialize_with_backend(
    backend_name: str,
    debug_mode: bool,
    log_level: LogLevel,
    loader_debug_logs: bool,
) -> None:
    global __initilized_instance

    set_active_backend(backend_name)

    try:
        if loader_debug_logs and backend_name == BACKEND_VULKAN:
            os.environ["VK_LOADER_DEBUG"] = "all"

        # Force import now so backend availability errors are distinct from runtime init errors.
        get_backend_module(backend_name)

        native.init(debug_mode, log_level.value)
        check_for_errors()

        devivces = [
            DeviceInfo(ii, *dev_obj)
            for ii, dev_obj in enumerate(native.get_devices())
        ]

        if backend_name != BACKEND_VULKAN:
            _set_initialized_state(backend_name, devivces)
            return

        is_cuda = any(dev.is_nvidia() for dev in devivces)
        cuda_uuids = get_cuda_device_map() if is_cuda else None

        if cuda_uuids is None:
            _set_initialized_state(backend_name, devivces)
            return

        # try to match CUDA devices to Vulkan devices by UUID
        cuda_uuid_to_index = {
            uuid_bytes: cuda_index
            for cuda_index, uuid_bytes in cuda_uuids.items()
        }
        matched_devices: List[Tuple[int, DeviceInfo]] = []
        unmatched_devices: List[DeviceInfo] = []
        for dev in devivces:
            if dev.uuid is not None and dev.uuid in cuda_uuid_to_index:
                matched_devices.append((cuda_uuid_to_index[dev.uuid], dev))
            else:
                unmatched_devices.append(dev)

        matched_devices.sort(key=lambda x: x[0])
        result = [dev for _, dev in matched_devices] + unmatched_devices

        for dev_id, dev in enumerate(result):
            dev.sorted_index = dev_id

        _set_initialized_state(backend_name, result)
    except Exception:
        if not __initilized_instance:
            clear_active_backend()
        raise

def initialize(
    debug_mode: bool = False,
    log_level: LogLevel = LogLevel.WARNING,
    loader_debug_logs: bool = False,
    backend: Optional[str] = None,
):
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
        backend (`Optional[str]`): Runtime backend to use. Supported values are
            "vulkan", "pycuda", and "dummy". If omitted, the currently selected backend is
            reused. If no backend was selected yet, `VKDISPATCH_BACKEND` is used
            when set, otherwise "vulkan" is used.
    """

    global __initilized_instance
    env_backend = os.environ.get("VKDISPATCH_BACKEND")
    backend_name = normalize_backend_name(
        backend
        if backend is not None
        else get_active_backend_name(env_backend)
    )
    backend_explicitly_selected = (backend is not None) or (env_backend is not None)

    if __initilized_instance:
        if __backend_name != backend_name:
            raise RuntimeError(
                f"vkdispatch is already initialized with backend '{__backend_name}'. "
                f"Cannot reinitialize with '{backend_name}' in the same process."
            )
        return

    if (
        not backend_explicitly_selected
        and backend_name == BACKEND_VULKAN
    ):
        try:
            _initialize_with_backend(
                BACKEND_VULKAN,
                debug_mode=debug_mode,
                log_level=log_level,
                loader_debug_logs=loader_debug_logs,
            )
            return
        except BackendUnavailableError as vulkan_error:
            try:
                _initialize_with_backend(
                    BACKEND_PYCUDA,
                    debug_mode=debug_mode,
                    log_level=log_level,
                    loader_debug_logs=loader_debug_logs,
                )
                return
            except Exception as pycuda_error:
                raise _build_no_gpu_backend_error(vulkan_error, pycuda_error) from pycuda_error

    try:
        _initialize_with_backend(
            backend_name,
            debug_mode=debug_mode,
            log_level=log_level,
            loader_debug_logs=loader_debug_logs,
        )
    except BackendUnavailableError as backend_error:
        if backend_name == BACKEND_VULKAN:
            raise _build_vulkan_backend_error(backend_error) from backend_error
        raise


def get_devices() -> List[DeviceInfo]:
    """
    Get a list of DeviceInfo instances representing all the Vulkan devices on the system.

    Returns:
        `List[DeviceInfo]`: A list of DeviceInfo instances.
    """

    global __device_infos

    initialize()
    
    return __device_infos


def get_backend() -> str:
    if __initilized_instance:
        return __backend_name

    return get_active_backend_name()

def __log_noinit(text: str, end: str = '\n', level: LogLevel = LogLevel.ERROR, stack_offset: int = 1):
    """
    A function which logs a message at the specified log level.

    Args:
        level (`LogLevel`): The log level.
        message (`str`): The message to log.
    """

    frame = inspect.stack()[stack_offset]
    native.log(
        level.value, 
        (text + end).encode(), 
        os.path.relpath(frame.filename, os.getcwd()).encode(), 
        frame.lineno
    )

def log(text: str, end: str = '\n', level: LogLevel = LogLevel.ERROR, stack_offset: int = 1):
    """
    A function which logs a message at the specified log level.

    Args:
        level (`LogLevel`): The log level.
        message (`str`): The message to log.
    """

    initialize()

    __log_noinit(text, end, level, stack_offset + 1)

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

def set_log_level(level: LogLevel):
    """
    A function which sets the log level.

    Args:
        level (`LogLevel`): The log level.
    """

    initialize()

    native.set_log_level(level.value)
