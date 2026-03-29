"""Brython-friendly pure-Python shim for ``vkdispatch_native``.

This module mirrors the Cython-exposed API used by ``vkdispatch`` and provides
dummy metadata helpers for docs/codegen flows.

Runtime GPU operations are intentionally denied so the dummy backend fails fast
when used outside codegen-only scripts.
"""

# --- Runtime state ---

_initialized = False
_debug_mode = False
_log_level = 2
_error_string = None
_next_handle = 1

_contexts = {}
_signals = {}

# Device limits exposed through get_devices(); mutable so docs UI can tune them.
_DEFAULT_SUBGROUP_SIZE = 32
_DEFAULT_MAX_WORKGROUP_SIZE = (1024, 1024, 64)
_DEFAULT_MAX_WORKGROUP_INVOCATIONS = 1024
_DEFAULT_MAX_WORKGROUP_COUNT = (65535, 65535, 65535)
_DEFAULT_MAX_COMPUTE_SHARED_MEMORY_SIZE = 64 * 1024

_device_subgroup_size = _DEFAULT_SUBGROUP_SIZE
_device_subgroup_enabled = True
_device_max_workgroup_size = _DEFAULT_MAX_WORKGROUP_SIZE
_device_max_workgroup_invocations = _DEFAULT_MAX_WORKGROUP_INVOCATIONS
_device_max_workgroup_count = _DEFAULT_MAX_WORKGROUP_COUNT
_device_max_compute_shared_memory_size = _DEFAULT_MAX_COMPUTE_SHARED_MEMORY_SIZE


# --- Internal objects ---

class _Signal:
    __slots__ = ("done",)

    def __init__(self, done=True):
        self.done = bool(done)


class _Context:
    __slots__ = (
        "device_indices",
        "queue_families",
        "queue_count",
        "queue_to_device",
        "stopped",
    )

    def __init__(self, device_indices, queue_families):
        self.device_indices = list(device_indices)
        self.queue_families = [list(fam) for fam in queue_families]

        normalized = []
        for fam in self.queue_families:
            normalized.append(fam if len(fam) > 0 else [0])
        self.queue_families = normalized

        self.queue_count = sum(len(fam) for fam in self.queue_families)
        if self.queue_count <= 0:
            self.queue_families = [[0]]
            self.queue_count = 1

        queue_to_device = []
        for dev_idx, fam in enumerate(self.queue_families):
            for _ in fam:
                queue_to_device.append(dev_idx)

        if len(queue_to_device) == 0:
            queue_to_device = [0]

        self.queue_to_device = queue_to_device
        self.stopped = False

# --- Internal helpers ---

def _new_handle(registry, obj):
    global _next_handle
    handle = _next_handle
    _next_handle += 1
    registry[handle] = obj
    return handle

def _set_error(message):
    global _error_string
    _error_string = str(message)


def _clear_error():
    global _error_string
    _error_string = None


_DUMMY_CODEGEN_ONLY_ERROR = (
    "The 'dummy' backend is codegen-only and does not support runtime GPU "
    "operations. Use backend='vulkan', backend='cuda', or backend='opencl' for execution."
)


def _deny_runtime_native_call(function_name):
    raise RuntimeError(f"{_DUMMY_CODEGEN_ONLY_ERROR} (native call: {function_name})")


def _as_positive_int(name, value):
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError("%s must be an integer" % name) from exc

    if parsed <= 0:
        raise ValueError("%s must be greater than zero" % name)

    return parsed


def _as_positive_triplet(name, value):
    try:
        parts = list(value)
    except Exception as exc:
        raise ValueError("%s must contain exactly 3 integers" % name) from exc

    if len(parts) != 3:
        raise ValueError("%s must contain exactly 3 integers" % name)

    return (
        _as_positive_int("%s[0]" % name, parts[0]),
        _as_positive_int("%s[1]" % name, parts[1]),
        _as_positive_int("%s[2]" % name, parts[2]),
    )


# --- API: context/init/errors/logging ---


def reset_device_options():
    global _device_subgroup_size
    global _device_subgroup_enabled
    global _device_max_workgroup_size
    global _device_max_workgroup_invocations
    global _device_max_workgroup_count
    global _device_max_compute_shared_memory_size

    _device_subgroup_size = _DEFAULT_SUBGROUP_SIZE
    _device_subgroup_enabled = True
    _device_max_workgroup_size = _DEFAULT_MAX_WORKGROUP_SIZE
    _device_max_workgroup_invocations = _DEFAULT_MAX_WORKGROUP_INVOCATIONS
    _device_max_workgroup_count = _DEFAULT_MAX_WORKGROUP_COUNT
    _device_max_compute_shared_memory_size = _DEFAULT_MAX_COMPUTE_SHARED_MEMORY_SIZE


def set_device_options(
    subgroup_size=None,
    subgroup_enabled=None,
    max_workgroup_size=None,
    max_workgroup_invocations=None,
    max_workgroup_count=None,
    max_compute_shared_memory_size=None,
):
    global _device_subgroup_size
    global _device_subgroup_enabled
    global _device_max_workgroup_size
    global _device_max_workgroup_invocations
    global _device_max_workgroup_count
    global _device_max_compute_shared_memory_size

    if subgroup_size is not None:
        _device_subgroup_size = _as_positive_int("subgroup_size", subgroup_size)

    if subgroup_enabled is not None:
        if not isinstance(subgroup_enabled, bool):
            raise ValueError("subgroup_enabled must be a boolean value")
        _device_subgroup_enabled = subgroup_enabled

    if max_workgroup_size is not None:
        _device_max_workgroup_size = _as_positive_triplet(
            "max_workgroup_size",
            max_workgroup_size,
        )

    if max_workgroup_invocations is not None:
        _device_max_workgroup_invocations = _as_positive_int(
            "max_workgroup_invocations",
            max_workgroup_invocations,
        )

    if max_workgroup_count is not None:
        _device_max_workgroup_count = _as_positive_triplet(
            "max_workgroup_count",
            max_workgroup_count,
        )

    if max_compute_shared_memory_size is not None:
        _device_max_compute_shared_memory_size = _as_positive_int(
            "max_compute_shared_memory_size",
            max_compute_shared_memory_size,
        )


def init(debug, log_level):
    global _initialized, _debug_mode, _log_level
    _initialized = True
    _debug_mode = bool(debug)
    _log_level = int(log_level)
    _clear_error()


def log(log_level, text, file_str, line_str):
    # Keep logging quiet in docs/brython by default.
    # Function kept for API compatibility.
    _ = log_level
    _ = text
    _ = file_str
    _ = line_str


def set_log_level(log_level):
    global _log_level
    _log_level = int(log_level)


def get_devices():
    if not _initialized:
        init(False, _log_level)

    # One plausible fake discrete GPU with compute+graphics queue families.
    device_tuple = (
        0,  # version_variant
        1,  # version_major
        3,  # version_minor
        0,  # version_patch
        1001000,  # driver_version
        0x1BAD,  # vendor_id
        0x0001,  # device_id
        2,  # device_type (Discrete GPU)
        "VKDispatch Web Dummy GPU",
        1,  # shader_buffer_float32_atomics
        1,  # shader_buffer_float32_atomic_add
        1,  # float_64_support
        1,  # float_16_support
        1,  # int_64_support
        1,  # int_16_support
        1,  # storage_buffer_16_bit_access
        1,  # uniform_and_storage_buffer_16_bit_access
        1,  # storage_push_constant_16
        1,  # storage_input_output_16
        _device_max_workgroup_size,  # max_workgroup_size
        _device_max_workgroup_invocations,  # max_workgroup_invocations
        _device_max_workgroup_count,  # max_workgroup_count
        8,  # max_descriptor_set_count
        256,  # max_push_constant_size
        1 << 30,  # max_storage_buffer_range
        65536,  # max_uniform_buffer_range
        0,  # uniform_buffer_alignment
        _device_subgroup_size,  # subgroup_size
        0x7FFFFFFF if _device_subgroup_enabled else 0,  # supported_stages
        0x7FFFFFFF if _device_subgroup_enabled else 0,  # supported_operations
        1,  # quad_operations_in_all_stages
        _device_max_compute_shared_memory_size,  # max_compute_shared_memory_size
        [
            (8, 0x006),  # compute + transfer
            (4, 0x007),  # graphics + compute + transfer
        ],
        1,  # scalar_block_layout
        1,  # timeline_semaphores
        bytes((0x56, 0x4B, 0x44, 0x30, 0x57, 0x45, 0x42, 0x31, 0x44, 0x55, 0x4D, 0x4D, 0x59, 0x00, 0x00, 0x01)),
    )

    return [device_tuple]


def context_create(device_indicies, queue_families):
    try:
        ctx = _Context(device_indicies, queue_families)
        return _new_handle(_contexts, ctx)
    except Exception as exc:
        _set_error("Failed to create context: %s" % exc)
        return 0


def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    _ = wait_for_timestamp
    _ = queue_index
    signal_obj = _signals.get(int(signal_ptr))
    if signal_obj is None:
        return True
    return bool(signal_obj.done)


def signal_insert(context, queue_index):
    _ = context
    _ = queue_index
    return _new_handle(_signals, _Signal(done=True))


def signal_destroy(signal_ptr):
    _signals.pop(int(signal_ptr), None)


def context_destroy(context):
    _contexts.pop(int(context), None)


def get_error_string():
    if _error_string is None:
        return 0
    return _error_string


def context_stop_threads(context):
    ctx = _contexts.get(int(context))
    if ctx is not None:
        ctx.stopped = True


# --- API: buffers ---


def buffer_create(context, size, per_device):
    _deny_runtime_native_call("buffer_create")


def buffer_destroy(buffer):
    _deny_runtime_native_call("buffer_destroy")


def buffer_get_queue_signal(buffer, queue_index):
    _deny_runtime_native_call("buffer_get_queue_signal")


def buffer_wait_staging_idle(buffer, queue_index):
    _deny_runtime_native_call("buffer_wait_staging_idle")


def buffer_write_staging(buffer, queue_index, data, size):
    _deny_runtime_native_call("buffer_write_staging")


def buffer_read_staging(buffer, queue_index, size):
    _deny_runtime_native_call("buffer_read_staging")


def buffer_write(buffer, offset, size, index):
    _deny_runtime_native_call("buffer_write")


def buffer_read(buffer, offset, size, index):
    _deny_runtime_native_call("buffer_read")


# --- API: command lists ---


def command_list_create(context):
    _deny_runtime_native_call("command_list_create")


def command_list_destroy(command_list):
    _deny_runtime_native_call("command_list_destroy")


def command_list_get_instance_size(command_list):
    _deny_runtime_native_call("command_list_get_instance_size")


def command_list_reset(command_list):
    _deny_runtime_native_call("command_list_reset")


def command_list_submit(command_list, data, instance_count, index):
    _deny_runtime_native_call("command_list_submit")


# --- API: descriptor sets ---


def descriptor_set_create(plan):
    _deny_runtime_native_call("descriptor_set_create")


def descriptor_set_destroy(descriptor_set):
    _deny_runtime_native_call("descriptor_set_destroy")


def descriptor_set_write_buffer(
    descriptor_set,
    binding,
    object,
    offset,
    range,
    uniform,
    read_access,
    write_access,
):
    _deny_runtime_native_call("descriptor_set_write_buffer")


def descriptor_set_write_image(
    descriptor_set,
    binding,
    object,
    sampler_obj,
    read_access,
    write_access,
):
    _deny_runtime_native_call("descriptor_set_write_image")


# --- API: images/samplers ---


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    _deny_runtime_native_call("image_create")


def image_destroy(image):
    _deny_runtime_native_call("image_destroy")


def image_create_sampler(
    context,
    mag_filter,
    min_filter,
    mip_mode,
    address_mode,
    mip_lod_bias,
    min_lod,
    max_lod,
    border_color,
):
    _deny_runtime_native_call("image_create_sampler")


def image_destroy_sampler(sampler):
    _deny_runtime_native_call("image_destroy_sampler")


def image_write(image, data, offset, extent, baseLayer, layerCount, device_index):
    _deny_runtime_native_call("image_write")


def image_format_block_size(format):
    _deny_runtime_native_call("image_format_block_size")


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _deny_runtime_native_call("image_read")


# --- API: compute stage ---


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    _deny_runtime_native_call("stage_compute_plan_create")


def stage_compute_plan_destroy(plan):
    _deny_runtime_native_call("stage_compute_plan_destroy")


def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    _deny_runtime_native_call("stage_compute_record")


# --- API: FFT stage ---


def stage_fft_plan_create(
    context,
    dims,
    axes,
    buffer_size,
    do_r2c,
    normalize,
    pad_left,
    pad_right,
    frequency_zeropadding,
    kernel_num,
    kernel_convolution,
    conjugate_convolution,
    convolution_features,
    input_buffer_size,
    num_batches,
    single_kernel_multiple_batches,
    keep_shader_code,
):
    _deny_runtime_native_call("stage_fft_plan_create")


def stage_fft_plan_destroy(plan):
    _deny_runtime_native_call("stage_fft_plan_destroy")


def stage_fft_record(command_list, plan, buffer, inverse, kernel, input_buffer):
    _deny_runtime_native_call("stage_fft_record")


__all__ = [
    "reset_device_options",
    "set_device_options",
    "init",
    "log",
    "set_log_level",
    "get_devices",
    "context_create",
    "signal_wait",
    "signal_insert",
    "signal_destroy",
    "context_destroy",
    "get_error_string",
    "context_stop_threads",
    "buffer_create",
    "buffer_destroy",
    "buffer_get_queue_signal",
    "buffer_wait_staging_idle",
    "buffer_write_staging",
    "buffer_read_staging",
    "buffer_write",
    "buffer_read",
    "command_list_create",
    "command_list_destroy",
    "command_list_get_instance_size",
    "command_list_reset",
    "command_list_submit",
    "descriptor_set_create",
    "descriptor_set_destroy",
    "descriptor_set_write_buffer",
    "descriptor_set_write_image",
    "image_create",
    "image_destroy",
    "image_create_sampler",
    "image_destroy_sampler",
    "image_write",
    "image_format_block_size",
    "image_read",
    "stage_compute_plan_create",
    "stage_compute_plan_destroy",
    "stage_compute_record",
    "stage_fft_plan_create",
    "stage_fft_plan_destroy",
    "stage_fft_record"
]
