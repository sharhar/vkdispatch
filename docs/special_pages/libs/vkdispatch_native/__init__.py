"""Brython-friendly pure-Python shim for ``vkdispatch_native``.

This module mirrors the Cython-exposed API used by ``vkdispatch`` and provides
an in-memory fake runtime suitable for docs execution and shader-source
compilation paths.
"""

# NOTE: Keep this file dependency-light so it works under Brython.

LOG_LEVEL_VERBOSE = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

# Descriptor type enum values mirrored from vkdispatch_native/stages_extern.pxd.
DESCRIPTOR_TYPE_STORAGE_BUFFER = 1
DESCRIPTOR_TYPE_STORAGE_IMAGE = 2
DESCRIPTOR_TYPE_UNIFORM_BUFFER = 3
DESCRIPTOR_TYPE_UNIFORM_IMAGE = 4
DESCRIPTOR_TYPE_SAMPLER = 5

# Image format block sizes for formats exposed in vkdispatch.base.image.image_format.
_IMAGE_BLOCK_SIZES = {
    13: 1,
    14: 1,
    20: 2,
    21: 2,
    27: 3,
    28: 3,
    41: 4,
    42: 4,
    74: 2,
    75: 2,
    76: 2,
    81: 4,
    82: 4,
    83: 4,
    88: 6,
    89: 6,
    90: 6,
    95: 8,
    96: 8,
    97: 8,
    98: 4,
    99: 4,
    100: 4,
    101: 8,
    102: 8,
    103: 8,
    104: 12,
    105: 12,
    106: 12,
    107: 16,
    108: 16,
    109: 16,
    110: 8,
    111: 8,
    112: 8,
    113: 16,
    114: 16,
    115: 16,
    116: 24,
    117: 24,
    118: 24,
    119: 32,
    120: 32,
    121: 32,
}

# --- Runtime state ---

_initialized = False
_debug_mode = False
_log_level = LOG_LEVEL_WARNING
_error_string = None
_next_handle = 1

_contexts = {}
_signals = {}
_buffers = {}
_command_lists = {}
_compute_plans = {}
_descriptor_sets = {}
_images = {}
_samplers = {}
_fft_plans = {}


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


class _Buffer:
    __slots__ = (
        "context_handle",
        "size",
        "device_data",
        "staging_data",
        "signal_handles",
    )

    def __init__(self, context_handle, queue_count, size):
        self.context_handle = context_handle
        self.size = int(size)

        if queue_count <= 0:
            queue_count = 1

        self.device_data = [bytearray(self.size) for _ in range(queue_count)]
        self.staging_data = [bytearray(self.size) for _ in range(queue_count)]

        signal_handles = []
        for _ in range(queue_count):
            signal_handles.append(_new_handle(_signals, _Signal(done=True)))
        self.signal_handles = signal_handles


class _CommandList:
    __slots__ = ("context_handle", "commands", "compute_instance_size")

    def __init__(self, context_handle):
        self.context_handle = context_handle
        self.commands = []
        self.compute_instance_size = 0


class _ComputePlan:
    __slots__ = ("context_handle", "shader_source", "bindings", "pc_size", "shader_name")

    def __init__(self, context_handle, shader_source, bindings, pc_size, shader_name):
        self.context_handle = context_handle
        self.shader_source = shader_source
        self.bindings = list(bindings)
        self.pc_size = int(pc_size)
        self.shader_name = shader_name


class _DescriptorSet:
    __slots__ = ("plan_handle", "buffer_bindings", "image_bindings")

    def __init__(self, plan_handle):
        self.plan_handle = plan_handle
        self.buffer_bindings = {}
        self.image_bindings = {}


class _Image:
    __slots__ = (
        "context_handle",
        "extent",
        "layers",
        "format",
        "type",
        "view_type",
        "generate_mips",
        "block_size",
        "queue_data",
    )

    def __init__(
        self,
        context_handle,
        queue_count,
        extent,
        layers,
        format_,
        image_type,
        view_type,
        generate_mips,
    ):
        self.context_handle = context_handle
        self.extent = tuple(extent)
        self.layers = int(layers)
        self.format = int(format_)
        self.type = int(image_type)
        self.view_type = int(view_type)
        self.generate_mips = int(generate_mips)

        self.block_size = image_format_block_size(self.format)

        if queue_count <= 0:
            queue_count = 1

        width = max(1, int(self.extent[0]))
        height = max(1, int(self.extent[1]))
        depth = max(1, int(self.extent[2]))
        layer_count = max(1, self.layers)
        total_bytes = width * height * depth * layer_count * self.block_size

        self.queue_data = [bytearray(total_bytes) for _ in range(queue_count)]


class _Sampler:
    __slots__ = (
        "context_handle",
        "mag_filter",
        "min_filter",
        "mip_mode",
        "address_mode",
        "mip_lod_bias",
        "min_lod",
        "max_lod",
        "border_color",
    )

    def __init__(
        self,
        context_handle,
        mag_filter,
        min_filter,
        mip_mode,
        address_mode,
        mip_lod_bias,
        min_lod,
        max_lod,
        border_color,
    ):
        self.context_handle = context_handle
        self.mag_filter = int(mag_filter)
        self.min_filter = int(min_filter)
        self.mip_mode = int(mip_mode)
        self.address_mode = int(address_mode)
        self.mip_lod_bias = float(mip_lod_bias)
        self.min_lod = float(min_lod)
        self.max_lod = float(max_lod)
        self.border_color = int(border_color)


class _FFTPlan:
    __slots__ = (
        "context_handle",
        "dims",
        "axes",
        "buffer_size",
        "input_buffer_size",
        "kernel_num",
    )

    def __init__(
        self,
        context_handle,
        dims,
        axes,
        buffer_size,
        input_buffer_size,
        kernel_num,
    ):
        self.context_handle = context_handle
        self.dims = list(dims)
        self.axes = list(axes)
        self.buffer_size = int(buffer_size)
        self.input_buffer_size = int(input_buffer_size)
        self.kernel_num = int(kernel_num)


# --- Internal helpers ---


def _new_handle(registry, obj):
    global _next_handle
    handle = _next_handle
    _next_handle += 1
    registry[handle] = obj
    return handle


def _to_bytes(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    try:
        return bytes(value)
    except Exception:
        return b""


def _normalize_extent(extent):
    values = list(extent)
    if len(values) < 3:
        values.extend([1] * (3 - len(values)))
    return (int(values[0]), int(values[1]), int(values[2]))


def _queue_indices(ctx, queue_index, all_on_negative=False):
    if ctx is None or ctx.queue_count <= 0:
        return []

    if queue_index is None:
        return [0]

    queue_index = int(queue_index)

    if all_on_negative and queue_index in (-1, -2):
        return list(range(ctx.queue_count))

    if 0 <= queue_index < ctx.queue_count:
        return [queue_index]

    return []


def _set_error(message):
    global _error_string
    _error_string = str(message)


def _clear_error():
    global _error_string
    _error_string = None


# --- API: context/init/errors/logging ---


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
        (1024, 1024, 64),  # max_workgroup_size
        1024,  # max_workgroup_invocations
        (65535, 65535, 65535),  # max_workgroup_count
        8,  # max_descriptor_set_count
        256,  # max_push_constant_size
        1 << 30,  # max_storage_buffer_range
        65536,  # max_uniform_buffer_range
        16,  # uniform_buffer_alignment
        32,  # subgroup_size
        0x7FFFFFFF,  # supported_stages
        0x7FFFFFFF,  # supported_operations
        1,  # quad_operations_in_all_stages
        64 * 1024,  # max_compute_shared_memory_size
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
    _ = per_device
    ctx = _contexts.get(int(context))
    if ctx is None:
        _set_error("Invalid context handle for buffer_create")
        return 0

    size = int(size)
    if size < 0:
        size = 0

    return _new_handle(_buffers, _Buffer(int(context), ctx.queue_count, size))


def buffer_destroy(buffer):
    obj = _buffers.pop(int(buffer), None)
    if obj is None:
        return

    for signal_handle in obj.signal_handles:
        _signals.pop(signal_handle, None)


def buffer_get_queue_signal(buffer, queue_index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return _new_handle(_signals, _Signal(done=True))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.signal_handles):
        queue_index = 0

    return obj.signal_handles[queue_index]


def buffer_wait_staging_idle(buffer, queue_index):
    _ = buffer
    _ = queue_index
    return True


def buffer_write_staging(buffer, queue_index, data, size):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.staging_data):
        return

    payload = _to_bytes(data)
    size = min(int(size), len(payload), obj.size)
    if size <= 0:
        return

    obj.staging_data[queue_index][:size] = payload[:size]


def buffer_read_staging(buffer, queue_index, size):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return bytes(int(size))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.staging_data):
        return bytes(int(size))

    size = int(size)
    if size <= 0:
        return b""

    data = obj.staging_data[queue_index]
    if size <= len(data):
        return bytes(data[:size])

    return bytes(data) + bytes(size - len(data))


def buffer_write(buffer, offset, size, index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return

    offset = int(offset)
    size = int(size)

    if size <= 0 or offset < 0:
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        return

    queue_indices = _queue_indices(ctx, index, all_on_negative=True)
    if len(queue_indices) == 0:
        return

    for queue_index in queue_indices:
        if queue_index >= len(obj.device_data) or queue_index >= len(obj.staging_data):
            continue

        end = min(offset + size, obj.size)
        copy_size = end - offset
        if copy_size <= 0:
            continue

        obj.device_data[queue_index][offset:end] = obj.staging_data[queue_index][:copy_size]

        signal_handle = obj.signal_handles[queue_index]
        signal_obj = _signals.get(signal_handle)
        if signal_obj is not None:
            signal_obj.done = True


def buffer_read(buffer, offset, size, index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return

    offset = int(offset)
    size = int(size)

    if size <= 0 or offset < 0:
        return

    queue_index = int(index)
    if queue_index < 0 or queue_index >= len(obj.device_data):
        return

    end = min(offset + size, obj.size)
    copy_size = end - offset
    if copy_size <= 0:
        return

    obj.staging_data[queue_index][:copy_size] = obj.device_data[queue_index][offset:end]

    signal_handle = obj.signal_handles[queue_index]
    signal_obj = _signals.get(signal_handle)
    if signal_obj is not None:
        signal_obj.done = True


# --- API: command lists ---


def command_list_create(context):
    if int(context) not in _contexts:
        _set_error("Invalid context handle for command_list_create")
        return 0

    return _new_handle(_command_lists, _CommandList(int(context)))


def command_list_destroy(command_list):
    _command_lists.pop(int(command_list), None)


def command_list_get_instance_size(command_list):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return 0

    return int(obj.compute_instance_size)


def command_list_reset(command_list):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return

    obj.commands = []
    obj.compute_instance_size = 0


def command_list_submit(command_list, data, instance_count, index):
    _ = data
    _ = instance_count
    _ = index

    obj = _command_lists.get(int(command_list))
    if obj is None:
        return True

    # No-op fake execution path: commands are accepted but not executed.
    # Keep the command list intact (native keeps it until reset/destroy).
    _ = obj.commands
    return True


# --- API: descriptor sets ---


def descriptor_set_create(plan):
    if int(plan) not in _compute_plans:
        _set_error("Invalid compute plan handle for descriptor_set_create")
        return 0

    return _new_handle(_descriptor_sets, _DescriptorSet(int(plan)))


def descriptor_set_destroy(descriptor_set):
    _descriptor_sets.pop(int(descriptor_set), None)


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
    ds = _descriptor_sets.get(int(descriptor_set))
    if ds is None:
        return

    ds.buffer_bindings[int(binding)] = (
        int(object),
        int(offset),
        int(range),
        int(uniform),
        int(read_access),
        int(write_access),
    )


def descriptor_set_write_image(
    descriptor_set,
    binding,
    object,
    sampler_obj,
    read_access,
    write_access,
):
    ds = _descriptor_sets.get(int(descriptor_set))
    if ds is None:
        return

    ds.image_bindings[int(binding)] = (
        int(object),
        int(sampler_obj),
        int(read_access),
        int(write_access),
    )


# --- API: images/samplers ---


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    ctx = _contexts.get(int(context))
    if ctx is None:
        _set_error("Invalid context handle for image_create")
        return 0

    norm_extent = _normalize_extent(extent)
    obj = _Image(
        int(context),
        ctx.queue_count,
        norm_extent,
        int(layers),
        int(format),
        int(type),
        int(view_type),
        int(generate_mips),
    )

    return _new_handle(_images, obj)


def image_destroy(image):
    _images.pop(int(image), None)


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
    if int(context) not in _contexts:
        _set_error("Invalid context handle for image_create_sampler")
        return 0

    sampler = _Sampler(
        int(context),
        mag_filter,
        min_filter,
        mip_mode,
        address_mode,
        mip_lod_bias,
        min_lod,
        max_lod,
        border_color,
    )
    return _new_handle(_samplers, sampler)


def image_destroy_sampler(sampler):
    _samplers.pop(int(sampler), None)


def image_write(image, data, offset, extent, baseLayer, layerCount, device_index):
    _ = offset
    _ = baseLayer

    obj = _images.get(int(image))
    if obj is None:
        return

    payload = _to_bytes(data)

    extent = _normalize_extent(extent)
    layer_count = max(1, int(layerCount))
    region_size = max(0, extent[0] * extent[1] * extent[2] * layer_count * obj.block_size)
    if region_size <= 0:
        return

    copy_size = min(region_size, len(payload))
    if copy_size <= 0:
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        return

    queue_indices = _queue_indices(ctx, device_index, all_on_negative=True)
    if len(queue_indices) == 0:
        return

    for queue_index in queue_indices:
        if queue_index < 0 or queue_index >= len(obj.queue_data):
            continue
        obj.queue_data[queue_index][:copy_size] = payload[:copy_size]


def image_format_block_size(format):
    return int(_IMAGE_BLOCK_SIZES.get(int(format), 4))


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount

    obj = _images.get(int(image))
    out_size = max(0, int(out_size))

    if obj is None:
        return bytes(out_size)

    queue_index = int(device_index)
    if queue_index < 0 or queue_index >= len(obj.queue_data):
        queue_index = 0

    data = obj.queue_data[queue_index]
    if out_size <= len(data):
        return bytes(data[:out_size])

    return bytes(data) + bytes(out_size - len(data))


# --- API: compute stage ---


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    if int(context) not in _contexts:
        _set_error("Invalid context handle for stage_compute_plan_create")
        return 0

    source_bytes = _to_bytes(shader_source)
    name_bytes = _to_bytes(shader_name)

    plan = _ComputePlan(int(context), source_bytes, list(bindings), int(pc_size), name_bytes)
    return _new_handle(_compute_plans, plan)


def stage_compute_plan_destroy(plan):
    _compute_plans.pop(int(plan), None)


def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    cl = _command_lists.get(int(command_list))
    cp = _compute_plans.get(int(plan))

    if cl is None or cp is None:
        return

    cl.commands.append(
        {
            "type": "compute",
            "plan": int(plan),
            "descriptor_set": int(descriptor_set),
            "blocks": (int(blocks_x), int(blocks_y), int(blocks_z)),
        }
    )
    cl.compute_instance_size += max(0, int(cp.pc_size))


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
    _ = do_r2c
    _ = normalize
    _ = pad_left
    _ = pad_right
    _ = frequency_zeropadding
    _ = kernel_convolution
    _ = conjugate_convolution
    _ = convolution_features
    _ = num_batches
    _ = single_kernel_multiple_batches
    _ = keep_shader_code

    if int(context) not in _contexts:
        _set_error("Invalid context handle for stage_fft_plan_create")
        return 0

    plan = _FFTPlan(
        int(context),
        list(dims),
        list(axes),
        int(buffer_size),
        int(input_buffer_size),
        int(kernel_num),
    )

    return _new_handle(_fft_plans, plan)


def stage_fft_plan_destroy(plan):
    _fft_plans.pop(int(plan), None)


def stage_fft_record(command_list, plan, buffer, inverse, kernel, input_buffer):
    _ = buffer
    _ = inverse
    _ = kernel
    _ = input_buffer

    cl = _command_lists.get(int(command_list))
    if cl is None or int(plan) not in _fft_plans:
        return

    cl.commands.append(
        {
            "type": "fft",
            "plan": int(plan),
        }
    )


__all__ = [
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
    "stage_fft_record",
    "LOG_LEVEL_VERBOSE",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_WARNING",
    "LOG_LEVEL_ERROR",
    "DESCRIPTOR_TYPE_STORAGE_BUFFER",
    "DESCRIPTOR_TYPE_STORAGE_IMAGE",
    "DESCRIPTOR_TYPE_UNIFORM_BUFFER",
    "DESCRIPTOR_TYPE_UNIFORM_IMAGE",
    "DESCRIPTOR_TYPE_SAMPLER",
]
