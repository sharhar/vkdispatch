"""PyCUDA-backed runtime shim mirroring the vkdispatch_native API surface.

This module intentionally matches the function names exposed by the Cython
extension so existing Python runtime objects can call into either backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import re
import threading
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as exc:  # pragma: no cover - import failure path
    raise ImportError(
        "The PyCUDA backend requires both 'pycuda' and 'numpy' to be installed."
    ) from exc


# Log level constants mirrored from native bindings.
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

_LOCAL_X_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_X\s+(\d+)")
_LOCAL_Y_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_Y\s+(\d+)")
_LOCAL_Z_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_Z\s+(\d+)")
_KERNEL_SIGNATURE_RE = re.compile(r"vkdispatch_main\s*\(([^)]*)\)", re.S)
_BINDING_PARAM_RE = re.compile(r"vkdispatch_binding_(\d+)_ptr$")
_SAMPLER_PARAM_RE = re.compile(r"vkdispatch_sampler_(\d+)$")


# --- Runtime state ---

_initialized = False
_debug_mode = False
_log_level = LOG_LEVEL_WARNING
_error_string: Optional[str] = None
_next_handle = 1

_contexts: Dict[int, "_Context"] = {}
_signals: Dict[int, "_Signal"] = {}
_buffers: Dict[int, "_Buffer"] = {}
_command_lists: Dict[int, "_CommandList"] = {}
_compute_plans: Dict[int, "_ComputePlan"] = {}
_descriptor_sets: Dict[int, "_DescriptorSet"] = {}
_images: Dict[int, object] = {}
_samplers: Dict[int, object] = {}
_fft_plans: Dict[int, object] = {}
_external_stream_cache: Dict[int, object] = {}
_stream_override = threading.local()


# --- Internal objects ---


@dataclass
class _Signal:
    context_handle: int
    queue_index: int
    event: Optional["cuda.Event"] = None
    submitted: bool = True
    done: bool = True


@dataclass
class _Context:
    device_index: int
    pycuda_context: "cuda.Context"
    streams: List["cuda.Stream"]
    queue_count: int
    queue_to_device: List[int]
    uses_primary_context: bool = False
    stopped: bool = False


@dataclass
class _Buffer:
    context_handle: int
    size: int
    device_ptr: int
    device_allocation: Optional["cuda.DeviceAllocation"]
    owns_allocation: bool
    staging_data: List[object]
    signal_handles: List[int]


@dataclass
class _CommandRecord:
    plan_handle: int
    descriptor_set_handle: int
    blocks: Tuple[int, int, int]
    pc_size: int


@dataclass
class _CommandList:
    context_handle: int
    commands: List[_CommandRecord] = field(default_factory=list)
    compute_instance_size: int = 0
    pc_scratch: Optional["cuda.DeviceAllocation"] = None
    pc_scratch_size: int = 0
    pc_host_staging: Optional[object] = None
    pc_host_staging_size: int = 0


@dataclass
class _KernelParam:
    kind: str
    binding: Optional[int]
    raw_name: str


@dataclass
class _ComputePlan:
    context_handle: int
    shader_source: bytes
    bindings: List[int]
    pc_size: int
    shader_name: bytes
    module: SourceModule
    function: object
    local_size: Tuple[int, int, int]
    params: List[_KernelParam]


@dataclass
class _DescriptorSet:
    plan_handle: int
    buffer_bindings: Dict[int, Tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    image_bindings: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)


@dataclass
class _ResolvedLaunch:
    plan: _ComputePlan
    blocks: Tuple[int, int, int]
    pc_offset: int
    pc_size: int
    args: Tuple[object, ...]
    pc_scratch: Optional["cuda.DeviceAllocation"] = None


# --- Helper utilities ---


def _new_handle(registry: Dict[int, object], obj: object) -> int:
    global _next_handle
    handle = _next_handle
    _next_handle += 1
    registry[handle] = obj
    return handle


def _to_bytes(value) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def _set_error(message: str) -> None:
    global _error_string
    _error_string = str(message)


def _clear_error() -> None:
    global _error_string
    _error_string = None


def _coerce_stream_handle(stream_obj) -> Optional[int]:
    if stream_obj is None:
        return None

    if isinstance(stream_obj, int):
        return int(stream_obj)

    for attr_name in ("cuda_stream", "ptr", "handle"):
        if hasattr(stream_obj, attr_name):
            try:
                return int(getattr(stream_obj, attr_name))
            except Exception:
                pass

    nested = getattr(stream_obj, "stream", None)
    if nested is not None and nested is not stream_obj:
        try:
            return _coerce_stream_handle(nested)
        except Exception:
            pass

    try:
        return int(stream_obj)
    except Exception as exc:
        raise TypeError(
            "Unable to extract a CUDA stream handle from the provided object. "
            "Pass an int handle or an object with .cuda_stream/.ptr/.handle."
        ) from exc


def _stream_override_stack() -> List[Optional[int]]:
    stack = getattr(_stream_override, "stack", None)
    if stack is None:
        stack = []
        _stream_override.stack = stack
    return stack


def _get_stream_override_handle() -> Optional[int]:
    stack = getattr(_stream_override, "stack", None)
    if not stack:
        return None
    return stack[-1]


def _wrap_external_stream(handle: int):
    handle = int(handle)

    if handle in _external_stream_cache:
        return _external_stream_cache[handle]

    if handle == 0:
        return None

    ctor_attempts = [
        lambda: cuda.Stream(handle=handle),
        lambda: cuda.Stream(ptr=handle),
        lambda: cuda.Stream(int(handle)),
    ]

    external_cls = getattr(cuda, "ExternalStream", None)
    if external_cls is not None:
        ctor_attempts.insert(0, lambda: external_cls(handle))

    last_error = None
    for ctor in ctor_attempts:
        try:
            stream_obj = ctor()
            _external_stream_cache[handle] = stream_obj
            return stream_obj
        except Exception as exc:  # pragma: no cover - depends on pycuda version
            last_error = exc

    raise RuntimeError(
        f"Failed to wrap external CUDA stream handle {handle} with PyCUDA. "
        "This PyCUDA version may not support external stream wrappers."
    ) from last_error


def _stream_for_queue(ctx: _Context, queue_index: int):
    override_handle = _get_stream_override_handle()
    if override_handle is None:
        return ctx.streams[queue_index]
    return _wrap_external_stream(int(override_handle))


def _buffer_device_ptr(buffer_obj: _Buffer) -> int:
    return int(buffer_obj.device_ptr)


def _queue_indices(ctx: _Context, queue_index: int, *, all_on_negative: bool = False) -> List[int]:
    if ctx.queue_count <= 0:
        return []

    if queue_index is None:
        return [0]

    queue_index = int(queue_index)

    if all_on_negative and queue_index < 0:
        return list(range(ctx.queue_count))

    if queue_index == -1:
        return [0]

    if 0 <= queue_index < ctx.queue_count:
        return [queue_index]

    return []


def _context_from_handle(context_handle: int) -> Optional[_Context]:
    ctx = _contexts.get(int(context_handle))
    if ctx is None:
        _set_error(f"Invalid context handle {context_handle}")
    return ctx


@contextmanager
def _activate_context(ctx: _Context):
    ctx.pycuda_context.push()
    try:
        yield
    finally:
        cuda.Context.pop()


def _record_signal(signal: _Signal, stream: "cuda.Stream") -> None:
    signal.submitted = True
    signal.done = False
    if signal.event is None:
        signal.event = cuda.Event()
    signal.event.record(stream)


def _query_signal(signal: _Signal) -> bool:
    if signal.event is None:
        return bool(signal.done)

    try:
        done = signal.event.query()
    except Exception:
        return False

    signal.done = bool(done)
    return signal.done


def _allocate_staging_storage(size: int):
    try:
        # Pagelocked host memory improves async HtoD/DtoH throughput and overlap.
        return cuda.pagelocked_empty(int(size), np.uint8)
    except Exception:
        return bytearray(int(size))


def _ensure_command_payload_staging(command_list: _CommandList, required_size: int):
    if required_size <= 0:
        required_size = 1

    if (
        command_list.pc_host_staging is not None
        and command_list.pc_host_staging_size >= required_size
    ):
        return command_list.pc_host_staging

    command_list.pc_host_staging = _allocate_staging_storage(required_size)
    command_list.pc_host_staging_size = required_size
    return command_list.pc_host_staging


def _write_command_payload_staging(
    command_list: _CommandList,
    payload: bytes,
    instance_count: int,
) -> int:
    instance_count = int(instance_count)
    if instance_count <= 0:
        return 0

    instance_size = int(command_list.compute_instance_size)
    expected_size = instance_size * instance_count if instance_size > 0 else len(payload)

    if instance_size > 0 and len(payload) < expected_size:
        raise RuntimeError(
            f"Instance payload is too small ({len(payload)} bytes) for "
            f"{instance_count} instances of size {instance_size}"
        )

    if expected_size <= 0:
        _ensure_command_payload_staging(command_list, 1)
        return 0

    staging = _ensure_command_payload_staging(command_list, expected_size)
    payload_view = memoryview(payload)[:expected_size]
    memoryview(staging)[:expected_size] = payload_view
    return expected_size


def _parse_local_size(source: str) -> Tuple[int, int, int]:
    x_match = _LOCAL_X_RE.search(source)
    y_match = _LOCAL_Y_RE.search(source)
    z_match = _LOCAL_Z_RE.search(source)

    x = int(x_match.group(1)) if x_match else 1
    y = int(y_match.group(1)) if y_match else 1
    z = int(z_match.group(1)) if z_match else 1

    return (x, y, z)


def _parse_kernel_params(source: str) -> List[_KernelParam]:
    signature_match = _KERNEL_SIGNATURE_RE.search(source)
    if signature_match is None:
        raise RuntimeError("Could not find vkdispatch_main kernel signature in CUDA source")

    signature_blob = signature_match.group(1).strip()
    if len(signature_blob) == 0:
        return []

    params: List[_KernelParam] = []

    for raw_decl in [part.strip() for part in signature_blob.split(",") if len(part.strip()) > 0]:
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", raw_decl)
        if name_match is None:
            raise RuntimeError(f"Unable to parse kernel parameter declaration '{raw_decl}'")

        param_name = name_match.group(1)

        if param_name == "vkdispatch_uniform_ptr":
            params.append(_KernelParam("uniform", 0, param_name))
            continue

        if param_name == "vkdispatch_pc_ptr":
            params.append(_KernelParam("push_constant", None, param_name))
            continue

        binding_match = _BINDING_PARAM_RE.match(param_name)
        if binding_match is not None:
            params.append(_KernelParam("storage", int(binding_match.group(1)), param_name))
            continue

        sampler_match = _SAMPLER_PARAM_RE.match(param_name)
        if sampler_match is not None:
            params.append(_KernelParam("sampler", int(sampler_match.group(1)), param_name))
            continue

        params.append(_KernelParam("unknown", None, param_name))

    return params


def _resolve_buffer_pointer(descriptor_set: _DescriptorSet, binding: int) -> int:
    binding_info = descriptor_set.buffer_bindings.get(binding)
    if binding_info is None:
        raise RuntimeError(f"Missing descriptor buffer binding {binding}")

    buffer_handle, offset, _range, _uniform, _read_access, _write_access = binding_info

    buffer_obj = _buffers.get(int(buffer_handle))
    if buffer_obj is None:
        raise RuntimeError(f"Invalid buffer handle {buffer_handle} for binding {binding}")

    return _buffer_device_ptr(buffer_obj) + int(offset)


def _ensure_pc_scratch(command_list: _CommandList, required_size: int) -> "cuda.DeviceAllocation":
    if required_size <= 0:
        required_size = 1

    if command_list.pc_scratch is not None and command_list.pc_scratch_size >= required_size:
        return command_list.pc_scratch

    command_list.pc_scratch = cuda.mem_alloc(required_size)
    command_list.pc_scratch_size = required_size
    return command_list.pc_scratch


def _build_kernel_args(
    plan: _ComputePlan,
    descriptor_set: Optional[_DescriptorSet],
    command_list: _CommandList,
    pc_data: bytes,
    stream: "cuda.Stream",
) -> List[object]:
    args: List[object] = []

    for param in plan.params:
        if param.kind == "uniform":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, 0)))
            continue

        if param.kind == "storage":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            if param.binding is None:
                raise RuntimeError("Storage parameter has no binding index")

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, param.binding)))
            continue

        if param.kind == "push_constant":
            pc_scratch = _ensure_pc_scratch(command_list, len(pc_data))

            if len(pc_data) > 0:
                cuda.memcpy_htod_async(pc_scratch, pc_data, stream)

            args.append(np.uintp(int(pc_scratch)))
            continue

        if param.kind == "sampler":
            raise RuntimeError("PyCUDA backend does not support sampled image bindings yet")

        raise RuntimeError(
            f"Unsupported kernel parameter '{param.raw_name}'. "
            "Expected vkdispatch_uniform_ptr / vkdispatch_binding_<N>_ptr / vkdispatch_pc_ptr."
        )

    return args


def _build_kernel_args_template(
    plan: _ComputePlan,
    descriptor_set: Optional[_DescriptorSet],
    command_list: _CommandList,
    pc_size: int,
) -> Tuple[Tuple[object, ...], Optional["cuda.DeviceAllocation"]]:
    args: List[object] = []
    pc_scratch: Optional["cuda.DeviceAllocation"] = None

    for param in plan.params:
        if param.kind == "uniform":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, 0)))
            continue

        if param.kind == "storage":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            if param.binding is None:
                raise RuntimeError("Storage parameter has no binding index")

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, param.binding)))
            continue

        if param.kind == "push_constant":
            if pc_scratch is None:
                pc_scratch = _ensure_pc_scratch(command_list, int(pc_size))
            args.append(np.uintp(int(pc_scratch)))
            continue

        if param.kind == "sampler":
            raise RuntimeError("PyCUDA backend does not support sampled image bindings yet")

        raise RuntimeError(
            f"Unsupported kernel parameter '{param.raw_name}'. "
            "Expected vkdispatch_uniform_ptr / vkdispatch_binding_<N>_ptr / vkdispatch_pc_ptr."
        )

    return tuple(args), pc_scratch


# --- API: context/init/logging ---


def init(debug, log_level):
    global _initialized, _debug_mode, _log_level

    _debug_mode = bool(debug)
    _log_level = int(log_level)
    _clear_error()

    if _initialized:
        return

    cuda.init()
    _initialized = True


def log(log_level, text, file_str, line_str):
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

    try:
        device_count = cuda.Device.count()
    except Exception as exc:
        _set_error(f"Failed to enumerate CUDA devices: {exc}")
        return []

    driver_version = 0
    try:
        driver_version = int(cuda.get_driver_version())
    except Exception:
        driver_version = 0

    devices = []

    for index in range(device_count):
        dev = cuda.Device(index)
        attrs = dev.get_attributes()
        cc_major, cc_minor = dev.compute_capability()
        total_memory = int(dev.total_memory())

        max_workgroup_size = (
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_X, 0)),
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Y, 0)),
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Z, 0)),
        )

        max_workgroup_count = (
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_X, 0)),
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_Y, 0)),
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_Z, 0)),
        )

        subgroup_size = int(attrs.get(cuda.device_attribute.WARP_SIZE, 0))
        max_shared_memory = int(
            attrs.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK, 0)
        )

        try:
            bus_id = str(dev.pci_bus_id())
        except Exception:
            bus_id = f"cuda-device-{index}"

        uuid_bytes = hashlib.md5(bus_id.encode("utf-8")).digest()

        devices.append(
            (
                0,  # Vulkan variant
                int(cc_major),  # major
                int(cc_minor),  # minor
                0,  # patch
                driver_version,
                0,  # vendor id unknown in this API layer
                index,  # device id
                2,  # discrete gpu
                str(dev.name()),
                1,  # shader_buffer_float32_atomics
                1,  # shader_buffer_float32_atomic_add
                1,  # float64 support
                1 if (cc_major > 5 or (cc_major == 5 and cc_minor >= 3)) else 0,  # float16 support
                1,  # int64
                1,  # int16
                1,  # storage_buffer_16_bit_access
                1,  # uniform_and_storage_buffer_16_bit_access
                1,  # storage_push_constant_16
                1,  # storage_input_output_16
                max_workgroup_size,
                int(attrs.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, 0)),
                max_workgroup_count,
                8,  # max descriptor sets (virtualized for parity)
                4096,  # max push constant size
                min(total_memory, (1 << 31) - 1),
                65536,
                16,
                subgroup_size,
                0x7FFFFFFF,  # supported stages (virtualized for parity)
                0x7FFFFFFF,  # supported operations (virtualized for parity)
                1,
                max_shared_memory,
                [(1, 0x002)],  # compute queue
                1,  # scalar block layout
                1,  # timeline semaphores equivalent
                uuid_bytes,
            )
        )

    return devices


def context_create(device_indicies, queue_families):
    if not _initialized:
        init(False, _log_level)

    try:
        device_ids = [int(x) for x in device_indicies]
    except Exception:
        _set_error("context_create expected a list of integer device indices")
        return 0

    if len(device_ids) != 1:
        _set_error("PyCUDA backend currently supports exactly one device")
        return 0

    if len(queue_families) != 1 or len(queue_families[0]) != 1:
        _set_error("PyCUDA backend currently supports exactly one queue")
        return 0

    device_index = device_ids[0]

    pycuda_context = None
    context_pushed = False

    try:
        if device_index < 0 or device_index >= cuda.Device.count():
            _set_error(f"Invalid CUDA device index {device_index}")
            return 0

        dev = cuda.Device(device_index)
        uses_primary_context = False

        if hasattr(dev, "retain_primary_context"):
            pycuda_context = dev.retain_primary_context()
            uses_primary_context = True
            pycuda_context.push()
        else:  # pragma: no cover - fallback for older PyCUDA
            pycuda_context = dev.make_context()
        context_pushed = True
        stream = cuda.Stream()

        ctx = _Context(
            device_index=device_index,
            pycuda_context=pycuda_context,
            streams=[stream],
            queue_count=1,
            queue_to_device=[0],
            uses_primary_context=uses_primary_context,
            stopped=False,
        )
        handle = _new_handle(_contexts, ctx)

        # Leave no context current after creation.
        cuda.Context.pop()
        context_pushed = False
        return handle
    except Exception as exc:
        if context_pushed:
            try:
                cuda.Context.pop()
            except Exception:
                pass

        if pycuda_context is not None:
            try:
                pycuda_context.detach()
            except Exception:
                pass

        _set_error(f"Failed to create PyCUDA context: {exc}")
        return 0


def context_destroy(context):
    ctx = _contexts.pop(int(context), None)
    if ctx is None:
        return

    try:
        with _activate_context(ctx):
            for stream in ctx.streams:
                stream.synchronize()
    except Exception:
        pass

    try:
        ctx.pycuda_context.detach()
    except Exception:
        pass


def context_stop_threads(context):
    ctx = _contexts.get(int(context))
    if ctx is not None:
        ctx.stopped = True


def get_error_string():
    if _error_string is None:
        return 0
    return _error_string


def cuda_stream_override_begin(stream_obj):
    try:
        stack = _stream_override_stack()
        stack.append(_coerce_stream_handle(stream_obj))
    except Exception as exc:
        _set_error(f"Failed to activate external CUDA stream override: {exc}")


def cuda_stream_override_end():
    stack = _stream_override_stack()
    if len(stack) > 0:
        stack.pop()


# --- API: signals ---


def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    signal_obj = _signals.get(int(signal_ptr))
    if signal_obj is None:
        return True

    if not bool(wait_for_timestamp):
        # PyCUDA records signals synchronously on submission; host-side "recorded" waits
        # should therefore complete immediately once an event exists.
        if signal_obj.event is None:
            return bool(signal_obj.done)
        return bool(signal_obj.submitted)

    if signal_obj.done:
        return True

    if signal_obj.event is None:
        return bool(signal_obj.done)

    ctx = _contexts.get(signal_obj.context_handle)
    if ctx is None:
        return _query_signal(signal_obj)

    try:
        with _activate_context(ctx):
            signal_obj.event.synchronize()
        signal_obj.done = True
        return True
    except Exception:
        return _query_signal(signal_obj)


def signal_insert(context, queue_index):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    selected = _queue_indices(ctx, int(queue_index))
    if len(selected) == 0:
        selected = [0]

    signal = _Signal(context_handle=int(context), queue_index=selected[0], submitted=False, done=False)
    handle = _new_handle(_signals, signal)

    try:
        with _activate_context(ctx):
            _record_signal(signal, _stream_for_queue(ctx, selected[0]))
    except Exception as exc:
        _set_error(f"Failed to insert signal: {exc}")
        return 0

    return handle


def signal_destroy(signal_ptr):
    _signals.pop(int(signal_ptr), None)


# --- API: buffers ---


def buffer_create(context, size, per_device):
    _ = per_device

    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    size = int(size)
    if size <= 0:
        _set_error("Buffer size must be greater than zero")
        return 0

    try:
        with _activate_context(ctx):
            allocation = cuda.mem_alloc(size)

        signal_handles = [
            _new_handle(_signals, _Signal(context_handle=int(context), queue_index=i, done=True))
            for i in range(ctx.queue_count)
        ]

        obj = _Buffer(
            context_handle=int(context),
            size=size,
            device_ptr=int(allocation),
            device_allocation=allocation,
            owns_allocation=True,
            staging_data=[_allocate_staging_storage(size) for _ in range(ctx.queue_count)],
            signal_handles=signal_handles,
        )
        return _new_handle(_buffers, obj)
    except Exception as exc:
        _set_error(f"Failed to create CUDA buffer: {exc}")
        return 0


def buffer_create_external(context, size, device_ptr):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    size = int(size)
    device_ptr = int(device_ptr)

    if size <= 0:
        _set_error("External buffer size must be greater than zero")
        return 0

    if device_ptr == 0:
        _set_error("External buffer device pointer must be non-zero")
        return 0

    try:
        signal_handles = [
            _new_handle(_signals, _Signal(context_handle=int(context), queue_index=i, done=True))
            for i in range(ctx.queue_count)
        ]

        obj = _Buffer(
            context_handle=int(context),
            size=size,
            device_ptr=device_ptr,
            device_allocation=None,
            owns_allocation=False,
            staging_data=[_allocate_staging_storage(size) for _ in range(ctx.queue_count)],
            signal_handles=signal_handles,
        )
        return _new_handle(_buffers, obj)
    except Exception as exc:
        _set_error(f"Failed to create external CUDA buffer alias: {exc}")
        return 0


def buffer_destroy(buffer):
    obj = _buffers.pop(int(buffer), None)
    if obj is None:
        return

    for signal_handle in obj.signal_handles:
        _signals.pop(signal_handle, None)

    ctx = _contexts.get(obj.context_handle)
    if ctx is None or not obj.owns_allocation or obj.device_allocation is None:
        return

    try:
        with _activate_context(ctx):
            obj.device_allocation.free()
    except Exception:
        pass


def buffer_get_queue_signal(buffer, queue_index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return _new_handle(_signals, _Signal(context_handle=0, queue_index=0, done=True))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.signal_handles):
        queue_index = 0

    return obj.signal_handles[queue_index]


def buffer_wait_staging_idle(buffer, queue_index):
    signal_handle = buffer_get_queue_signal(buffer, queue_index)
    signal_obj = _signals.get(int(signal_handle))
    if signal_obj is None:
        return True
    return _query_signal(signal_obj)


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

    payload_view = memoryview(payload)[:size]
    staging_view = memoryview(obj.staging_data[queue_index])
    staging_view[:size] = payload_view


def buffer_read_staging(buffer, queue_index, size):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return bytes(int(size))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.staging_data):
        return bytes(int(size))

    size = max(0, int(size))
    staging = obj.staging_data[queue_index]

    if size <= len(staging):
        return bytes(staging[:size])

    return bytes(staging) + bytes(size - len(staging))


def buffer_write(buffer, offset, size, index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        _set_error(f"Missing context for buffer handle {buffer}")
        return

    offset = int(offset)
    size = int(size)
    if size <= 0 or offset < 0:
        return

    try:
        with _activate_context(ctx):
            for queue_index in _queue_indices(ctx, int(index), all_on_negative=True):
                stream = _stream_for_queue(ctx, queue_index)
                end = min(offset + size, obj.size)
                copy_size = end - offset
                if copy_size <= 0:
                    continue

                src_view = memoryview(obj.staging_data[queue_index])[:copy_size]
                cuda.memcpy_htod_async(_buffer_device_ptr(obj) + offset, src_view, stream)

                signal = _signals.get(obj.signal_handles[queue_index])
                if signal is not None:
                    _record_signal(signal, stream)
    except Exception as exc:
        _set_error(f"Failed to write CUDA buffer: {exc}")


def buffer_read(buffer, offset, size, index):
    obj = _buffers.get(int(buffer))
    if obj is None:
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        _set_error(f"Missing context for buffer handle {buffer}")
        return

    queue_index = int(index)
    if queue_index < 0 or queue_index >= ctx.queue_count:
        _set_error(f"Invalid queue index {queue_index} for buffer read")
        return

    offset = int(offset)
    size = int(size)
    if size <= 0 or offset < 0:
        return

    try:
        with _activate_context(ctx):
            stream = _stream_for_queue(ctx, queue_index)
            end = min(offset + size, obj.size)
            copy_size = end - offset
            if copy_size <= 0:
                return

            dst_view = memoryview(obj.staging_data[queue_index])[:copy_size]
            cuda.memcpy_dtoh_async(dst_view, _buffer_device_ptr(obj) + offset, stream)

            signal = _signals.get(obj.signal_handles[queue_index])
            if signal is not None:
                _record_signal(signal, stream)
    except Exception as exc:
        _set_error(f"Failed to read CUDA buffer: {exc}")


# --- API: command lists ---


def command_list_create(context):
    if int(context) not in _contexts:
        _set_error("Invalid context handle for command_list_create")
        return 0

    return _new_handle(_command_lists, _CommandList(context_handle=int(context)))


def command_list_destroy(command_list):
    obj = _command_lists.pop(int(command_list), None)
    if obj is None:
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        return

    if obj.pc_scratch is None:
        return

    try:
        with _activate_context(ctx):
            obj.pc_scratch.free()
    except Exception:
        pass


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


def command_list_prepare_cuda_capture(command_list, payload_size):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        _set_error("Invalid command list handle for command_list_prepare_cuda_capture")
        return

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        _set_error(f"Missing context for command list {command_list}")
        return

    payload_size = max(0, int(payload_size))

    try:
        _ensure_command_payload_staging(obj, max(1, payload_size))

        max_pc_size = 0
        for command in obj.commands:
            max_pc_size = max(max_pc_size, int(command.pc_size))

        if max_pc_size > 0:
            with _activate_context(ctx):
                _ensure_pc_scratch(obj, max_pc_size)
    except Exception as exc:
        _set_error(f"Failed to prepare CUDA capture resources: {exc}")


def command_list_write_payload_staging(command_list, data, instance_count):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        _set_error("Invalid command list handle for command_list_write_payload_staging")
        return

    try:
        payload = _to_bytes(data) if data is not None else b""
        _write_command_payload_staging(obj, payload, int(instance_count))
    except Exception as exc:
        _set_error(f"Failed to write CUDA command payload staging: {exc}")


def command_list_submit(command_list, data, instance_count, index):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return True

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        _set_error(f"Missing context for command list {command_list}")
        return True

    payload = _to_bytes(data) if data is not None else b""
    instance_count = int(instance_count)
    if instance_count <= 0:
        return True

    instance_size = int(obj.compute_instance_size)

    if instance_size > 0 and len(payload) < instance_size * instance_count:
        _set_error(
            f"Instance payload is too small ({len(payload)} bytes) for "
            f"{instance_count} instances of size {instance_size}"
        )
        return True

    queue_targets = _queue_indices(ctx, int(index), all_on_negative=True)
    if len(queue_targets) == 0:
        queue_targets = [0]

    try:
        payload_nbytes = instance_size * instance_count if instance_size > 0 else len(payload)
        if len(payload) > 0:
            _write_command_payload_staging(obj, payload, instance_count)
        elif payload_nbytes > 0 and (
            obj.pc_host_staging is None or obj.pc_host_staging_size < payload_nbytes
        ):
            raise RuntimeError(
                "Command payload staging is not prepared. "
                "Provide payload data or call command_list_prepare_cuda_capture(...) first."
            )

        with _activate_context(ctx):
            payload_view = (
                memoryview(obj.pc_host_staging)[:payload_nbytes]
                if payload_nbytes > 0 and obj.pc_host_staging is not None
                else None
            )

            for queue_index in queue_targets:
                stream = _stream_for_queue(ctx, queue_index)
                resolved_launches: List[_ResolvedLaunch] = []
                pc_offset = 0

                for command in obj.commands:
                    plan = _compute_plans.get(command.plan_handle)
                    if plan is None:
                        raise RuntimeError(f"Invalid compute plan handle {command.plan_handle}")

                    descriptor_set = None
                    if command.descriptor_set_handle != 0:
                        descriptor_set = _descriptor_sets.get(command.descriptor_set_handle)
                        if descriptor_set is None:
                            raise RuntimeError(
                                f"Invalid descriptor set handle {command.descriptor_set_handle}"
                            )

                    pc_size = int(command.pc_size)
                    args, pc_scratch = _build_kernel_args_template(plan, descriptor_set, obj, pc_size)
                    resolved_launches.append(
                        _ResolvedLaunch(
                            plan=plan,
                            blocks=command.blocks,
                            pc_offset=pc_offset,
                            pc_size=pc_size,
                            args=args,
                            pc_scratch=pc_scratch,
                        )
                    )
                    pc_offset += pc_size

                for instance in range(instance_count):
                    instance_base = instance * instance_size

                    for launch in resolved_launches:
                        if launch.pc_scratch is not None and launch.pc_size > 0:
                            start = instance_base + launch.pc_offset
                            end = start + launch.pc_size
                            cuda.memcpy_htod_async(
                                launch.pc_scratch,
                                payload_view[start:end],
                                stream,
                            )

                        launch.plan.function(
                            *launch.args,
                            block=launch.plan.local_size,
                            grid=launch.blocks,
                            stream=stream,
                        )
    except Exception as exc:
        _set_error(f"Failed to submit CUDA command list: {exc}")

    return True


# --- API: descriptor sets ---


def descriptor_set_create(plan):
    if int(plan) not in _compute_plans:
        _set_error("Invalid compute plan handle for descriptor_set_create")
        return 0

    return _new_handle(_descriptor_sets, _DescriptorSet(plan_handle=int(plan)))


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
        _set_error("Invalid descriptor set handle for descriptor_set_write_buffer")
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
        _set_error("Invalid descriptor set handle for descriptor_set_write_image")
        return

    ds.image_bindings[int(binding)] = (
        int(object),
        int(sampler_obj),
        int(read_access),
        int(write_access),
    )


# --- API: compute stage ---


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    source_bytes = _to_bytes(shader_source)
    shader_name_bytes = _to_bytes(shader_name)
    source_text = source_bytes.decode("utf-8", errors="replace")

    try:
        with _activate_context(ctx):
            module = SourceModule(
                source_text,
                no_extern_c=True,
                options=["-w"]
            )
            function = module.get_function("vkdispatch_main")
    except Exception as exc:
        _set_error(f"Failed to compile CUDA kernel '{shader_name_bytes.decode(errors='ignore')}': {exc}")
        return 0

    try:
        params = _parse_kernel_params(source_text)
        local_size = _parse_local_size(source_text)
    except Exception as exc:
        _set_error(f"Failed to parse CUDA kernel metadata: {exc}")
        return 0

    plan = _ComputePlan(
        context_handle=int(context),
        shader_source=source_bytes,
        bindings=[int(x) for x in bindings],
        pc_size=int(pc_size),
        shader_name=shader_name_bytes,
        module=module,
        function=function,
        local_size=local_size,
        params=params,
    )

    return _new_handle(_compute_plans, plan)


def stage_compute_plan_destroy(plan):
    if plan is None:
        return
    _compute_plans.pop(int(plan), None)


def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    cl = _command_lists.get(int(command_list))
    cp = _compute_plans.get(int(plan))
    if cl is None or cp is None:
        _set_error("Invalid command list or compute plan handle for stage_compute_record")
        return

    cl.commands.append(
        _CommandRecord(
            plan_handle=int(plan),
            descriptor_set_handle=int(descriptor_set),
            blocks=(int(blocks_x), int(blocks_y), int(blocks_z)),
            pc_size=int(cp.pc_size),
        )
    )
    cl.compute_instance_size += int(cp.pc_size)


# --- API: images/samplers (not yet implemented on PyCUDA backend) ---


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    _ = context
    _ = extent
    _ = layers
    _ = format
    _ = type
    _ = view_type
    _ = generate_mips
    _set_error("PyCUDA backend does not support image objects yet")
    return 0


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
    _ = context
    _ = mag_filter
    _ = min_filter
    _ = mip_mode
    _ = address_mode
    _ = mip_lod_bias
    _ = min_lod
    _ = max_lod
    _ = border_color
    _set_error("PyCUDA backend does not support image samplers yet")
    return 0


def image_destroy_sampler(sampler):
    _samplers.pop(int(sampler), None)


def image_write(image, data, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = data
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("PyCUDA backend does not support image writes yet")


def image_format_block_size(format):
    return int(_IMAGE_BLOCK_SIZES.get(int(format), 4))


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("PyCUDA backend does not support image reads yet")
    return bytes(max(0, int(out_size)))


# --- API: FFT stage (not yet implemented on PyCUDA backend) ---


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
    _ = context
    _ = dims
    _ = axes
    _ = buffer_size
    _ = do_r2c
    _ = normalize
    _ = pad_left
    _ = pad_right
    _ = frequency_zeropadding
    _ = kernel_num
    _ = kernel_convolution
    _ = conjugate_convolution
    _ = convolution_features
    _ = input_buffer_size
    _ = num_batches
    _ = single_kernel_multiple_batches
    _ = keep_shader_code
    _set_error("PyCUDA backend does not support FFT plans yet")
    return 0


def stage_fft_plan_destroy(plan):
    _fft_plans.pop(int(plan), None)


def stage_fft_record(command_list, plan, buffer, inverse, kernel, input_buffer):
    _ = command_list
    _ = plan
    _ = buffer
    _ = inverse
    _ = kernel
    _ = input_buffer
    _set_error("PyCUDA backend does not support FFT stages yet")


__all__ = [
    "LOG_LEVEL_VERBOSE",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_WARNING",
    "LOG_LEVEL_ERROR",
    "DESCRIPTOR_TYPE_STORAGE_BUFFER",
    "DESCRIPTOR_TYPE_STORAGE_IMAGE",
    "DESCRIPTOR_TYPE_UNIFORM_BUFFER",
    "DESCRIPTOR_TYPE_UNIFORM_IMAGE",
    "DESCRIPTOR_TYPE_SAMPLER",
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
]
