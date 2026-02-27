"""pyopencl-backed runtime shim mirroring the vkdispatch_native API surface.

This module intentionally matches the function names exposed by the Cython
extension so existing Python runtime objects can call into either backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import threading
from typing import Dict, List, Optional, Tuple

import os
import sys

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - import failure path
    raise ImportError(
        "The OpenCL Python backend requires both 'pyopencl' and 'numpy' to be installed."
    ) from exc

try:
    import pyopencl as cl
except Exception as exc:  # pragma: no cover - import failure path
    raise ImportError(
        "The OpenCL runtime backend requires the 'pyopencl' package "
        "(`pip install pyopencl`)."
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
_REQD_LOCAL_RE = re.compile(r"reqd_work_group_size\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_KERNEL_SIGNATURE_RE = re.compile(r"vkdispatch_main\s*\(([^)]*)\)", re.S)
_BINDING_PARAM_RE = re.compile(r"vkdispatch_binding_(\d+)_ptr$")
_SAMPLER_PARAM_RE = re.compile(r"vkdispatch_sampler_(\d+)$")
_PUSH_CONSTANT_STRUCT_RE = re.compile(
    r"typedef\s+struct\s+PushConstant\s*\{(?P<body>.*?)\}\s*PushConstant\s*;",
    re.S,
)
_PUSH_CONSTANT_FIELD_RE = re.compile(
    r"(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*\[\s*(?P<count>\d+)\s*\])?$"
)
_VECTOR_TYPE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*?)([2-4])$")
_OPENCL_VERSION_RE = re.compile(r"OpenCL\s+(\d+)\.(\d+)")
_DIGIT_RE = re.compile(r"(\d+)")


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

_marker_helpers = threading.local()


# --- Internal objects ---


@dataclass(frozen=True)
class _DeviceEntry:
    logical_index: int
    platform_index: int
    device_index: int
    platform: object
    device: object


@dataclass
class _Signal:
    context_handle: int
    queue_index: int
    event: Optional[object] = None
    submitted: bool = True
    done: bool = True


@dataclass
class _Context:
    device_index: int
    cl_context: object
    queues: List[object]
    queue_count: int
    queue_to_device: List[int]
    sub_buffer_alignment: int
    stopped: bool = False


@dataclass
class _Buffer:
    context_handle: int
    size: int
    cl_buffer: object
    staging_data: List[bytearray]
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


@dataclass
class _KernelParam:
    kind: str
    binding: Optional[int]
    raw_name: str


@dataclass(frozen=True)
class _PushConstantTypeLayout:
    host_elem_size: int
    opencl_elem_size: int
    opencl_align: int


@dataclass(frozen=True)
class _PushConstantFieldDecl:
    type_name: str
    field_name: str
    count: int


@dataclass(frozen=True)
class _PushConstantFieldLayout:
    type_name: str
    field_name: str
    count: int
    host_offset: int
    opencl_offset: int
    host_elem_size: int
    opencl_elem_size: int


@dataclass(frozen=True)
class _PushConstantLayout:
    fields: Tuple[_PushConstantFieldLayout, ...]
    host_size: int
    opencl_size: int
    opencl_alignment: int
    needs_repack: bool


@dataclass
class _ComputePlan:
    context_handle: int
    shader_source: bytes
    bindings: List[int]
    shader_name: bytes
    program: object
    kernel: object
    local_size: Tuple[int, int, int]
    params: List[_KernelParam]
    pc_size: int
    pc_layout: Optional[_PushConstantLayout] = None


@dataclass
class _DescriptorSet:
    plan_handle: int
    buffer_bindings: Dict[int, Tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    image_bindings: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)


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

def _enumerate_opencl_devices() -> List[_DeviceEntry]:
    entries: List[_DeviceEntry] = []
    
    if (
        sys.platform.startswith("linux")
        and "OCL_ICD_VENDORS" not in os.environ
        and "OPENCL_VENDOR_PATH" not in os.environ
        and os.path.isdir("/etc/OpenCL/vendors")
    ):
        os.environ["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"

    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to get OpenCL Platform: {exc}"
        ) from exc

    logical_index = 0
    for platform_index, platform in enumerate(platforms):
        try:
            devices = platform.get_devices()
        except Exception:
            continue

        for device_index, device in enumerate(devices):
            entries.append(
                _DeviceEntry(
                    logical_index=logical_index,
                    platform_index=platform_index,
                    device_index=device_index,
                    platform=platform,
                    device=device,
                )
            )
            logical_index += 1

    return entries


def _coerce_int(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _opencl_version_components(version_text: str) -> Tuple[int, int]:
    if not isinstance(version_text, str):
        return (0, 0)

    match = _OPENCL_VERSION_RE.search(version_text)
    if match is None:
        return (0, 0)

    return (_coerce_int(match.group(1), 0), _coerce_int(match.group(2), 0))


def _driver_version_number(driver_text: str) -> int:
    if not isinstance(driver_text, str):
        return 0

    pieces = _DIGIT_RE.findall(driver_text)
    if len(pieces) == 0:
        return 0

    folded = 0
    weight = 1_000_000
    for token in pieces[:3]:
        folded += _coerce_int(token, 0) * weight
        weight = max(1, weight // 1000)
    return folded


def _device_type_to_vkdispatch(device_type: int) -> int:
    if device_type & getattr(cl.device_type, "GPU", 0):
        return 2
    if device_type & getattr(cl.device_type, "ACCELERATOR", 0):
        return 3
    if device_type & getattr(cl.device_type, "CPU", 0):
        return 4
    return 0


def _device_uuid(entry: _DeviceEntry, device_name: str, driver_version: str) -> bytes:
    platform_vendor = ""
    platform_name = ""
    try:
        platform_vendor = str(entry.platform.vendor)
    except Exception:
        platform_vendor = ""
    try:
        platform_name = str(entry.platform.name)
    except Exception:
        platform_name = ""

    seed = (
        f"opencl:{entry.platform_index}:{entry.device_index}:"
        f"{platform_vendor}:"
        f"{platform_name}:"
        f"{device_name}:{driver_version}"
    )
    return hashlib.md5(seed.encode("utf-8")).digest()


def _device_attr(device, attr_name: str, default):
    try:
        return getattr(device, attr_name)
    except Exception:
        return default


def _context_from_handle(context_handle: int) -> Optional[_Context]:
    ctx = _contexts.get(int(context_handle))
    if ctx is None:
        _set_error(f"Invalid context handle {context_handle}")
    return ctx


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


def _record_signal(signal: _Signal, event_obj: Optional[object]) -> None:
    signal.submitted = True
    signal.done = event_obj is None
    signal.event = event_obj


def _query_signal(signal: _Signal) -> bool:
    if signal.event is None:
        return bool(signal.done)

    try:
        complete = int(getattr(getattr(cl, "command_execution_status", object()), "COMPLETE", 0))
        status = _coerce_int(signal.event.command_execution_status, 0)
        done = status == complete
    except Exception:
        done = False

    signal.done = bool(done)
    return signal.done


def _wait_signal(signal: _Signal) -> bool:
    if signal.event is None:
        return bool(signal.done)

    try:
        signal.event.wait()
        signal.done = True
        return True
    except Exception:
        return _query_signal(signal)


def _parse_local_size(source: str) -> Tuple[int, int, int]:
    x_match = _LOCAL_X_RE.search(source)
    y_match = _LOCAL_Y_RE.search(source)
    z_match = _LOCAL_Z_RE.search(source)

    if x_match is not None and y_match is not None and z_match is not None:
        return (
            _coerce_int(x_match.group(1), 1),
            _coerce_int(y_match.group(1), 1),
            _coerce_int(z_match.group(1), 1),
        )

    reqd_match = _REQD_LOCAL_RE.search(source)
    if reqd_match is not None:
        return (
            _coerce_int(reqd_match.group(1), 1),
            _coerce_int(reqd_match.group(2), 1),
            _coerce_int(reqd_match.group(3), 1),
        )

    return (1, 1, 1)


_PUSH_CONSTANT_SCALAR_LAYOUTS: Dict[str, Tuple[int, int]] = {
    "char": (1, 1),
    "uchar": (1, 1),
    "short": (2, 2),
    "ushort": (2, 2),
    "int": (4, 4),
    "uint": (4, 4),
    "long": (8, 8),
    "ulong": (8, 8),
    "half": (2, 2),
    "float": (4, 4),
    "double": (8, 8),
}

_PUSH_CONSTANT_MATRIX_LAYOUTS: Dict[str, _PushConstantTypeLayout] = {
    "vkdispatch_mat2": _PushConstantTypeLayout(host_elem_size=16, opencl_elem_size=16, opencl_align=8),
    "vkdispatch_mat3": _PushConstantTypeLayout(host_elem_size=36, opencl_elem_size=36, opencl_align=1),
    "vkdispatch_mat4": _PushConstantTypeLayout(host_elem_size=64, opencl_elem_size=64, opencl_align=16),
    "vkdispatch_packed_float3": _PushConstantTypeLayout(host_elem_size=12, opencl_elem_size=12, opencl_align=1),
}


def _extract_push_constant_struct_body(source: str) -> Optional[str]:
    struct_match = _PUSH_CONSTANT_STRUCT_RE.search(source)
    if struct_match is None:
        return None
    return struct_match.group("body")


def _parse_push_constant_struct_fields(body: str) -> List[_PushConstantFieldDecl]:
    fields: List[_PushConstantFieldDecl] = []

    for raw_decl in body.split(";"):
        decl = " ".join(raw_decl.strip().split())
        if len(decl) == 0:
            continue

        field_match = _PUSH_CONSTANT_FIELD_RE.fullmatch(decl)
        if field_match is None:
            raise RuntimeError(f"Unable to parse PushConstant field declaration '{decl}'")

        type_name = field_match.group("type")
        field_name = field_match.group("name")
        count_token = field_match.group("count")
        count = 1 if count_token is None else _coerce_int(count_token, 0)

        if count <= 0:
            raise RuntimeError(f"Invalid PushConstant array size for field '{field_name}'")

        fields.append(_PushConstantFieldDecl(type_name=type_name, field_name=field_name, count=count))

    return fields


def _push_constant_type_layout(type_name: str) -> _PushConstantTypeLayout:
    matrix_layout = _PUSH_CONSTANT_MATRIX_LAYOUTS.get(type_name)
    if matrix_layout is not None:
        return matrix_layout

    scalar_layout = _PUSH_CONSTANT_SCALAR_LAYOUTS.get(type_name)
    if scalar_layout is not None:
        size, align = scalar_layout
        return _PushConstantTypeLayout(host_elem_size=size, opencl_elem_size=size, opencl_align=align)

    vector_match = _VECTOR_TYPE_RE.fullmatch(type_name)
    if vector_match is not None:
        scalar_name = vector_match.group(1)
        lane_count = _coerce_int(vector_match.group(2), 0)
        scalar_info = _PUSH_CONSTANT_SCALAR_LAYOUTS.get(scalar_name)
        if scalar_info is None:
            raise RuntimeError(f"Unsupported PushConstant vector scalar type '{scalar_name}'")

        scalar_size, _scalar_align = scalar_info
        host_elem_size = scalar_size * lane_count

        if lane_count == 3:
            opencl_elem_size = scalar_size * 4
            opencl_align = scalar_size * 4
        else:
            opencl_elem_size = host_elem_size
            opencl_align = opencl_elem_size

        return _PushConstantTypeLayout(
            host_elem_size=host_elem_size,
            opencl_elem_size=opencl_elem_size,
            opencl_align=opencl_align,
        )

    raise RuntimeError(f"Unsupported PushConstant field type '{type_name}'")


def _compute_push_constant_layout(field_decls: List[_PushConstantFieldDecl]) -> _PushConstantLayout:
    host_offset = 0
    opencl_offset = 0
    max_opencl_align = 1
    needs_repack = False
    field_layouts: List[_PushConstantFieldLayout] = []

    for field_decl in field_decls:
        type_layout = _push_constant_type_layout(field_decl.type_name)

        opencl_offset = _align_up(opencl_offset, type_layout.opencl_align)

        if type_layout.opencl_align > max_opencl_align:
            max_opencl_align = type_layout.opencl_align

        if host_offset != opencl_offset:
            needs_repack = True
        if type_layout.host_elem_size != type_layout.opencl_elem_size:
            needs_repack = True

        field_layouts.append(
            _PushConstantFieldLayout(
                type_name=field_decl.type_name,
                field_name=field_decl.field_name,
                count=field_decl.count,
                host_offset=host_offset,
                opencl_offset=opencl_offset,
                host_elem_size=type_layout.host_elem_size,
                opencl_elem_size=type_layout.opencl_elem_size,
            )
        )

        host_offset += type_layout.host_elem_size * field_decl.count
        opencl_offset += type_layout.opencl_elem_size * field_decl.count

    opencl_size = _align_up(opencl_offset, max_opencl_align)
    if opencl_size != host_offset:
        needs_repack = True

    return _PushConstantLayout(
        fields=tuple(field_layouts),
        host_size=host_offset,
        opencl_size=opencl_size,
        opencl_alignment=max_opencl_align,
        needs_repack=needs_repack,
    )


def _build_push_constant_layout(source: str, expected_host_size: int) -> Optional[_PushConstantLayout]:
    expected_host_size = int(expected_host_size)
    if expected_host_size <= 0:
        return None

    body = _extract_push_constant_struct_body(source)
    if body is None:
        raise RuntimeError("Could not find PushConstant struct declaration in OpenCL source")

    field_decls = _parse_push_constant_struct_fields(body)
    if len(field_decls) == 0:
        raise RuntimeError("PushConstant struct declaration is empty")

    layout = _compute_push_constant_layout(field_decls)
    if layout.host_size != expected_host_size:
        raise RuntimeError(
            f"PushConstant host layout mismatch. Expected {expected_host_size} bytes "
            f"but parsed {layout.host_size} bytes from OpenCL source."
        )

    return layout


def _repack_push_constant_payload(
    push_constant_payload: bytes,
    layout: Optional[_PushConstantLayout],
) -> bytes:
    payload = _to_bytes(push_constant_payload)

    if layout is None or not layout.needs_repack:
        return payload

    if len(payload) != int(layout.host_size):
        raise RuntimeError(
            f"PushConstant payload length mismatch for repack. "
            f"Expected {layout.host_size} bytes but got {len(payload)} bytes."
        )

    out = bytearray(int(layout.opencl_size))

    for field in layout.fields:
        if field.host_elem_size > field.opencl_elem_size:
            raise RuntimeError(
                f"PushConstant field '{field.field_name}' host element size ({field.host_elem_size}) "
                f"exceeds OpenCL ABI element size ({field.opencl_elem_size})."
            )

        for element_index in range(int(field.count)):
            host_start = field.host_offset + (element_index * field.host_elem_size)
            host_end = host_start + field.host_elem_size
            opencl_start = field.opencl_offset + (element_index * field.opencl_elem_size)
            opencl_end = opencl_start + field.host_elem_size
            out[opencl_start:opencl_end] = payload[host_start:host_end]

    return bytes(out)


def _parse_kernel_params(source: str) -> List[_KernelParam]:
    signature_match = _KERNEL_SIGNATURE_RE.search(source)
    if signature_match is None:
        raise RuntimeError("Could not find vkdispatch_main kernel signature in OpenCL source")

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

        if param_name == "vkdispatch_pc_value":
            params.append(_KernelParam("push_constant_value", None, param_name))
            continue

        binding_match = _BINDING_PARAM_RE.match(param_name)
        if binding_match is not None:
            params.append(_KernelParam("storage", _coerce_int(binding_match.group(1), 0), param_name))
            continue

        sampler_match = _SAMPLER_PARAM_RE.match(param_name)
        if sampler_match is not None:
            params.append(_KernelParam("sampler", _coerce_int(sampler_match.group(1), 0), param_name))
            continue

        params.append(_KernelParam("unknown", None, param_name))

    return params


def _buffer_access_flags(read_access: int, write_access: int) -> int:
    read_enabled = int(read_access) != 0
    write_enabled = int(write_access) != 0

    if read_enabled and not write_enabled:
        return int(cl.mem_flags.READ_ONLY)
    if write_enabled and not read_enabled:
        return int(cl.mem_flags.WRITE_ONLY)
    return int(cl.mem_flags.READ_WRITE)


def _resolve_descriptor_buffer(
    descriptor_set: _DescriptorSet,
    binding: int,
    ctx: _Context,
    keepalive: List[object],
):
    binding_info = descriptor_set.buffer_bindings.get(int(binding))
    if binding_info is None:
        raise RuntimeError(f"Missing descriptor buffer binding {binding}")

    buffer_handle, offset, requested_range, _uniform, read_access, write_access = binding_info

    buffer_obj = _buffers.get(int(buffer_handle))
    if buffer_obj is None:
        raise RuntimeError(f"Invalid buffer handle {buffer_handle} for binding {binding}")

    offset = int(offset)
    requested_range = int(requested_range)

    if offset < 0:
        raise RuntimeError(f"Negative descriptor offset {offset} for binding {binding}")

    max_size = int(buffer_obj.size)
    if offset > max_size:
        raise RuntimeError(f"Descriptor offset {offset} exceeds buffer size {max_size} for binding {binding}")

    sub_size = max_size - offset if requested_range <= 0 else requested_range
    if sub_size < 0:
        raise RuntimeError(f"Invalid descriptor range {sub_size} for binding {binding}")

    if offset + sub_size > max_size:
        raise RuntimeError(
            f"Descriptor range (offset={offset}, size={sub_size}) exceeds buffer size {max_size} for binding {binding}"
        )

    if offset == 0 and sub_size == max_size:
        return buffer_obj.cl_buffer

    if (offset % ctx.sub_buffer_alignment) != 0:
        raise RuntimeError(
            f"Descriptor offset {offset} for binding {binding} is not aligned to "
            f"{ctx.sub_buffer_alignment} bytes required by this OpenCL device"
        )

    sub_buffer = buffer_obj.cl_buffer.get_sub_region(
        int(offset),
        int(sub_size),
        _buffer_access_flags(read_access, write_access),
    )
    keepalive.append(sub_buffer)
    return sub_buffer


def _build_kernel_args(
    plan: _ComputePlan,
    descriptor_set: Optional[_DescriptorSet],
    ctx: _Context,
    push_constant_payload: bytes = b"",
) -> Tuple[List[object], List[object]]:
    args: List[object] = []
    keepalive: List[object] = []

    for param in plan.params:
        if param.kind == "uniform":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")
            args.append(_resolve_descriptor_buffer(descriptor_set, 0, ctx, keepalive))
            continue

        if param.kind == "storage":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")
            if param.binding is None:
                raise RuntimeError("Storage parameter has no binding index")
            args.append(_resolve_descriptor_buffer(descriptor_set, int(param.binding), ctx, keepalive))
            continue

        if param.kind == "push_constant_value":
            if int(plan.pc_size) <= 0:
                raise RuntimeError(
                    f"Kernel parameter '{param.raw_name}' expects push-constant data, but this compute plan has pc_size={plan.pc_size}."
                )

            if len(push_constant_payload) == 0:
                raise RuntimeError(
                    "Missing push-constant payload for OpenCL by-value push-constant parameter "
                    f"'{param.raw_name}'."
                )

            if len(push_constant_payload) != int(plan.pc_size):
                raise RuntimeError(
                    f"Push-constant payload size mismatch for parameter '{param.raw_name}'. "
                    f"Expected {plan.pc_size} bytes but got {len(push_constant_payload)} bytes."
                )

            args.append(_repack_push_constant_payload(push_constant_payload, plan.pc_layout))
            continue

        if param.kind == "sampler":
            raise RuntimeError("OpenCL backend does not support image/sampler bindings")

        raise RuntimeError(
            f"Unsupported kernel parameter '{param.raw_name}'. "
            "Expected vkdispatch_uniform_ptr / vkdispatch_pc_value / vkdispatch_binding_<N>_ptr."
        )

    return args, keepalive


def _marker_wait_functions() -> List[object]:
    cached = getattr(_marker_helpers, "funcs", None)
    if cached is not None:
        return cached

    funcs: List[object] = []
    for fn_name in (
        "enqueue_marker",
        "enqueue_marker_with_wait_list",
        "enqueue_barrier_with_wait_list",
    ):
        fn = getattr(cl, fn_name, None)
        if fn is not None:
            funcs.append(fn)

    _marker_helpers.funcs = funcs
    return funcs


# --- API: context/init/logging ---


def init(debug, log_level):
    global _initialized, _debug_mode, _log_level

    _debug_mode = bool(debug)
    _log_level = int(log_level)
    _clear_error()

    if _initialized:
        return

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

    entries = _enumerate_opencl_devices()
    devices = []

    for entry in entries:
        device = entry.device
        opencl_version = _device_attr(device, "version", "")
        version_major, version_minor = _opencl_version_components(opencl_version)
        version_patch = 0

        driver_version = str(_device_attr(device, "driver_version", ""))
        driver_version_num = _driver_version_number(driver_version)

        vendor_id = _coerce_int(_device_attr(device, "vendor_id", 0), 0)
        device_id = int(entry.logical_index)
        device_type = _device_type_to_vkdispatch(_coerce_int(_device_attr(device, "type", 0), 0))
        device_name = str(_device_attr(device, "name", f"OpenCL Device {entry.logical_index}"))

        extensions = str(_device_attr(device, "extensions", ""))
        float32_atomic_support = (
            "cl_ext_float_atomics" in extensions
            or "cl_khr_float_atomics" in extensions
        )
        float64_support = "cl_khr_fp64" in extensions or _coerce_int(_device_attr(device, "double_fp_config", 0), 0) != 0
        float16_support = "cl_khr_fp16" in extensions or _coerce_int(_device_attr(device, "half_fp_config", 0), 0) != 0
        int64_support = _coerce_int(_device_attr(device, "address_bits", 0), 0) >= 64
        int16_support = _coerce_int(_device_attr(device, "preferred_vector_width_short", 0), 0) > 0

        max_work_item_sizes = tuple(
            _coerce_int(x, 1)
            for x in _device_attr(device, "max_work_item_sizes", (1, 1, 1))
        )
        if len(max_work_item_sizes) < 3:
            max_work_item_sizes = (
                max_work_item_sizes + (1, 1, 1)
            )[:3]
        else:
            max_work_item_sizes = max_work_item_sizes[:3]

        max_workgroup_size = (
            max(1, int(max_work_item_sizes[0])),
            max(1, int(max_work_item_sizes[1])),
            max(1, int(max_work_item_sizes[2])),
        )
        max_workgroup_invocations = max(1, _coerce_int(_device_attr(device, "max_work_group_size", 1), 1))

        max_workgroup_count = (2 ** 31 - 1, 2 ** 31 - 1, 2 ** 31 - 1)

        max_storage_buffer_range = max(
            1,
            min(
                _coerce_int(_device_attr(device, "max_mem_alloc_size", 1), 1),
                (1 << 31) - 1,
            ),
        )
        max_uniform_buffer_range = max(1, _coerce_int(_device_attr(device, "max_constant_buffer_size", 65536), 65536))
        uniform_alignment = max(
            1,
            _coerce_int(_device_attr(device, "mem_base_addr_align", 8), 8) // 8,
        )
        max_push_constant_size = max(0, _coerce_int(_device_attr(device, "max_parameter_size", 0), 0))

        # subgroup_size = max(
        #     1,
        #     _coerce_int(_device_attr(device, "preferred_work_group_size_multiple", 1), 1),
        # )

        max_compute_shared_memory_size = max(
            1,
            _coerce_int(_device_attr(device, "local_mem_size", 1), 1),
        )

        uuid_bytes = _device_uuid(entry, device_name, driver_version)

        devices.append(
            (
                0,  # Vulkan variant
                int(version_major),
                int(version_minor),
                int(version_patch),
                int(driver_version_num),
                int(vendor_id),
                int(device_id),
                int(device_type),
                str(device_name),
                1 if float32_atomic_support else 0,
                1 if float32_atomic_support else 0,
                1 if float64_support else 0,
                1 if float16_support else 0,
                1 if int64_support else 0,
                1 if int16_support else 0,
                1 if int16_support else 0,  # storage_buffer_16_bit_access
                1 if int16_support else 0,  # uniform_and_storage_buffer_16_bit_access
                0,  # storage_push_constant_16
                1 if int16_support else 0,  # storage_input_output_16
                max_workgroup_size,
                int(max_workgroup_invocations),
                max_workgroup_count,
                8,  # max descriptor sets (virtualized for parity)
                int(max_push_constant_size),
                int(max_storage_buffer_range),
                int(max_uniform_buffer_range),
                int(uniform_alignment),
                0,  # subgroup size
                0,  # subgroup stages
                0,  # subgroup operations
                0,  # quad operations in all stages
                int(max_compute_shared_memory_size),
                [(1, 0x006)],  # compute + transfer queue
                1,  # scalar block layout equivalent
                0,  # timeline semaphores equivalent
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
        _set_error("OpenCL backend currently supports exactly one device")
        return 0

    try:
        normalized_families = [[int(x) for x in family] for family in queue_families]
    except Exception:
        _set_error("context_create expected queue_families to be a nested integer list")
        return 0

    if len(normalized_families) != 1 or len(normalized_families[0]) != 1:
        _set_error("OpenCL backend currently supports exactly one queue")
        return 0

    entries = _enumerate_opencl_devices()
    if len(entries) == 0:
        if _error_string is None:
            _set_error("No OpenCL devices were found")
        return 0

    logical_device_index = int(device_ids[0])
    if logical_device_index < 0 or logical_device_index >= len(entries):
        _set_error(
            f"Invalid OpenCL device index {logical_device_index}. "
            f"Expected range [0, {len(entries) - 1}]"
        )
        return 0

    entry = entries[logical_device_index]

    try:
        cl_context = cl.Context(devices=[entry.device])
        queue = cl.CommandQueue(cl_context, device=entry.device)
        sub_buffer_alignment = max(
            1,
            _coerce_int(_device_attr(entry.device, "mem_base_addr_align", 8), 8) // 8,
        )
        ctx = _Context(
            device_index=logical_device_index,
            cl_context=cl_context,
            queues=[queue],
            queue_count=1,
            queue_to_device=[0],
            sub_buffer_alignment=sub_buffer_alignment,
            stopped=False,
        )
        return _new_handle(_contexts, ctx)
    except Exception as exc:
        _set_error(f"Failed to create OpenCL context: {exc}")
        return 0


def context_destroy(context):
    ctx = _contexts.pop(int(context), None)
    if ctx is None:
        return

    for queue in ctx.queues:
        try:
            queue.finish()
        except Exception:
            pass
        try:
            queue.release()
        except Exception:
            pass

    try:
        ctx.cl_context.release()
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


# --- API: signals ---


def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    _ = queue_index

    signal_obj = _signals.get(int(signal_ptr))
    if signal_obj is None:
        return True

    if not bool(wait_for_timestamp):
        if signal_obj.event is None:
            return bool(signal_obj.done)
        return bool(signal_obj.submitted)

    return _wait_signal(signal_obj)


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
        event_obj = None
        for marker_fn in _marker_wait_functions():
            try:
                event_obj = marker_fn(ctx.queues[selected[0]])
                if event_obj is not None:
                    break
            except TypeError:
                try:
                    event_obj = marker_fn(ctx.queues[selected[0]], wait_for=[])
                    if event_obj is not None:
                        break
                except Exception:
                    continue
            except Exception:
                continue

        if event_obj is None:
            ctx.queues[selected[0]].finish()
            signal.done = True
            signal.submitted = True
        else:
            _record_signal(signal, event_obj)
    except Exception as exc:
        _set_error(f"Failed to insert signal: {exc}")
        return 0

    return handle


def signal_destroy(signal_ptr):
    signal_obj = _signals.pop(int(signal_ptr), None)
    if signal_obj is None:
        return

    try:
        if signal_obj.event is not None:
            signal_obj.event.release()
    except Exception:
        pass


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
        cl_buffer = cl.Buffer(ctx.cl_context, cl.mem_flags.READ_WRITE, size=size)
        signal_handles = [
            _new_handle(_signals, _Signal(context_handle=int(context), queue_index=i, done=True))
            for i in range(ctx.queue_count)
        ]
        obj = _Buffer(
            context_handle=int(context),
            size=size,
            cl_buffer=cl_buffer,
            staging_data=[bytearray(size) for _ in range(ctx.queue_count)],
            signal_handles=signal_handles,
        )
        return _new_handle(_buffers, obj)
    except Exception as exc:
        _set_error(f"Failed to create OpenCL buffer: {exc}")
        return 0


def buffer_create_external(context, size, device_ptr):
    _ = context
    _ = size
    _ = device_ptr
    _set_error("OpenCL backend does not support external buffer aliases in MVP")
    return 0


def buffer_destroy(buffer):
    obj = _buffers.pop(int(buffer), None)
    if obj is None:
        return

    for signal_handle in obj.signal_handles:
        signal_destroy(signal_handle)

    try:
        obj.cl_buffer.release()
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

    obj.staging_data[queue_index][:size] = payload[:size]


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
        for queue_index in _queue_indices(ctx, int(index), all_on_negative=True):
            queue = ctx.queues[queue_index]
            end = min(offset + size, obj.size)
            copy_size = end - offset
            if copy_size <= 0:
                continue

            host_src = np.frombuffer(obj.staging_data[queue_index], dtype=np.uint8, count=copy_size)
            event_obj = cl.enqueue_copy(
                queue,
                obj.cl_buffer,
                host_src,
                dst_offset=offset,
                is_blocking=False,
            )

            signal_obj = _signals.get(obj.signal_handles[queue_index])
            if signal_obj is not None:
                _record_signal(signal_obj, event_obj)
    except Exception as exc:
        _set_error(f"Failed to write OpenCL buffer: {exc}")


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
        queue = ctx.queues[queue_index]
        end = min(offset + size, obj.size)
        copy_size = end - offset
        if copy_size <= 0:
            return

        host_dst = np.frombuffer(obj.staging_data[queue_index], dtype=np.uint8, count=copy_size)
        event_obj = cl.enqueue_copy(
            queue,
            host_dst,
            obj.cl_buffer,
            src_offset=offset,
            is_blocking=False,
        )

        signal_obj = _signals.get(obj.signal_handles[queue_index])
        if signal_obj is not None:
            _record_signal(signal_obj, event_obj)
    except Exception as exc:
        _set_error(f"Failed to read OpenCL buffer: {exc}")


# --- API: command lists ---


def command_list_create(context):
    if int(context) not in _contexts:
        _set_error("Invalid context handle for command_list_create")
        return 0

    return _new_handle(_command_lists, _CommandList(context_handle=int(context)))


def command_list_destroy(command_list):
    _command_lists.pop(int(command_list), None)


def command_list_get_instance_size(command_list):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return 0

    return int(sum(int(command.pc_size) for command in obj.commands))


def command_list_reset(command_list):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return

    obj.commands = []


def command_list_submit(command_list, data, instance_count, index):
    obj = _command_lists.get(int(command_list))
    if obj is None:
        return True

    ctx = _contexts.get(obj.context_handle)
    if ctx is None:
        _set_error(f"Missing context for command list {command_list}")
        return True

    instance_count = int(instance_count)
    if instance_count <= 0:
        return True

    instance_size = command_list_get_instance_size(command_list)
    payload = _to_bytes(data)
    expected_payload_size = int(instance_size) * int(instance_count)

    if expected_payload_size == 0:
        if len(payload) != 0:
            _set_error(
                f"Unexpected push-constant data for command list with instance_size=0 "
                f"(got {len(payload)} bytes)."
            )
            return True
    elif len(payload) != expected_payload_size:
        _set_error(
            f"Push-constant data size mismatch. Expected {expected_payload_size} bytes "
            f"(instance_size={instance_size}, instance_count={instance_count}) but got {len(payload)} bytes."
        )
        return True

    queue_targets = _queue_indices(ctx, int(index), all_on_negative=True)
    if len(queue_targets) == 0:
        queue_targets = [0]

    try:
        for queue_index in queue_targets:
            queue = ctx.queues[queue_index]
            for instance_index in range(instance_count):
                instance_base_offset = instance_index * instance_size
                per_instance_offset = 0
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

                    command_pc_size = int(command.pc_size)
                    pc_payload = b""
                    if command_pc_size > 0 and len(payload) > 0:
                        pc_start = instance_base_offset + per_instance_offset
                        pc_end = pc_start + command_pc_size
                        pc_payload = payload[pc_start:pc_end]

                    args, _keepalive = _build_kernel_args(
                        plan,
                        descriptor_set,
                        ctx,
                        pc_payload,
                    )

                    for arg_index, arg_value in enumerate(args):
                        plan.kernel.set_arg(arg_index, arg_value)

                    local_x = max(1, int(plan.local_size[0]))
                    local_y = max(1, int(plan.local_size[1]))
                    local_z = max(1, int(plan.local_size[2]))

                    blocks_x = max(1, int(command.blocks[0]))
                    blocks_y = max(1, int(command.blocks[1]))
                    blocks_z = max(1, int(command.blocks[2]))

                    global_size = (
                        blocks_x * local_x,
                        blocks_y * local_y,
                        blocks_z * local_z,
                    )

                    cl.enqueue_nd_range_kernel(
                        queue,
                        plan.kernel,
                        global_size,
                        (local_x, local_y, local_z),
                    )

                    per_instance_offset += command_pc_size

                if per_instance_offset != instance_size:
                    raise RuntimeError(
                        f"Internal command list size mismatch: computed {per_instance_offset} bytes, "
                        f"expected {instance_size} bytes."
                    )
    except Exception as exc:
        _set_error(f"Failed to submit OpenCL command list: {exc}")

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
    _ = descriptor_set
    _ = binding
    _ = object
    _ = sampler_obj
    _ = read_access
    _ = write_access
    _set_error("OpenCL backend does not support image objects in MVP")


# --- API: compute stage ---


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    source_bytes = _to_bytes(shader_source)
    shader_name_bytes = _to_bytes(shader_name)
    source_text = source_bytes.decode("utf-8", errors="replace")
    pc_size = int(pc_size)

    try:
        program = cl.Program(ctx.cl_context, source_text).build()
        kernel = cl.Kernel(program, "vkdispatch_main")
    except Exception as exc:
        kernel_name = shader_name_bytes.decode("utf-8", errors="replace")
        _set_error(f"Failed to compile OpenCL kernel '{kernel_name}': {exc}")
        return 0

    try:
        params = _parse_kernel_params(source_text)
        local_size = _parse_local_size(source_text)
        pc_layout = _build_push_constant_layout(source_text, pc_size)
    except Exception as exc:
        _set_error(f"Failed to parse OpenCL kernel metadata: {exc}")
        return 0

    plan = _ComputePlan(
        context_handle=int(context),
        shader_source=source_bytes,
        bindings=[int(x) for x in bindings],
        shader_name=shader_name_bytes,
        program=program,
        kernel=kernel,
        local_size=local_size,
        params=params,
        pc_size=pc_size,
        pc_layout=pc_layout,
    )

    return _new_handle(_compute_plans, plan)


def stage_compute_plan_destroy(plan):
    plan_obj = _compute_plans.pop(int(plan), None)
    if plan_obj is None:
        return

    try:
        plan_obj.kernel.release()
    except Exception:
        pass

    try:
        plan_obj.program.release()
    except Exception:
        pass


def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    cl_obj = _command_lists.get(int(command_list))
    cp_obj = _compute_plans.get(int(plan))
    if cl_obj is None or cp_obj is None:
        _set_error("Invalid command list or compute plan handle for stage_compute_record")
        return

    cl_obj.commands.append(
        _CommandRecord(
            plan_handle=int(plan),
            descriptor_set_handle=int(descriptor_set),
            blocks=(int(blocks_x), int(blocks_y), int(blocks_z)),
            pc_size=int(cp_obj.pc_size),
        )
    )


# --- API: images/samplers (MVP unsupported) ---


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    _ = context
    _ = extent
    _ = layers
    _ = format
    _ = type
    _ = view_type
    _ = generate_mips
    _set_error("OpenCL backend does not support image objects in MVP")
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
    _set_error("OpenCL backend does not support image samplers in MVP")
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
    _set_error("OpenCL backend does not support image writes in MVP")


def image_format_block_size(format):
    return int(_IMAGE_BLOCK_SIZES.get(int(format), 4))


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("OpenCL backend does not support image reads in MVP")
    return bytes(max(0, int(out_size)))


# --- API: FFT stage (MVP unsupported) ---


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
    _set_error("OpenCL backend does not support FFT plans in MVP")
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
    _set_error("OpenCL backend does not support FFT stages in MVP")


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
    "buffer_create_external",
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
