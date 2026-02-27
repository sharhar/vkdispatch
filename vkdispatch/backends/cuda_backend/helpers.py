from __future__ import annotations

from contextlib import contextmanager
import re
import sys
from typing import Dict, List, Optional, Tuple

from . import state as state
from .bindings import driver, np, drv_call, drv_check, to_int
from .constants import (
    BINDING_PARAM_RE,
    KERNEL_SIGNATURE_RE,
    LOCAL_X_RE,
    LOCAL_Y_RE,
    LOCAL_Z_RE,
    SAMPLER_PARAM_RE,
)
from .cuda_primitives import _ByValueKernelArg, cuda
from .state import CUDABuffer, CUDAComputePlan, CUDAContext, CUDADescriptorSet, CUDAKernelParam, CUDASignal


def new_handle(registry: Dict[int, object], obj: object) -> int:
    handle = state.next_handle
    state.next_handle += 1
    registry[handle] = obj
    return handle


def to_bytes(value) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def set_error(message: str) -> None:
    state.error_string = str(message)


def clear_error() -> None:
    state.error_string = None


def coerce_stream_handle(stream_obj) -> Optional[int]:
    if stream_obj is None:
        return None

    if isinstance(stream_obj, int):
        return int(stream_obj)

    cuda_stream_protocol = getattr(stream_obj, "__cuda_stream__", None)
    if cuda_stream_protocol is not None:
        try:
            proto_value = cuda_stream_protocol() if callable(cuda_stream_protocol) else cuda_stream_protocol
            if isinstance(proto_value, tuple) and len(proto_value) > 0:
                proto_value = proto_value[0]
            return int(proto_value)
        except Exception:
            pass

    for attr_name in ("cuda_stream", "ptr", "handle"):
        if hasattr(stream_obj, attr_name):
            try:
                return int(getattr(stream_obj, attr_name))
            except Exception:
                pass

    nested = getattr(stream_obj, "stream", None)
    if nested is not None and nested is not stream_obj:
        try:
            return coerce_stream_handle(nested)
        except Exception:
            pass

    try:
        return int(stream_obj)
    except Exception as exc:
        raise TypeError(
            "Unable to extract a CUDA stream handle from the provided object. "
            "Pass an int handle or an object with __cuda_stream__/.cuda_stream/.ptr/.handle."
        ) from exc


def stream_override_stack() -> List[Optional[int]]:
    stack = getattr(state.stream_override, "stack", None)
    if stack is None:
        stack = []
        state.stream_override.stack = stack
    return stack


def get_stream_override_handle() -> Optional[int]:
    stack = getattr(state.stream_override, "stack", None)
    if not stack:
        return None
    return stack[-1]


def wrap_external_stream(handle: int):
    handle = int(handle)

    if handle in state.external_stream_cache:
        return state.external_stream_cache[handle]

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
            state.external_stream_cache[handle] = stream_obj
            return stream_obj
        except Exception as exc:  # pragma: no cover - depends on cuda-python version
            last_error = exc

    raise RuntimeError(
        f"Failed to wrap external CUDA stream handle {handle} with CUDA Python. "
        "This CUDA Python version may not support external stream wrappers."
    ) from last_error


def stream_for_queue(ctx: CUDAContext, queue_index: int):
    override_handle = get_stream_override_handle()
    if override_handle is None:
        return ctx.streams[queue_index]
    return wrap_external_stream(int(override_handle))


def buffer_device_ptr(buffer_obj: CUDABuffer) -> int:
    return int(buffer_obj.device_ptr)


def queue_indices(ctx: CUDAContext, queue_index: int, *, all_on_negative: bool = False) -> List[int]:
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


def context_from_handle(context_handle: int) -> Optional[CUDAContext]:
    ctx = state.contexts.get(int(context_handle))
    if ctx is None:
        set_error(f"Invalid context handle {context_handle}")
    return ctx


@contextmanager
def activate_context(ctx: CUDAContext):
    ctx.cuda_context.push()
    try:
        yield
    finally:
        cuda.Context.pop()


def record_signal(signal: CUDASignal, stream: "cuda.Stream") -> None:
    signal.submitted = True
    signal.done = False
    if signal.event is None:
        signal.event = cuda.Event()
    signal.event.record(stream)


def query_signal(signal: CUDASignal) -> bool:
    if signal.event is None:
        return bool(signal.done)

    try:
        done = signal.event.query()
    except Exception:
        return False

    signal.done = bool(done)
    return signal.done


def allocate_staging_storage(size: int):
    try:
        # Pagelocked host memory improves async HtoD/DtoH throughput and overlap.
        return cuda.pagelocked_empty(int(size), np.uint8)
    except Exception:
        return bytearray(int(size))


def fallback_max_kernel_param_size(compute_capability_major: int) -> int:
    # CUDA kernels support at least 4 KiB of launch parameters on legacy devices.
    # Volta+ devices commonly expose a larger 32 KiB-ish argument space.
    return 32764 if int(compute_capability_major) >= 7 else 4096


def query_max_kernel_param_size(device_raw, compute_capability_major: int) -> int:
    attr_names = (
        "CU_DEVICE_ATTRIBUTE_MAX_PARAMETER_SIZE",
        "CU_DEVICE_ATTRIBUTE_MAX_PARAMETER_SIZE_SUPPORTED",
        "CU_DEVICE_ATTRIBUTE_MAX_KERNEL_PARAMETER_SIZE",
    )

    attr_enum_container = getattr(driver, "CUdevice_attribute", None)
    if attr_enum_container is not None:
        for attr_name in attr_names:
            attr_enum = getattr(attr_enum_container, attr_name, None)
            if attr_enum is None:
                continue

            try:
                queried_value = drv_check(
                    drv_call("cuDeviceGetAttribute", attr_enum, device_raw),
                    "cuDeviceGetAttribute",
                )
                queried_size = int(to_int(queried_value))
                if queried_size > 0:
                    return queried_size
            except Exception:
                continue

    print(
        "Warning: Unable to query max kernel parameter size from CUDA driver. Falling back to a conservative default.",
        file=sys.stderr,
    )

    return fallback_max_kernel_param_size(compute_capability_major)


def parse_local_size(source: str) -> Tuple[int, int, int]:
    x_match = LOCAL_X_RE.search(source)
    y_match = LOCAL_Y_RE.search(source)
    z_match = LOCAL_Z_RE.search(source)

    x = int(x_match.group(1)) if x_match else 1
    y = int(y_match.group(1)) if y_match else 1
    z = int(z_match.group(1)) if z_match else 1

    return (x, y, z)


def parse_kernel_params(source: str) -> List[CUDAKernelParam]:
    signature_match = KERNEL_SIGNATURE_RE.search(source)
    if signature_match is None:
        raise RuntimeError("Could not find vkdispatch_main kernel signature in CUDA source")

    signature_blob = signature_match.group(1).strip()
    if len(signature_blob) == 0:
        return []

    params: List[CUDAKernelParam] = []

    for raw_decl in [part.strip() for part in signature_blob.split(",") if len(part.strip()) > 0]:
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", raw_decl)
        if name_match is None:
            raise RuntimeError(f"Unable to parse kernel parameter declaration '{raw_decl}'")

        param_name = name_match.group(1)

        if param_name == "vkdispatch_uniform_ptr":
            params.append(CUDAKernelParam("uniform", 0, param_name))
            continue

        if param_name == "vkdispatch_uniform_value":
            params.append(CUDAKernelParam("uniform_value", None, param_name))
            continue

        if param_name == "vkdispatch_pc_value":
            params.append(CUDAKernelParam("push_constant_value", None, param_name))
            continue

        binding_match = BINDING_PARAM_RE.match(param_name)
        if binding_match is not None:
            params.append(CUDAKernelParam("storage", int(binding_match.group(1)), param_name))
            continue

        sampler_match = SAMPLER_PARAM_RE.match(param_name)
        if sampler_match is not None:
            params.append(CUDAKernelParam("sampler", int(sampler_match.group(1)), param_name))
            continue

        params.append(CUDAKernelParam("unknown", None, param_name))

    return params


def resolve_buffer_pointer(descriptor_set: CUDADescriptorSet, binding: int) -> int:
    binding_info = descriptor_set.buffer_bindings.get(binding)
    if binding_info is None:
        raise RuntimeError(f"Missing descriptor buffer binding {binding}")

    buffer_handle, offset, _range, _uniform, _read_access, _write_access = binding_info

    buffer_obj = state.buffers.get(int(buffer_handle))
    if buffer_obj is None:
        raise RuntimeError(f"Invalid buffer handle {buffer_handle} for binding {binding}")

    return buffer_device_ptr(buffer_obj) + int(offset)


def build_kernel_args_template(
    plan: CUDAComputePlan,
    descriptor_set: Optional[CUDADescriptorSet],
    push_constant_payload: bytes = b"",
) -> Tuple[object, ...]:
    args: List[object] = []

    for param in plan.params:
        if param.kind == "uniform":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            args.append(np.uintp(resolve_buffer_pointer(descriptor_set, 0)))
            continue

        if param.kind == "uniform_value":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            if len(descriptor_set.inline_uniform_payload) == 0:
                raise RuntimeError(
                    "Missing inline uniform payload for CUDA by-value uniform parameter "
                    f"'{param.raw_name}'."
                )

            args.append(_ByValueKernelArg(descriptor_set.inline_uniform_payload, param.raw_name))
            continue

        if param.kind == "push_constant_value":
            if plan.pc_size <= 0:
                raise RuntimeError(
                    f"Kernel parameter '{param.raw_name}' expects push-constant data, but this compute plan has pc_size={plan.pc_size}."
                )

            if len(push_constant_payload) == 0:
                raise RuntimeError(
                    "Missing push-constant payload for CUDA by-value push-constant parameter "
                    f"'{param.raw_name}'."
                )

            if len(push_constant_payload) != int(plan.pc_size):
                raise RuntimeError(
                    f"Push-constant payload size mismatch for parameter '{param.raw_name}'. "
                    f"Expected {plan.pc_size} bytes but got {len(push_constant_payload)} bytes."
                )

            args.append(_ByValueKernelArg(push_constant_payload, param.raw_name))
            continue

        if param.kind == "storage":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            if param.binding is None:
                raise RuntimeError("Storage parameter has no binding index")

            args.append(np.uintp(resolve_buffer_pointer(descriptor_set, param.binding)))
            continue

        if param.kind == "sampler":
            raise RuntimeError("CUDA Python backend does not support sampled image bindings yet")

        raise RuntimeError(
            f"Unsupported kernel parameter '{param.raw_name}'. "
            "Expected vkdispatch_uniform_ptr / vkdispatch_uniform_value / vkdispatch_pc_value / vkdispatch_binding_<N>_ptr."
        )

    return tuple(args)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def estimate_kernel_param_size_bytes(args: Tuple[object, ...]) -> int:
    total_bytes = 0

    for arg in args:
        if isinstance(arg, _ByValueKernelArg):
            payload_size = len(arg.payload)
            # Kernel params are aligned by argument type. Use a conservative
            # 16-byte alignment for by-value structs.
            total_bytes = align_up(total_bytes, 16)
            total_bytes += payload_size
            continue

        total_bytes = align_up(total_bytes, 8)
        total_bytes += 8

    return total_bytes
