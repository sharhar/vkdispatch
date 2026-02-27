from __future__ import annotations

from contextlib import contextmanager
import re
import sys
from typing import Dict, List, Optional, Tuple

from . import _state as state
from ._bindings import driver, np, _drv_call, _drv_check, _to_int
from ._constants import (
    _BINDING_PARAM_RE,
    _KERNEL_SIGNATURE_RE,
    _LOCAL_X_RE,
    _LOCAL_Y_RE,
    _LOCAL_Z_RE,
    _SAMPLER_PARAM_RE,
)
from ._cuda_primitives import _ByValueKernelArg, cuda
from ._state import _Buffer, _ComputePlan, _Context, _DescriptorSet, _KernelParam, _Signal


def _new_handle(registry: Dict[int, object], obj: object) -> int:
    handle = state._next_handle
    state._next_handle += 1
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
    state._error_string = str(message)


def _clear_error() -> None:
    state._error_string = None


def _coerce_stream_handle(stream_obj) -> Optional[int]:
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
            return _coerce_stream_handle(nested)
        except Exception:
            pass

    try:
        return int(stream_obj)
    except Exception as exc:
        raise TypeError(
            "Unable to extract a CUDA stream handle from the provided object. "
            "Pass an int handle or an object with __cuda_stream__/.cuda_stream/.ptr/.handle."
        ) from exc


def _stream_override_stack() -> List[Optional[int]]:
    stack = getattr(state._stream_override, "stack", None)
    if stack is None:
        stack = []
        state._stream_override.stack = stack
    return stack


def _get_stream_override_handle() -> Optional[int]:
    stack = getattr(state._stream_override, "stack", None)
    if not stack:
        return None
    return stack[-1]


def _wrap_external_stream(handle: int):
    handle = int(handle)

    if handle in state._external_stream_cache:
        return state._external_stream_cache[handle]

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
            state._external_stream_cache[handle] = stream_obj
            return stream_obj
        except Exception as exc:  # pragma: no cover - depends on cuda-python version
            last_error = exc

    raise RuntimeError(
        f"Failed to wrap external CUDA stream handle {handle} with CUDA Python. "
        "This CUDA Python version may not support external stream wrappers."
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
    ctx = state._contexts.get(int(context_handle))
    if ctx is None:
        _set_error(f"Invalid context handle {context_handle}")
    return ctx


@contextmanager
def _activate_context(ctx: _Context):
    ctx.cuda_context.push()
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


def _fallback_max_kernel_param_size(compute_capability_major: int) -> int:
    # CUDA kernels support at least 4 KiB of launch parameters on legacy devices.
    # Volta+ devices commonly expose a larger 32 KiB-ish argument space.
    return 32764 if int(compute_capability_major) >= 7 else 4096


def _query_max_kernel_param_size(device_raw, compute_capability_major: int) -> int:
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
                queried_value = _drv_check(
                    _drv_call("cuDeviceGetAttribute", attr_enum, device_raw),
                    "cuDeviceGetAttribute",
                )
                queried_size = int(_to_int(queried_value))
                if queried_size > 0:
                    return queried_size
            except Exception:
                continue

    print(
        "Warning: Unable to query max kernel parameter size from CUDA driver. Falling back to a conservative default.",
        file=sys.stderr,
    )

    return _fallback_max_kernel_param_size(compute_capability_major)


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

        if param_name == "vkdispatch_uniform_value":
            params.append(_KernelParam("uniform_value", None, param_name))
            continue

        if param_name == "vkdispatch_pc_value":
            params.append(_KernelParam("push_constant_value", None, param_name))
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

    buffer_obj = state._buffers.get(int(buffer_handle))
    if buffer_obj is None:
        raise RuntimeError(f"Invalid buffer handle {buffer_handle} for binding {binding}")

    return _buffer_device_ptr(buffer_obj) + int(offset)


def _build_kernel_args_template(
    plan: _ComputePlan,
    descriptor_set: Optional[_DescriptorSet],
    push_constant_payload: bytes = b"",
) -> Tuple[object, ...]:
    args: List[object] = []

    for param in plan.params:
        if param.kind == "uniform":
            if descriptor_set is None:
                raise RuntimeError("Kernel requires a descriptor set but none was provided")

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, 0)))
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

            args.append(np.uintp(_resolve_buffer_pointer(descriptor_set, param.binding)))
            continue

        if param.kind == "sampler":
            raise RuntimeError("CUDA Python backend does not support sampled image bindings yet")

        raise RuntimeError(
            f"Unsupported kernel parameter '{param.raw_name}'. "
            "Expected vkdispatch_uniform_ptr / vkdispatch_uniform_value / vkdispatch_pc_value / vkdispatch_binding_<N>_ptr."
        )

    return tuple(args)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_kernel_param_size_bytes(args: Tuple[object, ...]) -> int:
    total_bytes = 0

    for arg in args:
        if isinstance(arg, _ByValueKernelArg):
            payload_size = len(arg.payload)
            # Kernel params are aligned by argument type. Use a conservative
            # 16-byte alignment for by-value structs.
            total_bytes = _align_up(total_bytes, 16)
            total_bytes += payload_size
            continue

        total_bytes = _align_up(total_bytes, 8)
        total_bytes += 8

    return total_bytes
