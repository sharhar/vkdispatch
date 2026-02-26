"""cuda-python-backed runtime shim mirroring the vkdispatch_native API surface.

This module intentionally matches the function names exposed by the Cython
extension so existing Python runtime objects can call into either backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - import failure path
    raise ImportError(
        "The CUDA Python backend requires both 'cuda-python' and 'numpy' to be installed."
    ) from exc

try:
    from cuda.bindings import driver, nvrtc
except Exception:
    try:
        from cuda import cuda as driver  # type: ignore
        from cuda import nvrtc  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure path
        raise ImportError(
            "The CUDA Python backend requires the NVIDIA cuda-python package "
            "(`pip install cuda-python`)."
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


def _to_int(value) -> int:
    if isinstance(value, int):
        return int(value)

    if hasattr(value, "value"):
        try:
            return int(value.value)
        except Exception:
            pass

    return int(value)


def _drv_call(names, *args):
    if isinstance(names, str):
        names = [names]

    last_error = None
    for name in names:
        fn = getattr(driver, name, None)
        if fn is not None:
            try:
                return fn(*args)
            except TypeError as exc:
                last_error = exc
                continue

    if last_error is not None:
        raise RuntimeError(f"CUDA Driver call failed for {names}: {last_error}") from last_error
    raise RuntimeError(f"CUDA Driver symbol not found: {names}")


def _nvrtc_call(names, *args):
    if isinstance(names, str):
        names = [names]

    last_error = None
    for name in names:
        fn = getattr(nvrtc, name, None)
        if fn is not None:
            try:
                return fn(*args)
            except TypeError as exc:
                last_error = exc
                continue

    if last_error is not None:
        raise RuntimeError(f"NVRTC call failed for {names}: {last_error}") from last_error
    raise RuntimeError(f"NVRTC symbol not found: {names}")


def _status_success(status) -> bool:
    try:
        return _to_int(status) == 0
    except Exception:
        return str(status).endswith("CUDA_SUCCESS") or str(status).endswith("NVRTC_SUCCESS")


def _drv_error_string(status) -> str:
    try:
        name_res = _drv_call("cuGetErrorName", status)
        string_res = _drv_call("cuGetErrorString", status)
        _name_status = name_res[0] if isinstance(name_res, tuple) else 1
        _string_status = string_res[0] if isinstance(string_res, tuple) else 1
        if _status_success(_name_status) and _status_success(_string_status):
            name = name_res[1] if isinstance(name_res, tuple) and len(name_res) > 1 else name_res
            text = string_res[1] if isinstance(string_res, tuple) and len(string_res) > 1 else string_res
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="replace")
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors="replace")
            return f"{name}: {text}"
    except Exception:
        pass

    return str(status)


def _drv_check(result, op_name: str):
    if isinstance(result, tuple):
        status = result[0]
        payload = result[1:]
    else:
        status = result
        payload = ()

    if not _status_success(status):
        raise RuntimeError(f"{op_name} failed ({_drv_error_string(status)})")

    if len(payload) == 0:
        return None

    if len(payload) == 1:
        return payload[0]

    return payload


def _nvrtc_check(result, op_name: str):
    if isinstance(result, tuple):
        status = result[0]
        payload = result[1:]
    else:
        status = result
        payload = ()

    if not _status_success(status):
        raise RuntimeError(f"{op_name} failed ({status})")

    if len(payload) == 0:
        return None

    if len(payload) == 1:
        return payload[0]

    return payload


def _nvrtc_read_bytes(program, size_api: str, read_api: str) -> bytes:
    raw_size = _nvrtc_check(_nvrtc_call(size_api, program), size_api)
    size = int(_to_int(raw_size))
    if size <= 0:
        return b""

    def _normalize_output(data) -> Optional[bytes]:
        if data is None:
            return None

        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode("utf-8", errors="replace")

        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
            if len(raw) >= size:
                return raw[:size]
            return raw + (b"\x00" * (size - len(raw)))

        if isinstance(data, (tuple, list)):
            for item in data:
                normalized = _normalize_output(item)
                if normalized is not None:
                    return normalized

        return None

    try:
        direct_data = _nvrtc_check(_nvrtc_call(read_api, program), read_api)
        normalized = _normalize_output(direct_data)
        if normalized is not None:
            return normalized
    except Exception:
        pass

    out_c = ctypes.create_string_buffer(size)
    out_bytearray = bytearray(size)
    out_bytes = bytes(size)

    for out_candidate in (out_bytes, out_bytearray, out_c):
        try:
            call_result = _nvrtc_check(_nvrtc_call(read_api, program, out_candidate), read_api)
            normalized_result = _normalize_output(call_result)
            if normalized_result is not None:
                return normalized_result

            if isinstance(out_candidate, bytearray):
                return bytes(out_candidate)

            if out_candidate is out_c:
                return bytes(out_c.raw)
        except Exception:
            continue

    return bytes(out_c.raw)


def _discover_cuda_include_dirs() -> List[str]:
    include_dirs: List[str] = []
    seen = set()

    def add_dir(path_like) -> None:
        if path_like is None:
            return
        try:
            resolved = str(Path(path_like).resolve())
        except Exception:
            resolved = str(path_like)
        if resolved in seen:
            return
        header_path = Path(resolved) / "cuda_runtime.h"
        if header_path.exists():
            seen.add(resolved)
            include_dirs.append(resolved)

    # Standard CUDA environment variables.
    for env_name in (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDAToolkit_ROOT",
    ):
        root = os.environ.get(env_name)
        if root:
            add_dir(Path(root) / "include")

    # CUDA toolkit from nvcc location.
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            nvcc_root = Path(nvcc_path).resolve().parent.parent
            add_dir(nvcc_root / "include")
        except Exception:
            pass

    # Common Unix install locations.
    add_dir("/usr/local/cuda/include")
    add_dir("/opt/cuda/include")
    add_dir("/usr/include")

    # Conda cudatoolkit layouts.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        add_dir(Path(conda_prefix) / "include")
        add_dir(Path(conda_prefix) / "targets" / "x86_64-linux" / "include")
        add_dir(Path(conda_prefix) / "Library" / "include")

    # NVIDIA pip wheel layout.
    for base in sys.path:
        add_dir(Path(base) / "nvidia" / "cuda_runtime" / "include")

    # Some environments expose this namespace package.
    try:
        spec = importlib.util.find_spec("nvidia.cuda_runtime")
        if spec is not None and spec.submodule_search_locations:
            for entry in spec.submodule_search_locations:
                add_dir(Path(entry) / "include")
    except Exception:
        pass

    return include_dirs


def _prepare_nvrtc_options(options: List[bytes]) -> List[bytes]:
    normalized: List[bytes] = []
    has_include_path = False

    for opt in options:
        as_str = opt.decode("utf-8", errors="replace")
        if as_str.startswith("-I") or as_str.startswith("--include-path"):
            has_include_path = True
        normalized.append(opt)

    if not has_include_path:
        for include_dir in _discover_cuda_include_dirs():
            normalized.append(f"--include-path={include_dir}".encode("utf-8"))

    return normalized


def _as_driver_handle(type_name: str, value):
    handle_type = getattr(driver, type_name, None)
    if handle_type is None:
        return value

    try:
        if isinstance(value, handle_type):
            return value
    except Exception:
        pass

    try:
        return handle_type(_to_int(value))
    except Exception:
        return value


def _writable_host_ptr(view: memoryview):
    byte_view = view.cast("B")
    try:
        c_buffer = (ctypes.c_ubyte * len(byte_view)).from_buffer(byte_view)
        return ctypes.addressof(c_buffer), c_buffer
    except Exception:
        copied = ctypes.create_string_buffer(byte_view.tobytes())
        return ctypes.addressof(copied), copied


def _readonly_host_ptr(view: memoryview):
    byte_view = view.cast("B")
    try:
        c_buffer = (ctypes.c_ubyte * len(byte_view)).from_buffer(byte_view)
        return ctypes.addressof(c_buffer), c_buffer
    except Exception:
        copied = ctypes.create_string_buffer(byte_view.tobytes())
        return ctypes.addressof(copied), copied


class _DeviceAllocation:
    def __init__(self, ptr: int):
        self.ptr = int(ptr)
        self.freed = False

    def __int__(self):
        return int(self.ptr)

    def free(self):
        if self.freed:
            return

        _drv_check(
            _drv_call(
                ["cuMemFree", "cuMemFree_v2"],
                _as_driver_handle("CUdeviceptr", self.ptr),
            ),
            "cuMemFree",
        )
        self.freed = True


class _ContextHandle:
    def __init__(self, context_raw, device_index: int, uses_primary_context: bool):
        self.context_raw = context_raw
        self.device_index = int(device_index)
        self.uses_primary_context = bool(uses_primary_context)
        self._detached = False

    def push(self):
        _drv_check(
            _drv_call(
                "cuCtxPushCurrent",
                _as_driver_handle("CUcontext", self.context_raw),
            ),
            "cuCtxPushCurrent",
        )

    def detach(self):
        if self._detached:
            return

        if self.uses_primary_context:
            dev = _drv_check(_drv_call("cuDeviceGet", int(self.device_index)), "cuDeviceGet")
            _drv_check(_drv_call("cuDevicePrimaryCtxRelease", dev), "cuDevicePrimaryCtxRelease")
        else:
            _drv_check(
                _drv_call(
                    ["cuCtxDestroy", "cuCtxDestroy_v2"],
                    _as_driver_handle("CUcontext", self.context_raw),
                ),
                "cuCtxDestroy",
            )
        self._detached = True


class _StreamHandle:
    def __init__(self, handle: Optional[int] = None, ptr: Optional[int] = None, *args, **kwargs):
        _ = kwargs
        if handle is None and ptr is None and len(args) == 1:
            handle = int(args[0])
        if handle is None and ptr is not None:
            handle = int(ptr)

        if handle is None:
            stream_raw = _drv_check(_drv_call("cuStreamCreate", 0), "cuStreamCreate")
            self.handle = int(_to_int(stream_raw))
            self.owned = True
        else:
            self.handle = int(handle)
            self.owned = False

    def synchronize(self):
        _drv_check(
            _drv_call(
                "cuStreamSynchronize",
                _as_driver_handle("CUstream", self.handle),
            ),
            "cuStreamSynchronize",
        )

    def __int__(self):
        return int(self.handle)

    @property
    def ptr(self):
        return int(self.handle)

    @property
    def cuda_stream(self):
        return int(self.handle)


class _EventHandle:
    def __init__(self):
        self.event_raw = _drv_check(_drv_call("cuEventCreate", 0), "cuEventCreate")

    def record(self, stream_obj: Optional["_StreamHandle"]):
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        _drv_check(
            _drv_call(
                "cuEventRecord",
                self.event_raw,
                _as_driver_handle("CUstream", stream_handle),
            ),
            "cuEventRecord",
        )

    def query(self) -> bool:
        res = _drv_call("cuEventQuery", self.event_raw)
        status = res[0] if isinstance(res, tuple) else res

        if _status_success(status):
            return True

        status_text = str(status)
        if "NOT_READY" in status_text:
            return False

        if _to_int(status) != 0:
            return False

        return True

    def synchronize(self):
        _drv_check(_drv_call("cuEventSynchronize", self.event_raw), "cuEventSynchronize")


class _KernelFunction:
    def __init__(self, function_raw):
        self.function_raw = function_raw

    def __call__(self, *args, block, grid, stream=None):
        arg_values = []

        def _dedupe(values):
            out = []
            seen = set()
            for value in values:
                key = f"{type(value).__name__}:{repr(value)}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(value)
            return out

        arg_ptr_values = []
        for arg in args:
            if isinstance(arg, _ByValueKernelArg):
                payload = arg.payload
                if len(payload) == 0:
                    payload = b"\x00"

                payload_storage = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
                arg_values.append(payload_storage)
                arg_ptr_values.append(ctypes.addressof(payload_storage))
                continue

            scalar_storage = ctypes.c_uint64(int(arg))
            arg_values.append(scalar_storage)
            arg_ptr_values.append(ctypes.addressof(scalar_storage))

        arg_ptr_array = None
        if len(arg_ptr_values) > 0:
            arg_ptr_array = (ctypes.c_void_p * len(arg_ptr_values))(
                *[ctypes.c_void_p(ptr) for ptr in arg_ptr_values]
            )

        kernel_param_variants = [None, 0, ctypes.c_void_p(0)]
        if arg_ptr_array is not None:
            array_ptr = ctypes.cast(arg_ptr_array, ctypes.POINTER(ctypes.c_void_p))
            kernel_param_variants = _dedupe(
                [
                    arg_ptr_array,
                    array_ptr,
                    ctypes.cast(array_ptr, ctypes.c_void_p),
                    ctypes.cast(array_ptr, ctypes.c_void_p).value,
                    tuple(arg_ptr_values),
                    list(arg_ptr_values),
                ]
            )

        stream_handle = 0 if stream is None else int(stream)
        stream_variants = _dedupe(
            [
                stream_handle,
                _as_driver_handle("CUstream", stream_handle),
            ]
        )

        function_candidates = [
            self.function_raw,
            _as_driver_handle("CUfunction", self.function_raw),
        ]
        try:
            function_candidates.append(_to_int(self.function_raw))
        except Exception:
            pass
        function_variants = _dedupe(function_candidates)

        extra_variants = [None, 0, ctypes.c_void_p(0)]
        last_error = None

        for function_handle in function_variants:
            for stream_value in stream_variants:
                for kernel_params in kernel_param_variants:
                    for extra in extra_variants:
                        try:
                            _drv_check(
                                _drv_call(
                                    "cuLaunchKernel",
                                    function_handle,
                                    int(grid[0]),
                                    int(grid[1]),
                                    int(grid[2]),
                                    int(block[0]),
                                    int(block[1]),
                                    int(block[2]),
                                    0,
                                    stream_value,
                                    kernel_params,
                                    extra,
                                ),
                                "cuLaunchKernel",
                            )
                            return
                        except Exception as exc:
                            last_error = exc

                        try:
                            _drv_check(
                                _drv_call(
                                    "cuLaunchKernel",
                                    function_handle,
                                    int(grid[0]),
                                    int(grid[1]),
                                    int(grid[2]),
                                    int(block[0]),
                                    int(block[1]),
                                    int(block[2]),
                                    0,
                                    stream_value,
                                    kernel_params,
                                ),
                                "cuLaunchKernel",
                            )
                            return
                        except Exception as exc:
                            last_error = exc
                            continue

        if last_error is None:
            raise RuntimeError("cuLaunchKernel failed with no diagnostic.")
        raise RuntimeError(f"cuLaunchKernel failed: {last_error}") from last_error


class SourceModule:
    def __init__(self, source: str, no_extern_c: bool = True, options: Optional[List[str]] = None):
        _ = no_extern_c
        if options is None:
            options = []

        program_name = b"vkdispatch.cu"
        source_bytes = source.encode("utf-8")
        program = _nvrtc_check(
            _nvrtc_call(
                "nvrtcCreateProgram",
                source_bytes,
                program_name,
                0,
                [],
                [],
            ),
            "nvrtcCreateProgram",
        )

        ptx = b""
        build_log = b""

        try:
            encoded_options = [opt.encode("utf-8") if isinstance(opt, str) else bytes(opt) for opt in options]
            encoded_options = _prepare_nvrtc_options(encoded_options)
            compile_result = _nvrtc_call("nvrtcCompileProgram", program, len(encoded_options), encoded_options)
            compile_status = compile_result[0] if isinstance(compile_result, tuple) else compile_result

            build_log = _nvrtc_read_bytes(program, "nvrtcGetProgramLogSize", "nvrtcGetProgramLog")
            if not _status_success(compile_status):
                clean_build_log = build_log.rstrip(b"\x00").decode("utf-8", errors="replace")
                if "could not open source file \"cuda_runtime.h\"" in clean_build_log:
                    discovered = _discover_cuda_include_dirs()
                    hint = (
                        " NVRTC could not find CUDA headers. "
                        f"Discovered include dirs: {discovered if len(discovered) > 0 else 'none'}. "
                        "Set CUDA_HOME/CUDA_PATH to your toolkit root or ensure nvcc is on PATH."
                    )
                else:
                    hint = ""
                raise RuntimeError(
                    f"NVRTC compilation failed: {clean_build_log}{hint}"
                )

            ptx = _nvrtc_read_bytes(program, "nvrtcGetPTXSize", "nvrtcGetPTX")
        finally:
            try:
                _nvrtc_check(_nvrtc_call("nvrtcDestroyProgram", program), "nvrtcDestroyProgram")
            except Exception:
                pass

        if len(ptx) == 0:
            raise RuntimeError("NVRTC compilation succeeded but produced an empty PTX payload.")
        if not ptx.endswith(b"\x00"):
            ptx += b"\x00"

        self.module_raw = _drv_check(
            _drv_call(["cuModuleLoadDataEx", "cuModuleLoadData"], ptx),
            "cuModuleLoadData",
        )

    def get_function(self, name: str):
        func_raw = _drv_check(
            _drv_call("cuModuleGetFunction", self.module_raw, name.encode("utf-8")),
            "cuModuleGetFunction",
        )
        return _KernelFunction(func_raw)


class _CudaDevice:
    class device_attribute:
        MAX_BLOCK_DIM_X = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
            0,
        )
        MAX_BLOCK_DIM_Y = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
            0,
        )
        MAX_BLOCK_DIM_Z = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
            0,
        )
        MAX_THREADS_PER_BLOCK = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            0,
        )
        MAX_GRID_DIM_X = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
            0,
        )
        MAX_GRID_DIM_Y = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
            0,
        )
        MAX_GRID_DIM_Z = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
            0,
        )
        WARP_SIZE = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
            0,
        )
        MAX_SHARED_MEMORY_PER_BLOCK = getattr(
            getattr(driver, "CUdevice_attribute", object()),
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
            0,
        )

    class Device:
        def __init__(self, index: int):
            self.index = int(index)
            self.device_raw = _drv_check(_drv_call("cuDeviceGet", self.index), "cuDeviceGet")

        @staticmethod
        def count():
            return int(_drv_check(_drv_call("cuDeviceGetCount"), "cuDeviceGetCount"))

        def get_attributes(self):
            attrs = {}
            for attr_name in (
                "MAX_BLOCK_DIM_X",
                "MAX_BLOCK_DIM_Y",
                "MAX_BLOCK_DIM_Z",
                "MAX_THREADS_PER_BLOCK",
                "MAX_GRID_DIM_X",
                "MAX_GRID_DIM_Y",
                "MAX_GRID_DIM_Z",
                "WARP_SIZE",
                "MAX_SHARED_MEMORY_PER_BLOCK",
            ):
                attr_enum = getattr(_CudaDevice.device_attribute, attr_name)
                try:
                    val = _drv_check(
                        _drv_call("cuDeviceGetAttribute", attr_enum, self.device_raw),
                        "cuDeviceGetAttribute",
                    )
                    attrs[attr_enum] = int(val)
                except Exception:
                    attrs[attr_enum] = 0
            return attrs

        def compute_capability(self):
            major_enum = getattr(
                getattr(driver, "CUdevice_attribute", object()),
                "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
                0,
            )
            minor_enum = getattr(
                getattr(driver, "CUdevice_attribute", object()),
                "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
                0,
            )
            major = _drv_check(_drv_call("cuDeviceGetAttribute", major_enum, self.device_raw), "cuDeviceGetAttribute")
            minor = _drv_check(_drv_call("cuDeviceGetAttribute", minor_enum, self.device_raw), "cuDeviceGetAttribute")
            return int(major), int(minor)

        def total_memory(self):
            return int(_drv_check(_drv_call(["cuDeviceTotalMem", "cuDeviceTotalMem_v2"], self.device_raw), "cuDeviceTotalMem"))

        def pci_bus_id(self):
            try:
                bus_id = _drv_check(_drv_call("cuDeviceGetPCIBusId", 64, self.device_raw), "cuDeviceGetPCIBusId")
                if isinstance(bus_id, (bytes, bytearray)):
                    return bus_id.decode("utf-8", errors="replace").rstrip("\x00")
                return str(bus_id)
            except Exception:
                return f"cuda-device-{self.index}"

        def name(self):
            try:
                name = _drv_check(_drv_call("cuDeviceGetName", 128, self.device_raw), "cuDeviceGetName")
                if isinstance(name, (bytes, bytearray)):
                    return name.decode("utf-8", errors="replace").rstrip("\x00")
                return str(name)
            except Exception:
                return f"CUDA Device {self.index}"

        def retain_primary_context(self):
            ctx_raw = _drv_check(_drv_call("cuDevicePrimaryCtxRetain", self.device_raw), "cuDevicePrimaryCtxRetain")
            return _ContextHandle(ctx_raw, self.index, True)

        def make_context(self):
            ctx_raw = _drv_check(
                _drv_call(["cuCtxCreate", "cuCtxCreate_v2"], 0, self.device_raw),
                "cuCtxCreate",
            )
            return _ContextHandle(ctx_raw, self.index, False)

    class Context:
        @staticmethod
        def pop():
            try:
                _drv_check(_drv_call("cuCtxPopCurrent"), "cuCtxPopCurrent")
                return
            except Exception:
                pass

            popped = ctypes.c_void_p()
            _drv_check(_drv_call("cuCtxPopCurrent", popped), "cuCtxPopCurrent")

    Stream = _StreamHandle
    ExternalStream = _StreamHandle
    Event = _EventHandle
    DeviceAllocation = _DeviceAllocation
    device_attribute = device_attribute

    @staticmethod
    def init():
        _drv_check(_drv_call("cuInit", 0), "cuInit")

    @staticmethod
    def get_driver_version():
        return int(_drv_check(_drv_call("cuDriverGetVersion"), "cuDriverGetVersion"))

    @staticmethod
    def mem_alloc(size: int):
        ptr = _drv_check(
            _drv_call(["cuMemAlloc", "cuMemAlloc_v2"], int(size)),
            "cuMemAlloc",
        )
        return _DeviceAllocation(int(_to_int(ptr)))

    @staticmethod
    def memcpy_htod_async(dst_ptr, src_obj, stream_obj):
        src_view = memoryview(src_obj).cast("B")
        host_ptr, _keepalive = _readonly_host_ptr(src_view)
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        _drv_check(
            _drv_call(
                ["cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"],
                _as_driver_handle("CUdeviceptr", int(dst_ptr)),
                host_ptr,
                len(src_view),
                _as_driver_handle("CUstream", stream_handle),
            ),
            "cuMemcpyHtoDAsync",
        )

    @staticmethod
    def memcpy_dtoh_async(dst_obj, src_ptr, stream_obj):
        dst_view = memoryview(dst_obj).cast("B")
        host_ptr, _keepalive = _writable_host_ptr(dst_view)
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        _drv_check(
            _drv_call(
                ["cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"],
                host_ptr,
                _as_driver_handle("CUdeviceptr", int(src_ptr)),
                len(dst_view),
                _as_driver_handle("CUstream", stream_handle),
            ),
            "cuMemcpyDtoHAsync",
        )

    @staticmethod
    def pagelocked_empty(size: int, dtype):
        return np.empty(int(size), dtype=dtype)


cuda = _CudaDevice


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
    cuda_context: "cuda.Context"
    streams: List["cuda.Stream"]
    queue_count: int
    queue_to_device: List[int]
    max_kernel_param_size: int
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


@dataclass
class _KernelParam:
    kind: str
    binding: Optional[int]
    raw_name: str


@dataclass
class _ByValueKernelArg:
    payload: bytes
    raw_name: str


@dataclass
class _ComputePlan:
    context_handle: int
    shader_source: bytes
    bindings: List[int]
    shader_name: bytes
    module: SourceModule
    function: object
    local_size: Tuple[int, int, int]
    params: List[_KernelParam]
    pc_size: int


@dataclass
class _DescriptorSet:
    plan_handle: int
    buffer_bindings: Dict[int, Tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    image_bindings: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    inline_uniform_payload: bytes = b""


@dataclass
class _ResolvedLaunch:
    plan: _ComputePlan
    blocks: Tuple[int, int, int]
    descriptor_set: Optional[_DescriptorSet]
    pc_size: int
    pc_offset: int
    static_args: Optional[Tuple[object, ...]] = None


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
    ctx = _contexts.get(int(context_handle))
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

    print("Warning: Unable to query max kernel parameter size from CUDA driver. Falling back to a conservative default.", file=sys.stderr)

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

    buffer_obj = _buffers.get(int(buffer_handle))
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
        _set_error("CUDA Python backend currently supports exactly one device")
        return 0

    if len(queue_families) != 1 or len(queue_families[0]) != 1:
        _set_error("CUDA Python backend currently supports exactly one queue")
        return 0

    device_index = device_ids[0]

    cuda_context = None
    context_pushed = False

    try:
        if device_index < 0 or device_index >= cuda.Device.count():
            _set_error(f"Invalid CUDA device index {device_index}")
            return 0

        dev = cuda.Device(device_index)
        cc_major, _cc_minor = dev.compute_capability()
        max_kernel_param_size = _query_max_kernel_param_size(dev.device_raw, cc_major)
        uses_primary_context = False

        if hasattr(dev, "retain_primary_context"):
            cuda_context = dev.retain_primary_context()
            uses_primary_context = True
            cuda_context.push()
        else:  # pragma: no cover - fallback for older CUDA Python
            cuda_context = dev.make_context()
        context_pushed = True
        stream = cuda.Stream()

        ctx = _Context(
            device_index=device_index,
            cuda_context=cuda_context,
            streams=[stream],
            queue_count=1,
            queue_to_device=[0],
            max_kernel_param_size=int(max_kernel_param_size),
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

        if cuda_context is not None:
            try:
                cuda_context.detach()
            except Exception:
                pass

        _set_error(f"Failed to create CUDA Python context: {exc}")
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
        ctx.cuda_context.detach()
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
        # CUDA Python records signals synchronously on submission; host-side "recorded" waits
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
        with _activate_context(ctx):
            for queue_index in queue_targets:
                stream = _stream_for_queue(ctx, queue_index)
                resolved_launches: List[_ResolvedLaunch] = []
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
                    first_instance_payload = b""
                    if command_pc_size > 0 and len(payload) > 0:
                        first_instance_payload = payload[per_instance_offset: per_instance_offset + command_pc_size]

                    static_args = None
                    if command_pc_size == 0:
                        static_args = _build_kernel_args_template(plan, descriptor_set, b"")
                        size_check_args = static_args
                    else:
                        size_check_args = _build_kernel_args_template(
                            plan,
                            descriptor_set,
                            first_instance_payload,
                        )

                    estimated_param_size = _estimate_kernel_param_size_bytes(size_check_args)
                    if estimated_param_size > int(ctx.max_kernel_param_size):
                        shader_name = plan.shader_name.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Kernel '{shader_name}' launch parameters require "
                            f"{estimated_param_size} bytes, exceeding device limit "
                            f"{ctx.max_kernel_param_size} bytes. "
                            "Reduce by-value uniform/push-constant payload size or switch large "
                            "uniform data to buffer-backed arguments."
                        )
                    resolved_launches.append(
                        _ResolvedLaunch(
                            plan=plan,
                            blocks=command.blocks,
                            descriptor_set=descriptor_set,
                            pc_size=command_pc_size,
                            pc_offset=per_instance_offset,
                            static_args=static_args,
                        )
                    )
                    per_instance_offset += command_pc_size

                if per_instance_offset != instance_size:
                    raise RuntimeError(
                        f"Internal command list size mismatch: computed {per_instance_offset} bytes, "
                        f"expected {instance_size} bytes."
                    )

                for instance_index in range(instance_count):
                    instance_base_offset = instance_index * instance_size
                    for launch in resolved_launches:
                        if launch.static_args is not None:
                            args = launch.static_args
                        else:
                            pc_start = instance_base_offset + launch.pc_offset
                            pc_end = pc_start + launch.pc_size
                            pc_payload = payload[pc_start:pc_end]
                            args = _build_kernel_args_template(
                                launch.plan,
                                launch.descriptor_set,
                                pc_payload,
                            )

                        launch.plan.function(
                            *args,
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
    _ = descriptor_set
    _ = binding
    _ = object
    _ = sampler_obj
    _ = read_access
    _ = write_access
    _set_error("CUDA Python backend does not support image objects yet")


def descriptor_set_write_inline_uniform(descriptor_set, payload):
    ds = _descriptor_sets.get(int(descriptor_set))
    if ds is None:
        _set_error("Invalid descriptor set handle for descriptor_set_write_inline_uniform")
        return

    try:
        ds.inline_uniform_payload = _to_bytes(payload)
    except Exception as exc:
        _set_error(f"Failed to store inline uniform payload: {exc}")


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
        shader_name=shader_name_bytes,
        module=module,
        function=function,
        local_size=local_size,
        params=params,
        pc_size=int(pc_size),
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


# --- API: images/samplers (not yet implemented on CUDA Python backend) ---


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    _ = context
    _ = extent
    _ = layers
    _ = format
    _ = type
    _ = view_type
    _ = generate_mips
    _set_error("CUDA Python backend does not support image objects yet")
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
    _set_error("CUDA Python backend does not support image samplers yet")
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
    _set_error("CUDA Python backend does not support image writes yet")


def image_format_block_size(format):
    return int(_IMAGE_BLOCK_SIZES.get(int(format), 4))


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("CUDA Python backend does not support image reads yet")
    return bytes(max(0, int(out_size)))


# --- API: FFT stage (not yet implemented on CUDA Python backend) ---


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
    _set_error("CUDA Python backend does not support FFT plans yet")
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
    _set_error("CUDA Python backend does not support FFT stages yet")


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
    "descriptor_set_write_inline_uniform",
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
