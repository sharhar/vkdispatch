from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import List, Optional

from .bindings import (
    np,
    driver,
    as_driver_handle,
    discover_cuda_include_dirs,
    drv_call,
    drv_check,
    nvrtc_call,
    nvrtc_check,
    nvrtc_read_bytes,
    prepare_nvrtc_options,
    readonly_host_ptr,
    status_success,
    to_int,
    writable_host_ptr,
)


@dataclass
class _ByValueKernelArg:
    payload: bytes
    raw_name: str


class _DeviceAllocation:
    def __init__(self, ptr: int):
        self.ptr = int(ptr)
        self.freed = False

    def __int__(self):
        return int(self.ptr)

    def free(self):
        if self.freed:
            return

        drv_check(
            drv_call(
                ["cuMemFree", "cuMemFree_v2"],
                as_driver_handle("CUdeviceptr", self.ptr),
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
        drv_check(
            drv_call(
                "cuCtxPushCurrent",
                as_driver_handle("CUcontext", self.context_raw),
            ),
            "cuCtxPushCurrent",
        )

    def detach(self):
        if self._detached:
            return

        if self.uses_primary_context:
            dev = drv_check(drv_call("cuDeviceGet", int(self.device_index)), "cuDeviceGet")
            drv_check(drv_call("cuDevicePrimaryCtxRelease", dev), "cuDevicePrimaryCtxRelease")
        else:
            drv_check(
                drv_call(
                    ["cuCtxDestroy", "cuCtxDestroy_v2"],
                    as_driver_handle("CUcontext", self.context_raw),
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
            stream_raw = drv_check(drv_call("cuStreamCreate", 0), "cuStreamCreate")
            self.handle = int(to_int(stream_raw))
            self.owned = True
        else:
            self.handle = int(handle)
            self.owned = False

    def synchronize(self):
        drv_check(
            drv_call(
                "cuStreamSynchronize",
                as_driver_handle("CUstream", self.handle),
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
        self.event_raw = drv_check(drv_call("cuEventCreate", 0), "cuEventCreate")

    def record(self, stream_obj: Optional["_StreamHandle"]):
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        drv_check(
            drv_call(
                "cuEventRecord",
                self.event_raw,
                as_driver_handle("CUstream", stream_handle),
            ),
            "cuEventRecord",
        )

    def query(self) -> bool:
        res = drv_call("cuEventQuery", self.event_raw)
        status = res[0] if isinstance(res, tuple) else res

        if status_success(status):
            return True

        status_text = str(status)
        if "NOT_READY" in status_text:
            return False

        if to_int(status) != 0:
            return False

        return True

    def synchronize(self):
        drv_check(drv_call("cuEventSynchronize", self.event_raw), "cuEventSynchronize")


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
                as_driver_handle("CUstream", stream_handle),
            ]
        )

        function_candidates = [
            self.function_raw,
            as_driver_handle("CUfunction", self.function_raw),
        ]
        try:
            function_candidates.append(to_int(self.function_raw))
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
                            drv_check(
                                drv_call(
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
                            drv_check(
                                drv_call(
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
        program = nvrtc_check(
            nvrtc_call(
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
            encoded_options = prepare_nvrtc_options(encoded_options)
            compile_result = nvrtc_call("nvrtcCompileProgram", program, len(encoded_options), encoded_options)
            compile_status = compile_result[0] if isinstance(compile_result, tuple) else compile_result

            build_log = nvrtc_read_bytes(program, "nvrtcGetProgramLogSize", "nvrtcGetProgramLog")
            if not status_success(compile_status):
                clean_build_log = build_log.rstrip(b"\x00").decode("utf-8", errors="replace")
                if 'could not open source file "cuda_runtime.h"' in clean_build_log:
                    discovered = discover_cuda_include_dirs()
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

            ptx = nvrtc_read_bytes(program, "nvrtcGetPTXSize", "nvrtcGetPTX")
        finally:
            try:
                nvrtc_check(nvrtc_call("nvrtcDestroyProgram", program), "nvrtcDestroyProgram")
            except Exception:
                pass

        if len(ptx) == 0:
            raise RuntimeError("NVRTC compilation succeeded but produced an empty PTX payload.")
        if not ptx.endswith(b"\x00"):
            ptx += b"\x00"

        self.module_raw = drv_check(
            drv_call(["cuModuleLoadDataEx", "cuModuleLoadData"], ptx),
            "cuModuleLoadData",
        )

    def get_function(self, name: str):
        func_raw = drv_check(
            drv_call("cuModuleGetFunction", self.module_raw, name.encode("utf-8")),
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
            self.device_raw = drv_check(drv_call("cuDeviceGet", self.index), "cuDeviceGet")

        @staticmethod
        def count():
            return int(drv_check(drv_call("cuDeviceGetCount"), "cuDeviceGetCount"))

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
                    val = drv_check(
                        drv_call("cuDeviceGetAttribute", attr_enum, self.device_raw),
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
            major = drv_check(drv_call("cuDeviceGetAttribute", major_enum, self.device_raw), "cuDeviceGetAttribute")
            minor = drv_check(drv_call("cuDeviceGetAttribute", minor_enum, self.device_raw), "cuDeviceGetAttribute")
            return int(major), int(minor)

        def total_memory(self):
            return int(drv_check(drv_call(["cuDeviceTotalMem", "cuDeviceTotalMem_v2"], self.device_raw), "cuDeviceTotalMem"))

        def pci_bus_id(self):
            try:
                bus_id = drv_check(drv_call("cuDeviceGetPCIBusId", 64, self.device_raw), "cuDeviceGetPCIBusId")
                if isinstance(bus_id, (bytes, bytearray)):
                    return bus_id.decode("utf-8", errors="replace").rstrip("\x00")
                return str(bus_id)
            except Exception:
                return f"cuda-device-{self.index}"

        def name(self):
            try:
                name = drv_check(drv_call("cuDeviceGetName", 128, self.device_raw), "cuDeviceGetName")
                if isinstance(name, (bytes, bytearray)):
                    return name.decode("utf-8", errors="replace").rstrip("\x00")
                return str(name)
            except Exception:
                return f"CUDA Device {self.index}"

        def retain_primary_context(self):
            ctx_raw = drv_check(drv_call("cuDevicePrimaryCtxRetain", self.device_raw), "cuDevicePrimaryCtxRetain")
            return _ContextHandle(ctx_raw, self.index, True)

        def make_context(self):
            ctx_raw = drv_check(
                drv_call(["cuCtxCreate", "cuCtxCreate_v2"], 0, self.device_raw),
                "cuCtxCreate",
            )
            return _ContextHandle(ctx_raw, self.index, False)

    class Context:
        @staticmethod
        def pop():
            try:
                drv_check(drv_call("cuCtxPopCurrent"), "cuCtxPopCurrent")
                return
            except Exception:
                pass

            popped = ctypes.c_void_p()
            drv_check(drv_call("cuCtxPopCurrent", popped), "cuCtxPopCurrent")

    Stream = _StreamHandle
    ExternalStream = _StreamHandle
    Event = _EventHandle
    DeviceAllocation = _DeviceAllocation
    device_attribute = device_attribute

    @staticmethod
    def init():
        drv_check(drv_call("cuInit", 0), "cuInit")

    @staticmethod
    def get_driver_version():
        return int(drv_check(drv_call("cuDriverGetVersion"), "cuDriverGetVersion"))

    @staticmethod
    def mem_alloc(size: int):
        ptr = drv_check(
            drv_call(["cuMemAlloc", "cuMemAlloc_v2"], int(size)),
            "cuMemAlloc",
        )
        return _DeviceAllocation(int(to_int(ptr)))

    @staticmethod
    def memcpy_htod_async(dst_ptr, src_obj, stream_obj):
        src_view = memoryview(src_obj).cast("B")
        host_ptr, _keepalive = readonly_host_ptr(src_view)
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        drv_check(
            drv_call(
                ["cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"],
                as_driver_handle("CUdeviceptr", int(dst_ptr)),
                host_ptr,
                len(src_view),
                as_driver_handle("CUstream", stream_handle),
            ),
            "cuMemcpyHtoDAsync",
        )

    @staticmethod
    def memcpy_dtoh_async(dst_obj, src_ptr, stream_obj):
        dst_view = memoryview(dst_obj).cast("B")
        host_ptr, _keepalive = writable_host_ptr(dst_view)
        stream_handle = 0 if stream_obj is None else int(stream_obj)
        drv_check(
            drv_call(
                ["cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"],
                host_ptr,
                as_driver_handle("CUdeviceptr", int(src_ptr)),
                len(dst_view),
                as_driver_handle("CUstream", stream_handle),
            ),
            "cuMemcpyDtoHAsync",
        )

    @staticmethod
    def pagelocked_empty(size: int, dtype):
        return np.empty(int(size), dtype=dtype)


cuda = _CudaDevice
