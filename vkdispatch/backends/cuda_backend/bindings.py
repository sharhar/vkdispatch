from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import List, Optional

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
