from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Optional

BACKEND_VULKAN = "vulkan"
BACKEND_PYCUDA = "pycuda"
BACKEND_CUDA_PYTHON = "cuda-python"
BACKEND_DUMMY = "dummy"

_BACKEND_ALIASES = {
    "cuda_python": BACKEND_CUDA_PYTHON,
    "cuda-bindings": BACKEND_CUDA_PYTHON,
    "cuda_bindings": BACKEND_CUDA_PYTHON,
}

CUDA_RUNTIME_BACKENDS = {BACKEND_PYCUDA, BACKEND_CUDA_PYTHON}

_VALID_BACKENDS = {BACKEND_VULKAN, BACKEND_PYCUDA, BACKEND_CUDA_PYTHON, BACKEND_DUMMY}
_active_backend_name: Optional[str] = None
_backend_modules: Dict[str, ModuleType] = {}


class BackendUnavailableError(ImportError):
    def __init__(self, backend_name: str, message: str):
        super().__init__(message)
        self.backend_name = backend_name


def normalize_backend_name(backend: Optional[str]) -> str:
    if backend is None:
        return BACKEND_VULKAN

    backend_name = backend.strip().lower()
    backend_name = _BACKEND_ALIASES.get(backend_name, backend_name)
    if backend_name not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unknown backend '{backend}'. Expected one of: {valid}")

    return backend_name


def set_active_backend(backend: str) -> str:
    global _active_backend_name

    backend_name = normalize_backend_name(backend)

    if _active_backend_name is not None and _active_backend_name != backend_name:
        raise RuntimeError(
            f"Backend is already set to '{_active_backend_name}' and cannot be changed to '{backend_name}' in this process."
        )

    _active_backend_name = backend_name
    return _active_backend_name


def clear_active_backend() -> None:
    global _active_backend_name
    _active_backend_name = None


def get_active_backend_name(default: Optional[str] = BACKEND_VULKAN) -> str:
    if _active_backend_name is not None:
        return _active_backend_name

    return normalize_backend_name(default)


def _load_backend_module(backend_name: str) -> ModuleType:
    if backend_name in _backend_modules:
        return _backend_modules[backend_name]

    try:
        if backend_name == BACKEND_VULKAN:
            module = importlib.import_module("vkdispatch_vulkan_native")
        elif backend_name == BACKEND_PYCUDA:
            module = importlib.import_module("vkdispatch.backends.pycuda_native")
        elif backend_name == BACKEND_CUDA_PYTHON:
            module = importlib.import_module("vkdispatch.backends.cuda_python_native")
        elif backend_name == BACKEND_DUMMY:
            module = importlib.import_module("vkdispatch.backends.dummy_native")
        else:
            # Defensive guard for future refactors.
            raise ValueError(f"Unsupported backend '{backend_name}'")
    except ImportError as exc:
        if backend_name == BACKEND_VULKAN:
            raise BackendUnavailableError(
                backend_name,
                "Vulkan backend is unavailable because the 'vkdispatch_native' package "
                f"could not be imported ({exc}).",
            ) from exc
        if backend_name == BACKEND_PYCUDA:
            raise BackendUnavailableError(
                backend_name,
                "PyCUDA backend is unavailable because the 'vkdispatch.backends.pycuda_native' "
                f"module could not be imported ({exc}).",
            ) from exc
        if backend_name == BACKEND_CUDA_PYTHON:
            raise BackendUnavailableError(
                backend_name,
                "CUDA Python backend is unavailable because the "
                "'vkdispatch.backends.cuda_python_native' module could not be imported "
                f"({exc}).",
            ) from exc
        raise

    _backend_modules[backend_name] = module
    return module


def get_backend_module(backend: Optional[str] = None) -> ModuleType:
    backend_name = normalize_backend_name(backend) if backend is not None else get_active_backend_name()
    return _load_backend_module(backend_name)


class _BackendProxy:
    def __getattr__(self, name: str):
        return getattr(get_backend_module(), name)


native = _BackendProxy()
