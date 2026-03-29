from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Optional

import os

BACKEND_VULKAN = "vulkan"
BACKEND_CUDA = "cuda"
BACKEND_OPENCL = "opencl"
BACKEND_DUMMY = "dummy"

_VALID_BACKENDS = {BACKEND_VULKAN, BACKEND_CUDA, BACKEND_OPENCL, BACKEND_DUMMY}
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

def get_environment_backend() -> Optional[str]:
    env_backend = os.environ.get("VKDISPATCH_BACKEND")
    if env_backend is not None:
        return normalize_backend_name(env_backend)
    return None

def get_active_backend_name(default: Optional[str] = None) -> str:
    if _active_backend_name is not None:
        return _active_backend_name
    
    if default is not None:
        return normalize_backend_name(default)

    env_backend = get_environment_backend()

    if env_backend is not None:
        return env_backend

    return BACKEND_VULKAN


def _load_backend_module(backend_name: str) -> ModuleType:
    if backend_name in _backend_modules:
        return _backend_modules[backend_name]

    try:
        if backend_name == BACKEND_VULKAN:
            module = importlib.import_module("vkdispatch_vulkan_native")
        elif backend_name == BACKEND_CUDA:
            module = importlib.import_module("vkdispatch.backends.cuda_backend")
        elif backend_name == BACKEND_OPENCL:
            module = importlib.import_module("vkdispatch.backends.opencl_backend")
        elif backend_name == BACKEND_DUMMY:
            module = importlib.import_module("vkdispatch.backends.dummy_backend")
        else:
            # Defensive guard for future refactors.
            raise ValueError(f"Unsupported backend '{backend_name}'")
    except ImportError as exc:
        if backend_name == BACKEND_VULKAN:
            raise BackendUnavailableError(
                backend_name,
                "Vulkan backend is unavailable because the 'vkdispatch_vulkan_native' package "
                f"could not be imported ({exc}).",
            ) from exc
        if backend_name == BACKEND_CUDA:
            raise BackendUnavailableError(
                backend_name,
                "CUDA Python backend is unavailable because the "
                "'vkdispatch.backends.cuda_backend' module could not be imported "
                f"({exc}).",
            ) from exc
        if backend_name == BACKEND_OPENCL:
            raise BackendUnavailableError(
                backend_name,
                "OpenCL backend is unavailable because the "
                "'vkdispatch.backends.opencl_backend' module could not be imported "
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
