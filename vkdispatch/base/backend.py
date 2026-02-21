from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Optional

BACKEND_VULKAN = "vulkan"
BACKEND_PYCUDA = "pycuda"

_VALID_BACKENDS = {BACKEND_VULKAN, BACKEND_PYCUDA}
_active_backend_name: Optional[str] = None
_backend_modules: Dict[str, ModuleType] = {}


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


def get_active_backend_name(default: Optional[str] = BACKEND_VULKAN) -> str:
    if _active_backend_name is not None:
        return _active_backend_name

    return normalize_backend_name(default)


def _load_backend_module(backend_name: str) -> ModuleType:
    if backend_name in _backend_modules:
        return _backend_modules[backend_name]

    if backend_name == BACKEND_VULKAN:
        module = importlib.import_module("vkdispatch_native")
    elif backend_name == BACKEND_PYCUDA:
        module = importlib.import_module("vkdispatch.backends.pycuda_native")
    else:
        # Defensive guard for future refactors.
        raise ValueError(f"Unsupported backend '{backend_name}'")

    _backend_modules[backend_name] = module
    return module


def get_backend_module(backend: Optional[str] = None) -> ModuleType:
    backend_name = normalize_backend_name(backend) if backend is not None else get_active_backend_name()
    return _load_backend_module(backend_name)


class _BackendProxy:
    def __getattr__(self, name: str):
        return getattr(get_backend_module(), name)


native = _BackendProxy()
