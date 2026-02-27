from __future__ import annotations

import re

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

_LOCAL_X_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_X\s+(\d+)")
_LOCAL_Y_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_Y\s+(\d+)")
_LOCAL_Z_RE = re.compile(r"#define\s+VKDISPATCH_EXPECTED_LOCAL_SIZE_Z\s+(\d+)")
_KERNEL_SIGNATURE_RE = re.compile(r"vkdispatch_main\s*\(([^)]*)\)", re.S)
_BINDING_PARAM_RE = re.compile(r"vkdispatch_binding_(\d+)_ptr$")
_SAMPLER_PARAM_RE = re.compile(r"vkdispatch_sampler_(\d+)$")
