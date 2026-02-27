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
