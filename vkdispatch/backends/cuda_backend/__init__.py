"""cuda-python-backed runtime shim mirroring the vkdispatch_native API surface.

This module intentionally matches the function names exposed by the Cython
extension so existing Python runtime objects can call into either backend.
"""

from __future__ import annotations

from ._constants import (
    DESCRIPTOR_TYPE_SAMPLER,
    DESCRIPTOR_TYPE_STORAGE_BUFFER,
    DESCRIPTOR_TYPE_STORAGE_IMAGE,
    DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    DESCRIPTOR_TYPE_UNIFORM_IMAGE,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_VERBOSE,
    LOG_LEVEL_WARNING,
)
from ._cuda_primitives import SourceModule, cuda
from .api_buffer import (
    buffer_create,
    buffer_create_external,
    buffer_destroy,
    buffer_get_queue_signal,
    buffer_read,
    buffer_read_staging,
    buffer_wait_staging_idle,
    buffer_write,
    buffer_write_staging,
)
from .api_command_list import (
    command_list_create,
    command_list_destroy,
    command_list_get_instance_size,
    command_list_reset,
    command_list_submit,
)
from .api_compute import (
    stage_compute_plan_create,
    stage_compute_plan_destroy,
    stage_compute_record,
)
from .api_context import (
    context_create,
    context_destroy,
    context_stop_threads,
    cuda_stream_override_begin,
    cuda_stream_override_end,
    get_devices,
    get_error_string,
    init,
    log,
    set_log_level,
)
from .api_descriptor import (
    descriptor_set_create,
    descriptor_set_destroy,
    descriptor_set_write_buffer,
    descriptor_set_write_image,
    descriptor_set_write_inline_uniform,
)
from .api_image_fft import (
    image_create,
    image_create_sampler,
    image_destroy,
    image_destroy_sampler,
    image_format_block_size,
    image_read,
    image_write,
    stage_fft_plan_create,
    stage_fft_plan_destroy,
    stage_fft_record,
)
from .api_signal import signal_destroy, signal_insert, signal_wait


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
