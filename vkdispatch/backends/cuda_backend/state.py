from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Tuple

from .constants import LOG_LEVEL_WARNING
from .cuda_primitives import SourceModule, cuda


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
