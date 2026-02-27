from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Tuple

from .constants import LOG_LEVEL_WARNING
from .cuda_primitives import SourceModule, cuda

#from .api_descriptor import CUDADescriptorSet

# --- Runtime state ---

initialized = False
debug_mode = False
log_level = LOG_LEVEL_WARNING
error_string: Optional[str] = None
next_handle = 1

contexts: Dict[int, "CUDAContext"] = {}
buffers: Dict[int, "CUDABuffer"] = {}
command_lists: Dict[int, "CUDACommandList"] = {}
compute_plans: Dict[int, "CUDAComputePlan"] = {}
external_stream_cache: Dict[int, object] = {}
stream_override = threading.local()


# --- Internal objects ---

@dataclass
class CUDAContext:
    device_index: int
    cuda_context: "cuda.Context"
    streams: List["cuda.Stream"]
    queue_count: int
    queue_to_device: List[int]
    max_kernel_param_size: int
    uses_primary_context: bool = False
    stopped: bool = False


@dataclass
class CUDABuffer:
    context_handle: int
    size: int
    device_ptr: int
    device_allocation: Optional["cuda.DeviceAllocation"]
    owns_allocation: bool
    staging_data: List[object]
    signal_handles: List[int]


@dataclass
class CUDACommandRecord:
    plan_handle: int
    descriptor_set_handle: int
    blocks: Tuple[int, int, int]
    pc_size: int


@dataclass
class CUDACommandList:
    context_handle: int
    commands: List[CUDACommandRecord] = field(default_factory=list)


@dataclass
class CUDAKernelParam:
    kind: str
    binding: Optional[int]
    raw_name: str


@dataclass
class CUDAComputePlan:
    context_handle: int
    shader_source: bytes
    bindings: List[int]
    shader_name: bytes
    module: SourceModule
    function: object
    local_size: Tuple[int, int, int]
    params: List[CUDAKernelParam]
    pc_size: int



