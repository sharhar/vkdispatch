import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from .arithmetic import is_number
from typing import Any, Union, Tuple

from ..global_codegen_callbacks import append_contents

from ..global_builder import GlobalBuilder

import numpy as np

from .common_builtins import dtype_to_floating, resolve_input

def barrier():
    # On Apple devices, a memory barrier is required before a barrier
    # to ensure memory operations are visible to all threads
    # (for some reason)
    if GlobalBuilder.obj.is_apple_device:
        memory_barrier()

    append_contents("barrier();\n")

def memory_barrier():
    append_contents("memoryBarrier();\n")

def memory_barrier_buffer():
    append_contents("memoryBarrierBuffer();\n")

def memory_barrier_shared():
    append_contents("memoryBarrierShared();\n")

def memory_barrier_image():
    append_contents("memoryBarrierImage();\n")

def group_memory_barrier():
    append_contents("groupMemoryBarrier();\n")