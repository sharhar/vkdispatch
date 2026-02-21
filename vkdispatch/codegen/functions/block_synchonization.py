from ..global_builder import get_builder

from . import utils

def barrier():
    # On Apple devices, a memory barrier is required before a barrier
    # to ensure memory operations are visible to all threads
    # (for some reason)
    if get_builder().is_apple_device:
        memory_barrier()

    utils.append_contents("barrier();\n")

def memory_barrier():
    utils.append_contents("memoryBarrier();\n")

def memory_barrier_buffer():
    utils.append_contents("memoryBarrierBuffer();\n")

def memory_barrier_shared():
    utils.append_contents("memoryBarrierShared();\n")

def memory_barrier_image():
    utils.append_contents("memoryBarrierImage();\n")

def group_memory_barrier():
    utils.append_contents("groupMemoryBarrier();\n")