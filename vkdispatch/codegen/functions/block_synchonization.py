from ..global_builder import get_builder

from . import utils

def barrier():
    # On Apple devices, a memory barrier is required before a barrier
    # to ensure memory operations are visible to all threads
    # (for some reason)
    if get_builder().is_apple_device:
        memory_barrier()

    utils.append_contents(utils.codegen_backend().barrier_statement() + "\n")

def memory_barrier():
    utils.append_contents(utils.codegen_backend().memory_barrier_statement() + "\n")

def memory_barrier_buffer():
    utils.append_contents(utils.codegen_backend().memory_barrier_buffer_statement() + "\n")

def memory_barrier_shared():
    utils.append_contents(utils.codegen_backend().memory_barrier_shared_statement() + "\n")

def memory_barrier_image():
    utils.append_contents(utils.codegen_backend().memory_barrier_image_statement() + "\n")

def group_memory_barrier():
    utils.append_contents(utils.codegen_backend().group_memory_barrier_statement() + "\n")
