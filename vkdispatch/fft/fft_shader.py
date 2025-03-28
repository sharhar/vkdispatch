import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple
from functools import lru_cache
import numpy as np

from .fft_planner import make_fft_planner

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        name: str = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:

    if axis is None:
        axis = len(buffer_shape) - 1

    total_buffer_length = np.round(np.prod(buffer_shape)).astype(np.int32)

    fft_length = buffer_shape[axis]

    stride = np.round(np.prod(buffer_shape[axis + 1:])).astype(np.int32)
    batch_y_stride = stride * fft_length
    batch_y_count = total_buffer_length // batch_y_stride

    batch_z_stride = 1
    batch_z_count = stride

    fft_planner = make_fft_planner(
        N=fft_length,
        stride=stride,
        batch_y_stride=batch_y_stride,
        batch_z_stride=batch_z_stride,
        name=name
    )

    builder = vc.ShaderBuilder(enable_exec_bounds=False)
    old_builder = vc.set_global_builder(builder)

    signature = vd.ShaderSignature.from_type_annotations(builder, [Buff[c64]])
    buffer = signature.get_variables()[0]

    fft_planner.allocate_resources(batch_y_count, batch_z_count)

    fft_planner.plan(input=buffer, output=buffer, inverse=inverse, normalize_inverse=normalize_inverse)

    vc.set_global_builder(old_builder)

    shader_object = vd.ShaderObject(
        builder.build(f"{fft_planner.name}_{batch_y_count}_{batch_z_count}"),
        signature,
        local_size=fft_planner.resources.local_size
    )

    exec_size = (fft_planner.batch_threads, batch_y_count, batch_z_count)

    fft_planner.resources.reset()
    fft_planner.reset()

    return shader_object, exec_size


def get_cache_info():
    return make_fft_shader.cache_info()

def print_cache_info():
    print(get_cache_info())

def cache_clear():
    return make_fft_shader.cache_clear()