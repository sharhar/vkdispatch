import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple
from functools import lru_cache
import numpy as np

from .config import FFTConfig, FFTParams
from .resources import allocate_fft_resources

from .plan import plan

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        name: str = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True,
        r2c: bool = False) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:

    if name is None:
        name = f"fft_shader_{buffer_shape}_{axis}_{inverse}_{normalize_inverse}_{r2c}"

    with vc.builder_context(enable_exec_bounds=False) as builder:
        signature = vd.ShaderSignature.from_type_annotations(builder, [Buff[c64]])
        buffer = signature.get_variables()[0]

        fft_config = FFTConfig(buffer_shape, axis)
        
        resources = allocate_fft_resources(fft_config)

        plan(resources, fft_config.params(inverse, normalize_inverse, r2c), input=buffer, output=buffer)

        shader_object = vd.ShaderObject(
            builder.build(name),
            signature,
            local_size=resources.local_size
        )

        return shader_object, fft_config.exec_size

def get_cache_info():
    return make_fft_shader.cache_info()

def print_cache_info():
    print(get_cache_info())

def cache_clear():
    return make_fft_shader.cache_clear()