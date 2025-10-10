import vkdispatch as vd
import vkdispatch.codegen as vc
import contextlib
from typing import Union, Tuple

from .manager import FFTManager

@contextlib.contextmanager
def fft_context(buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                output_map: Union[vd.MappingFunction, type, None] = None,
                input_map: Union[vd.MappingFunction, type, None] = None,
                kernel_map: Union[vd.MappingFunction, type, None] = None):

    try:
        with vc.builder_context(enable_exec_bounds=False) as builder:
            manager = FFTManager(
                builder=builder,
                buffer_shape=buffer_shape,
                axis=axis,
                max_register_count=max_register_count,
                output_map=output_map,
                input_map=input_map,
                kernel_map=kernel_map
            )

            yield manager

            manager.compile_shader()

    finally:
        pass        