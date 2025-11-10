import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from typing import Tuple, Optional
from functools import lru_cache

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None) -> vd.ShaderFunction:

    with vd.fft.fft_context(buffer_shape, axis=axis) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map
        )

        io_manager.read_input(
            r2c=r2c,
            inverse=inverse,
            signal_range=input_signal_range
        )

        ctx.execute(inverse=inverse)

        if inverse and normalize_inverse:
            ctx.registers.normalize()

        io_manager.write_output(
            r2c=r2c,
            inverse=inverse
        )

    return ctx.get_callable()

@lru_cache(maxsize=None)
def get_transposed_size(
        buffer_shape: Tuple, 
        axis: int = None) -> vd.ShaderFunction:
    
    config = vd.fft.FFTConfig(buffer_shape, axis)
    grid = vd.fft.FFTGridManager(config, True, False)

    return np.prod(grid.local_size) * np.prod(grid.workgroup_count) * config.register_count

@lru_cache(maxsize=None)
def make_transpose_shader(
        buffer_shape: Tuple, 
        axis: int = None) -> vd.ShaderFunction:

    with vd.fft.fft_context(buffer_shape, axis=axis) as ctx:
        args = ctx.declare_shader_args([vc.Buffer[c64], vc.Buffer[c64]])

        for read_op in vd.fft.global_reads_iterator(ctx.registers, format_transposed=False):
            read_op.read_from_buffer(args[1])

        for write_op in vd.fft.global_trasposed_write_iterator(ctx.registers):
            write_op.write_to_buffer(args[0])

    return ctx.get_callable()

__static_global_kernel_index: int = None

def set_global_kernel_index(index: Optional[int]):
    global __static_global_kernel_index
    __static_global_kernel_index = index

def mapped_kernel_index() -> Optional[int]:
    return __static_global_kernel_index

@lru_cache(maxsize=None)
def make_convolution_shader(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None, 
        normalize: bool = True,
        transposed_kernel: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None) -> vd.ShaderFunction:

    if kernel_map is None:
        def kernel_map_func(kernel_buffer: vc.Buffer[c64]):
            read_op = vd.fft.mapped_read_op()
            
            kernel_val = vc.new_complex_register()
            read_op.read_from_buffer(kernel_buffer, register=kernel_val)
            
            read_op.register[:] = vc.mult_complex(read_op.register, kernel_val.conjugate())

        kernel_map = vd.map(kernel_map_func, input_types=[vc.Buffer[c64]])

    with vd.fft.fft_context(buffer_shape, axis=axis) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map,
            kernel_map=kernel_map
        )

        vc.comment("Performing forward FFT stage in convolution shader")

        io_manager.read_input(signal_range=input_signal_range) 
        ctx.execute(inverse=False)
        ctx.register_shuffle()

        vc.comment("Performing convolution stage in convolution shader")
        backup_registers = None

        if kernel_num > 1:
            backup_registers = ctx.allocate_registers("backup")
            backup_registers.read_from_registers(ctx.registers)

        for kern_index in range(kernel_num):
            vc.comment(f"Processing kernel {kern_index}")

            if backup_registers is not None:
                ctx.registers.read_from_registers(backup_registers)

            set_global_kernel_index(kern_index)
            io_manager.read_kernel(format_transposed=transposed_kernel)
            set_global_kernel_index(None)
            
            ctx.execute(inverse=True)

            if normalize:
                ctx.registers.normalize()

            io_manager.write_output(inverse=True)
    
    return ctx.get_callable()

def get_cache_info():
    return make_fft_shader.cache_info()

def get_convoliution_cache_info():
    return make_convolution_shader.cache_info()

def print_cache_info():
    print(get_cache_info())
    print(get_convoliution_cache_info())

def cache_clear():
    make_convolution_shader.cache_clear()
    make_fft_shader.cache_clear()