import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from ..compat import numpy_compat as npc

from typing import Tuple, Optional
from functools import lru_cache
import threading

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_type: vd.dtype = None,
        output_type: vd.dtype = None,
        compute_type: vd.dtype = None,
        input_signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None) -> vd.ShaderFunction:

    if output_type is None:
        output_type = vd.complex64

    if input_type is None and input_map is None:
        input_type = output_type

    if compute_type is None:
        compute_type = vd.complex64

    name = f"fft_shader_{buffer_shape}_{axis}_{inverse}_{normalize_inverse}_{r2c}"

    with vd.fft.fft_context(buffer_shape, axis=axis, compute_type=compute_type, name=name) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map,
            output_type=output_type,
            input_type=input_type,
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
        axis: int = None,
        compute_type: vd.dtype = vd.complex64) -> vd.ShaderFunction:
    
    config = vd.fft.FFTConfig(buffer_shape, axis, compute_type=compute_type)
    grid = vd.fft.FFTGridManager(config, True, False)

    return npc.prod(grid.local_size) * npc.prod(grid.workgroup_count) * config.register_count

@lru_cache(maxsize=None)
def make_transpose_shader(
        buffer_shape: Tuple, 
        axis: int = None,
        kernel_inner_only: bool = False,
        input_type: vd.dtype = vd.complex64,
        output_type: vd.dtype = vd.complex64,
        compute_type: vd.dtype = vd.complex64) -> vd.ShaderFunction:

    with vd.fft.fft_context(buffer_shape, axis=axis, compute_type=compute_type) as ctx:
        args = ctx.declare_shader_args([vc.Buffer[output_type], vc.Buffer[input_type]])

        if kernel_inner_only:
            vc.if_statement(ctx.grid.global_outer_offset == 0)

        for read_op in vd.fft.global_reads_iterator(ctx.registers, format_transposed=False):
            read_op.read_from_buffer(args[1])

        for write_op in vd.fft.global_trasposed_write_iterator(ctx.registers, inner_only=kernel_inner_only):
            write_op.write_to_buffer(args[0])

        if kernel_inner_only:
            vc.end()

    return ctx.get_callable()

_kernel_index_state = threading.local()

def set_global_kernel_index(index: Optional[int]):
    _kernel_index_state.index = index

def mapped_kernel_index() -> Optional[int]:
    return getattr(_kernel_index_state, "index", None)

@lru_cache(maxsize=None)
def make_convolution_shader(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None, 
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_type: vd.dtype = None,
        output_type: vd.dtype = None,
        kernel_type: vd.dtype = None,
        compute_type: vd.dtype = None,
        input_signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None) -> vd.ShaderFunction:

    if output_type is None:
        output_type = vd.complex64

    if input_type is None and input_map is None:
        input_type = output_type

    if kernel_type is None:
        kernel_type = vd.complex64

    if compute_type is None:
        compute_type = vd.complex64

    if kernel_map is None:
        def kernel_map_func(kernel_buffer: vc.Buffer[kernel_type]):
            read_op = vd.fft.read_op()
            
            kernel_val = vc.new_register(compute_type)
            read_op.read_from_buffer(kernel_buffer, register=kernel_val)
            
            read_op.register[:] = vc.mult_complex(read_op.register, kernel_val.conjugate())

        kernel_map = vd.map(kernel_map_func, input_types=[vc.Buffer[kernel_type]])

    name = f"convolution_shader_{buffer_shape}_{axis}"

    with vd.fft.fft_context(buffer_shape, axis=axis, compute_type=compute_type, name=name) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map,
            output_type=output_type,
            input_type=input_type,
            kernel_map=kernel_map
        )

        vc.comment("""Convolution pipeline phase 1/3.
Load spatial-domain input samples and run a forward FFT into frequency space.
Then shuffle registers so lane layout matches kernel application and inverse passes.""")

        io_manager.read_input(signal_range=input_signal_range) 
        ctx.execute(inverse=False)
        ctx.register_shuffle()

        backup_registers = None

        if kernel_num > 1:
            backup_registers = ctx.allocate_registers("backup")
            backup_registers.read_from_registers(ctx.registers)

        for kern_index in range(kernel_num):
            vc.comment(f"""Convolution pipeline phase 2/3. Kernel {kern_index + 1}/{kernel_num}.
Map this kernel onto the current spectrum.""")

            if backup_registers is not None:
                ctx.registers.read_from_registers(backup_registers)

            set_global_kernel_index(kern_index)
            io_manager.read_kernel(format_transposed=transposed_kernel, inner_only=kernel_inner_only)
            
            vc.comment(f"""Convolution pipeline phase 3/3.
Run inverse FFT back to the spatial domain, optionally normalize by length,
and write this kernel's output slice to global memory.""")

            ctx.execute(inverse=True)

            if normalize:
                ctx.registers.normalize()

            io_manager.write_output(inverse=True)

            set_global_kernel_index(None)
    
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
