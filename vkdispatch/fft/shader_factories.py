import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import Tuple
from functools import lru_cache

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderFunction, Tuple[int, int, int]]:

    with vd.fft.fft_context(buffer_shape, axis=axis) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map
        )

        io_manager.read_input(
            r2c=r2c,
            inverse=inverse
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
def make_convolution_shader(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None, 
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderFunction, Tuple[int, int, int]]:

    if kernel_map is None:
        def kernel_map_func(kernel_buffer: vc.Buffer[c64]):
            read_op = vd.fft.mapped_read_op()
            
            kernel_val = vc.new_vec2(0)
            read_op.read_from_buffer(kernel_buffer, register=kernel_val)
            
            read_op.register[:] = vc.mult_conj_c64(read_op.register, kernel_val)

        kernel_map = vd.map(kernel_map_func, input_types=[vc.Buffer[c64]])

    with vd.fft.fft_context(buffer_shape, axis=axis) as ctx:
        io_manager = ctx.make_io_manager(
            input_map=input_map,
            output_map=output_map,
            kernel_map=kernel_map
        )

        vc.comment("Performing forward FFT stage in convolution shader")

        io_manager.read_input() 
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

            vc.set_kernel_index(kern_index)
            io_manager.read_kernel()
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