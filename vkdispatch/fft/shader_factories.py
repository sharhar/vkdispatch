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
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:

    with vd.fft.fft_context(
        buffer_shape,
        axis=axis,
        input_map=input_map,
        output_map=output_map
    ) as ctx:
        
        ctx.read_input(
            r2c=r2c,
            inverse=inverse
        )

        ctx.execute(inverse=inverse)

        ctx.write_output(
            r2c=r2c,
            inverse=inverse,
            normalize=normalize_inverse
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
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:

    if kernel_map is None:
        def kernel_map_func(kernel_buffer: vc.Buffer[c64]):
            img_val = vc.mapping_registers()[0]
            read_register = vc.mapping_registers()[1]

            read_register[:] = kernel_buffer[vc.mapping_index()]
            img_val[:] = vc.mult_conj_c64(img_val, read_register)

        kernel_map = vd.map(kernel_map_func, register_types=[c64], input_types=[vc.Buffer[c64]])

    with vd.fft.fft_context(
        buffer_shape,
        axis=axis,
        input_map=input_map,
        output_map=output_map,
        kernel_map=kernel_map
    ) as ctx:
        vc.comment("Performing forward FFT stage in convolution shader")

        ctx.read_input()
        ctx.execute(inverse=False)

        ctx.register_shuffle()
        
        #vc.barrier()
        #ctx.write_sdata()
        #vc.barrier()

        vc.comment("Performing convolution stage in convolution shader")
        backup_registers = None

        if kernel_num > 1:
            backup_registers = []
            for i in range(len(ctx.resources.registers)):
                backup_registers.append(vc.new(c64, 0, var_name=f"backup_register_{i}"))

            for i in range(len(ctx.resources.registers)):
                backup_registers[i][:] = ctx.resources.registers[i]

        # If backup_registers is None, then the data is read into the main registers as desired
        #ctx.read_sdata(registers=backup_registers)
        #vc.barrier()

        for kern_index in range(kernel_num):
            vc.comment(f"Processing kernel {kern_index}")

            if backup_registers is not None:
                # Restore the main registers from backup if needed
                for i in range(len(ctx.resources.registers)):
                    ctx.resources.registers[i][:] = backup_registers[i]

            #vc.barrier()
            vc.set_kernel_index(kern_index)
            ctx.read_kernel()
            ctx.execute(inverse=True)
            ctx.write_output(inverse=True, normalize=normalize)
    
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