import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple, Union
from functools import lru_cache
import numpy as np

from .memory_io import load_sdata_state_to_registers, FFTRegisterStageInvocation
from .config import FFTConfig, FFTParams
from .resources import allocate_fft_resources

from .io_manager import IOManager

from .plan import plan

@lru_cache(maxsize=None)
def make_fft_shader(
        buffer_shape: Tuple, 
        axis: int = None, 
        name: str = None, 
        inverse: bool = False, 
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:

    if name is None:
        name = f"fft_shader_{buffer_shape}_{axis}_{inverse}_{normalize_inverse}_{r2c}"

    with vc.builder_context(enable_exec_bounds=False) as builder:
        io_manager = IOManager(builder, output_map, input_map)

        fft_config = FFTConfig(buffer_shape, axis)
        
        resources = allocate_fft_resources(fft_config, False)

        plan(
            resources,
            fft_config.params(
                inverse,
                normalize_inverse,
                r2c),
            input=io_manager.input_proxy,
            output=io_manager.output_proxy)

        shader_object = vd.ShaderObject(
            builder.build(name),
            io_manager.signature,
            local_size=resources.local_size
        )

        return shader_object, resources.exec_size

@lru_cache(maxsize=None)
def make_convolution_shader(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None, 
        name: str = None, 
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None) -> Tuple[vd.ShaderObject, Tuple[int, int, int]]:
    if name is None:
        name = f"convolution_shader_{buffer_shape}_{axis}"

    if kernel_map is None:
        def kernel_map_func(kernel_buffer: vc.Buffer[c64]):
            img_val = vc.mapping_registers()[0]
            read_register = vc.mapping_registers()[1]

            read_register[:] = kernel_buffer[vc.mapping_index()]
            img_val[:] = vc.mult_conj_c64(img_val, read_register)

        kernel_map = vd.map(kernel_map_func, register_types=[c64], input_types=[vc.Buffer[c64]])

    with vc.builder_context(enable_exec_bounds=False) as builder:
        io_manager = IOManager(builder, output_map, input_map, kernel_map)

        fft_config = FFTConfig(buffer_shape, axis)
        
        resources = allocate_fft_resources(fft_config, True)

        vc.comment("Performing forward FFT stage in convolution shader")

        do_sdata_padding = plan(
            resources,
            fft_config.params(
                inverse=False,
            ),
            input=io_manager.input_proxy)

        vc.barrier()

        vc.comment("Performing convolution stage in convolution shader")

        inverse_params = fft_config.params(
                inverse=True,
                normalize=normalize)
        
        assert inverse_params.config.stages[0].instance_count == 1, "Something is very wrong"

        invocation = FFTRegisterStageInvocation(
            inverse_params.config.stages[0],
            1, 0,
            resources.tid,
            inverse_params.config.N
        )

        vc.comment(f"Loading state to registers in convolution shader")

        if kernel_num == 1:
            load_sdata_state_to_registers(
                resources,
                inverse_params,
                invocation.instance_id,
                inverse_params.config.N // inverse_params.config.stages[0].fft_length,
                resources.registers[invocation.register_selection],
                do_sdata_padding
            )

            vc.comment("Performing IFFT stage in convolution shader")

            vc.barrier()
                
            vc.set_kernel_index(0)

            plan(
                resources,
                inverse_params,
                input=io_manager.kernel_proxy,
                output=io_manager.output_proxy,
                do_sdata_padding=do_sdata_padding)

            shader_object = vd.ShaderObject(
                builder.build(name),
                io_manager.signature,
                local_size=resources.local_size
            )

            return shader_object, resources.exec_size

        backup_registers = []
        for i in range(len(resources.registers)):
            backup_registers.append(vc.new(c64, 0, var_name=f"backup_register_{i}"))

        load_sdata_state_to_registers(
            resources,
            inverse_params,
            invocation.instance_id,
            inverse_params.config.N // inverse_params.config.stages[0].fft_length,
            backup_registers[invocation.register_selection],
            do_sdata_padding
        )

        vc.comment("Performing IFFT stage in convolution shader")

        for kern_index in range(kernel_num):
            vc.barrier()
            
            for i in range(len(resources.registers)):
                resources.registers[i][:] = backup_registers[i]

            vc.set_kernel_index(kern_index)

            plan(
                resources,
                inverse_params,
                input=io_manager.kernel_proxy,
                output=io_manager.output_proxy,
                do_sdata_padding=do_sdata_padding)

        shader_object = vd.ShaderObject(
            builder.build(name),
            io_manager.signature,
            local_size=resources.local_size
        )

        return shader_object, resources.exec_size

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