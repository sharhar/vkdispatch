import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple, Union
from functools import lru_cache
import numpy as np

from .memory_io import load_sdata_state_to_registers, FFTRegisterStageInvocation
from .config import FFTConfig, FFTParams
from .resources import allocate_fft_resources

from .plan import plan

import dataclasses

@dataclasses.dataclass
class FFTInputOutput:
    input_object: Union[vd.Buffer, vd.MappingFunction]
    output_object: Union[vd.Buffer, vd.MappingFunction]
    kernel_object: Union[vd.Buffer, vd.MappingFunction]

    input_types: List[vd.dtype]
    signature: vd.ShaderSignature

    in_buff: vd.Buffer
    out_buff: vd.Buffer

    input_buffers: List[vc.Buffer]
    output_buffers: List[vc.Buffer]
    kernel_buffers: List[vc.Buffer]

    def __init__(self,
                 builder: vc.ShaderBuilder,
                 input_object: Union[vd.Buffer, vd.MappingFunction] = None,
                 output_object: Union[vd.Buffer, vd.MappingFunction] = None,
                 kernel_object: Union[vd.Buffer, vd.MappingFunction] = None):
        self.input_object = input_object
        self.output_object = output_object
        self.kernel_object = kernel_object

        if self.input_object is None and self.output_object is None:
            self.input_types = [Buff[c64]]

        elif self.output_object is None:
            self.input_types = [Buff[c64]] + self.input_object.buffer_types

        elif self.input_object is None:
            self.input_types = self.output_object.buffer_types + [Buff[c64]]

        else:
            self.input_types = self.output_object.buffer_types + self.input_object.buffer_types

        kernel_count = 0

        if self.kernel_object is not None:
            self.input_types += self.kernel_object.buffer_types

            kernel_count = len(self.kernel_object.buffer_types)

        self.signature = vd.ShaderSignature.from_type_annotations(builder, self.input_types)
        sig_vars = self.signature.get_variables()

        io_vars = sig_vars[:len(sig_vars)-kernel_count]

        if self.input_object is None and self.output_object is None:
            self.input_buffers = None
            self.output_buffers = None

            self.in_buff = io_vars[0]
            self.out_buff = io_vars[0]

        elif self.output_object is None:
            self.input_buffers = io_vars[1:]
            self.output_buffers = None

            self.in_buff = input_object
            self.out_buff = io_vars[0]

        elif self.input_object is None:
            self.input_buffers = None
            self.output_buffers = io_vars[:-1]

            self.in_buff = io_vars[-1]
            self.out_buff = output_object

        else:
            self.input_buffers = io_vars[len(self.output_object.buffer_types):]
            self.output_buffers = io_vars[:len(self.output_object.buffer_types)]

            self.in_buff = input_object
            self.out_buff = output_object

        if kernel_count > 0:
            self.kernel_buffers = sig_vars[len(sig_vars)-kernel_count:]

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
        io_object = FFTInputOutput(builder, input_map, output_map)

        fft_config = FFTConfig(buffer_shape, axis)
        
        resources = allocate_fft_resources(fft_config, False)

        plan(
            resources,
            fft_config.params(
                inverse,
                normalize_inverse,
                r2c,
                input_buffers=io_object.input_buffers,
                output_buffers=io_object.output_buffers),
            input=io_object.in_buff,
            output=io_object.out_buff)

        shader_object = vd.ShaderObject(
            builder.build(name),
            io_object.signature,
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
        io_object = FFTInputOutput(builder, input_map, output_map, kernel_map)

        fft_config = FFTConfig(buffer_shape, axis)
        
        resources = allocate_fft_resources(fft_config, True)

        backup_registers = []
        for i in range(len(resources.registers)):
            backup_registers.append(vc.new(c64, 0, var_name=f"backup_register_{i}"))
        
        vc.comment("Performing forward FFT stage in convolution shader")

        do_sdata_padding = plan(
            resources,
            fft_config.params(
                inverse=False,
                input_buffers=io_object.input_buffers,
            ),
            input=io_object.in_buff)

        vc.memory_barrier()
        vc.barrier()

        vc.comment("Performing convolution stage in convolution shader")

        inverse_params = fft_config.params(
                inverse=True,
                normalize=normalize,
                input_buffers=io_object.kernel_buffers, 
                output_buffers=io_object.output_buffers)
        
        assert inverse_params.config.stages[0].instance_count == 1, "Something is very wrong"

        invocation = FFTRegisterStageInvocation(
            inverse_params.config.stages[0],
            1, 0,
            resources.tid,
            inverse_params.config.N
        )

        vc.comment(f"Loading state to registers in convolution shader")

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
            vc.memory_barrier()
            vc.barrier()
            
            for i in range(len(resources.registers)):
                resources.registers[i][:] = backup_registers[i]

            vc.set_kernel_index(kern_index)

            plan(
                resources,
                inverse_params,
                input=io_object.kernel_object,
                output=io_object.out_buff,
                do_sdata_padding=do_sdata_padding)

        shader_object = vd.ShaderObject(
            builder.build(name),
            io_object.signature,
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