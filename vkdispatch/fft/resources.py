import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import dataclasses
from typing import List

from .config import FFTConfig
from .grid_manager import FFTGridManager

@dataclasses.dataclass
class FFTRegisterStageInvocation:
    output_stride: int
    block_width: int
    inner_block_offset: int
    block_index: int
    sub_sequence_offset: int
    register_selection: slice

    def __init__(self, stage_fft_length: int, stage_instance_count: int, output_stride: int, instance_index: int, tid: vc.ShaderVariable, N: int):
        self.output_stride = output_stride

        self.block_width = output_stride * stage_fft_length

        instance_index_stride = N // (stage_fft_length * stage_instance_count)

        self.instance_id = tid + instance_index_stride * instance_index

        self.inner_block_offset = self.instance_id % output_stride

        if output_stride == 1:
            self.inner_block_offset = 0
        
        self.sub_sequence_offset = self.instance_id * stage_fft_length - self.inner_block_offset * (stage_fft_length - 1)

        if self.block_width == N:
            self.inner_block_offset = self.instance_id
            self.sub_sequence_offset = self.inner_block_offset
        
        self.register_selection = slice(instance_index * stage_fft_length, (instance_index + 1) * stage_fft_length)


@dataclasses.dataclass
class FFTResources:
    registers: List[vc.ShaderVariable]
    radix_registers: List[vc.ShaderVariable]
    input_batch_offset: vc.ShaderVariable
    output_batch_offset: vc.ShaderVariable
    omega_register: vc.ShaderVariable
    subsequence_offset: Const[u32]
    io_index: Const[u32]
    io_index_2: Const[u32]

    output_strides: List[int]
    invocations: List[List[FFTRegisterStageInvocation]]

    def __init__(self, config: FFTConfig, grid: FFTGridManager):
        self.registers = [
            vc.new(c64, 0, var_name=f"register_{i}") for i in range(config.register_count)
        ]

        self.radix_registers = [
            vc.new(c64, 0, var_name=f"radix_{i}") for i in range(config.max_prime_radix)
        ]

        self.input_batch_offset = vc.new_uint(var_name="input_batch_offset")
        self.output_batch_offset = vc.new_uint(var_name="output_batch_offset")
        self.omega_register = vc.new(c64, 0, var_name="omega_register")
        self.subsequence_offset = vc.new_uint(0, var_name="subsequence_offset")
        self.io_index = vc.new_uint(0, var_name="io_index")
        self.io_index_2 = vc.new_uint(0, var_name="io_index_2")

        self.output_strides = []
        self.invocations = []
        
        output_stride = 1
        stage_count = len(config.stages)

        for i in range(stage_count):
            stage = config.stages[i]
            stage_invocations = []

            for ii in range(stage.instance_count):
                stage_invocations.append(FFTRegisterStageInvocation(
                    stage.fft_length,
                    stage.instance_count,
                    output_stride,
                    ii,
                    grid.tid,
                    config.N
            ))
                
            self.output_strides.append(output_stride)
            self.invocations.append(stage_invocations)
            
            output_stride *= config.stages[i].fft_length
