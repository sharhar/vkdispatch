import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import dataclasses
from typing import List

from .config import FFTConfig
from .grid_manager import FFTGridManager

@dataclasses.dataclass
class FFTResources:
    input_batch_offset: vc.ShaderVariable
    output_batch_offset: vc.ShaderVariable
    omega_register: vc.ShaderVariable
    subsequence_offset: Const[u32]
    io_index: Const[u32]
    io_index_2: Const[u32]

    radix_registers: List[vc.ShaderVariable]

    tid: vc.ShaderVariable

    grid: FFTGridManager

    config: FFTConfig

    def __init__(self, config: FFTConfig, grid: FFTGridManager):
        self.tid = grid.tid
        self.grid = grid
        self.config = config
        self.input_batch_offset = vc.new_uint_register(var_name="input_batch_offset")
        self.output_batch_offset = vc.new_uint_register(var_name="output_batch_offset")
        self.omega_register = vc.new_register(config.compute_type, var_name="omega_register")
        self.subsequence_offset = vc.new_uint_register(var_name="subsequence_offset")
        self.io_index = vc.new_uint_register(var_name="io_index")
        self.io_index_2 = vc.new_uint_register(var_name="io_index_2")

        self.radix_registers = [
            vc.new_register(config.compute_type, var_name=f"radix_register_{i}") for i in range(config.max_prime_radix)
        ]

    def stage_begin(self, stage_index: int):
        thread_count = self.config.stages[stage_index].thread_count

        if thread_count < self.config.batch_threads:
            vc.if_statement(self.tid < thread_count)
    
    def stage_end(self, stage_index: int):
        thread_count = self.config.stages[stage_index].thread_count

        if thread_count < self.config.batch_threads:
            vc.end()

    def invocation_gaurd(self, stage_index: int, invocation_index: int):
        stage = self.config.stages[stage_index]

        if stage.remainder_offset == 1 and invocation_index == stage.extra_ffts:
            vc.if_statement(self.tid < self.config.N // stage.registers_used)

    def invocation_end(self, stage_index: int):
        stage = self.config.stages[stage_index]

        if stage.remainder_offset == 1:
            vc.end()
