import vkdispatch.codegen as vc

import dataclasses
from typing import List, ContextManager

from .config import FFTConfig
from .grid_manager import FFTGridManager

import contextlib

@dataclasses.dataclass
class FFTResources:
    input_batch_offset: vc.ShaderVariable
    output_batch_offset: vc.ShaderVariable
    omega_register: vc.ShaderVariable
    subsequence_offset: vc.Const[vc.u32]
    io_index: vc.Const[vc.u32]
    io_index_2: vc.Const[vc.u32]

    radix_registers: List[vc.ShaderVariable]

    tid: vc.ShaderVariable
    grid: FFTGridManager
    config: FFTConfig

    stage_context: ContextManager
    invocation_context: ContextManager

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

        self.stage_context = None
        self.invocation_context = None

        self.radix_registers = [
            vc.new_register(config.compute_type, var_name=f"radix_register_{i}") for i in range(config.max_prime_radix)
        ]

    def stage_begin(self, stage_index: int):
        if self.stage_context is not None:
            raise RuntimeError("Stage context is already active. Cannot begin a new stage before ending the previous one.")

        thread_count = self.config.stages[stage_index].thread_count

        if thread_count >= self.config.batch_threads:
            return

        self.stage_context = vc.if_block(self.tid < thread_count)
        self.stage_context.__enter__()
    
    def stage_end(self, stage_index: int):
        thread_count = self.config.stages[stage_index].thread_count

        if thread_count >= self.config.batch_threads:
            return

        if self.stage_context is None:
            raise RuntimeError("No active stage context to end.")
        
        self.stage_context.__exit__(None, None, None)
        self.stage_context = None

    def invocation_gaurd(self, stage_index: int, invocation_index: int):
        stage = self.config.stages[stage_index]

        if stage.remainder_offset == 0 or invocation_index != stage.extra_ffts:
            return

        if self.invocation_context is not None:
            raise RuntimeError("Invocation context is already active. Cannot begin a new invocation guard before ending the previous one.")
        
        self.invocation_context = vc.if_block(self.tid < self.config.N // stage.registers_used)
        self.invocation_context.__enter__()

    def invocation_end(self, stage_index: int):
        stage = self.config.stages[stage_index]

        if stage.remainder_offset == 0:
            return

        if self.invocation_context is None:
            raise RuntimeError("No active invocation context to end.")
        
        self.invocation_context.__exit__(None, None, None)
        self.invocation_context = None
