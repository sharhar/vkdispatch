import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import dataclasses
from typing import List, Tuple
from functools import lru_cache
import numpy as np

from .resources import FFTResources
from .config import FFTRegisterStageConfig, FFTParams

from .memory_io import load_buffer_to_registers, store_registers_in_buffer



def set_batch_offsets(resources: FFTResources, params: FFTParams):
    input_batch_stride_y = params.batch_outer_stride
    output_batch_stride_y = params.batch_outer_stride

    if params.r2c and not params.inverse:
        output_batch_stride_y = (params.config.N // 2) + 1
        input_batch_stride_y = output_batch_stride_y * 2

    if params.r2c and params.inverse:
        input_batch_stride_y = (params.config.N // 2) + 1
        output_batch_stride_y = input_batch_stride_y * 2

    resources.input_batch_offset[:] = resources.global_outer_index * input_batch_stride_y + resources.global_inner_index * params.batch_inner_stride
    resources.output_batch_offset[:] = resources.global_outer_index * output_batch_stride_y + resources.global_inner_index * params.batch_inner_stride

def do_c64_mult_const(register_out: vc.ShaderVariable, register_in: vc.ShaderVariable, constant: complex):
    vc.comment(f"Multiplying {register_in} by {constant}")

    register_out.x = register_in.y * -constant.imag
    register_out.x = vc.fma(register_in.x, constant.real, register_out.x)

    register_out.y = register_in.y * constant.real
    register_out.y = vc.fma(register_in.x, constant.imag, register_out.y)

def radix_P(resources: FFTResources, params: FFTParams, register_list: List[vc.ShaderVariable]):
    assert len(register_list) <= len(resources.radix_registers), "Too many registers for radix_P"

    if len(register_list) == 1:
        return
    
    if len(register_list) == 2:
        vc.comment(f"Performing a DFT for Radix-2 FFT")
        resources.radix_registers[0][:] = register_list[1]
        register_list[1][:] = register_list[0] - resources.radix_registers[0]
        register_list[0][:] = register_list[0] + resources.radix_registers[0]
        return

    vc.comment(f"Performing a DFT for Radix-{len(register_list)} FFT")

    for i in range(0, len(register_list)):
        for j in range(0, len(register_list)):
            if j == 0:
                resources.radix_registers[i][:] = register_list[j]
                continue

            if i == 0:
                resources.radix_registers[i] += register_list[j]
                continue

            if i * j == len(register_list) // 2 and len(register_list) % 2 == 0:
                resources.radix_registers[i] -= register_list[j]
                continue

            omega = np.exp(1j * params.angle_factor * i * j / len(register_list))
            do_c64_mult_const(resources.omega_register, register_list[j], omega)
            resources.radix_registers[i] += resources.omega_register

    for i in range(0, len(register_list)):
        register_list[i][:] = resources.radix_registers[i]

def apply_cooley_tukey_twiddle_factors(resources: FFTResources, params: FFTParams, register_list: List[vc.ShaderVariable], twiddle_index: int = 0, twiddle_N: int = 1):
    if isinstance(twiddle_index, int) and twiddle_index == 0:
        return

    vc.comment(f"Applying Cooley-Tukey twiddle factors for twiddle index {twiddle_index} and twiddle N {twiddle_N}")

    if not isinstance(twiddle_index, int):
        resources.omega_register.x = params.angle_factor * twiddle_index / twiddle_N
        resources.omega_register[:] = vc.complex_from_euler_angle(resources.omega_register.x)
    
    inited_radix = False

    for i in range(len(register_list)):
        if i == 0:
            continue
        
        if isinstance(twiddle_index, int):
            if twiddle_index == 0:
                continue

            omega = np.exp(1j * params.angle_factor * i * twiddle_index / twiddle_N)

            scaled_angle = 2 * np.angle(omega) / np.pi
            rounded_angle = np.round(scaled_angle)

            if np.abs(scaled_angle - rounded_angle) < 1e-8:
                angle_int = int(rounded_angle)

                if angle_int == 1:
                    resources.omega_register.x = register_list[i].x
                    register_list[i].x = -register_list[i].y
                    register_list[i].y = resources.omega_register.x
                elif angle_int == -1:
                    resources.omega_register.x = register_list[i].x
                    register_list[i].x = register_list[i].y
                    register_list[i].y = -resources.omega_register.x
                elif angle_int == 2 or angle_int == -2:
                    register_list[i][:] = -register_list[i]
                
                continue

            do_c64_mult_const(resources.omega_register, register_list[i], omega)
            register_list[i][:] = resources.omega_register
            continue
        
        if not inited_radix:
            resources.radix_registers[1][:] = resources.omega_register
            inited_radix = True

        do_c64_mult_const(resources.radix_registers[0], register_list[i], resources.radix_registers[1])
        register_list[i][:] = resources.radix_registers[0]

        if i < len(register_list) - 1:
            do_c64_mult_const(resources.radix_registers[0], resources.omega_register, resources.radix_registers[1])
            resources.radix_registers[1][:] = resources.radix_registers[0]

def register_radix_composite(resources: FFTResources, params: FFTParams, register_list: List[vc.ShaderVariable], primes: List[int]):
    if len(register_list) == 1:
        return
    
    N = len(register_list)

    assert N == np.prod(primes), "Product of primes must be equal to the number of registers"

    vc.comment(f"Performing a Radix-{primes} FFT on {N} registers")

    output_stride = 1

    for prime in primes:
        sub_squences = [register_list[i::N//prime] for i in range(N//prime)]

        block_width = output_stride * prime

        for i in range(0, N // prime):
            inner_block_offset = i % output_stride
            block_index = (i * prime) // block_width

            apply_cooley_tukey_twiddle_factors(resources, params, sub_squences[i], twiddle_index=inner_block_offset, twiddle_N=block_width)
            radix_P(resources, params, sub_squences[i])
            
            sub_sequence_offset = block_index * block_width + inner_block_offset

            for j in range(prime):
                register_list[sub_sequence_offset + j * output_stride] = sub_squences[i][j]
        
        output_stride *= prime   

    return register_list

@dataclasses.dataclass
class FFTRegisterStageInvocation:
    stage: FFTRegisterStageConfig
    output_stride: int
    block_width: int
    inner_block_offset: int
    block_index: int
    sub_sequence_offset: int
    register_selection: slice

    def __init__(self, stage: FFTRegisterStageConfig, output_stride: int, instance_index: int, tid: vc.ShaderVariable):
        self.stage = stage
        self.output_stride = output_stride

        self.block_width = output_stride * stage.fft_length

        self.instance_id = tid * stage.instance_count + instance_index

        self.inner_block_offset = self.instance_id % output_stride

        if output_stride == 1:
            self.inner_block_offset = 0
        
        self.sub_sequence_offset = self.instance_id * stage.fft_length - self.inner_block_offset * (stage.fft_length - 1)

        self.register_selection = slice(instance_index * stage.fft_length, (instance_index + 1) * stage.fft_length)

def process_fft_register_stage(resources: FFTResources,
                               params: FFTParams, 
                               stage: FFTRegisterStageConfig, 
                               output_stride: int, 
                               input = None, 
                               output = None,
                               do_sdata_padding: bool = False) -> bool:
    do_runtime_if = stage.thread_count < params.config.batch_threads
    
    vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {params.config.N // stage.registers_used} groups")
    if do_runtime_if: vc.if_statement(resources.tid < stage.thread_count)

    stage_invocations: List[FFTRegisterStageInvocation] = []

    for i in range(stage.instance_count):
        stage_invocations.append(FFTRegisterStageInvocation(stage, output_stride, i , resources.tid))
    
    for ii, invocation in enumerate(stage_invocations):
        if stage.remainder_offset == 1 and ii == stage.extra_ffts:
            vc.if_statement(resources.tid < params.config.N // stage.registers_used)

        load_buffer_to_registers(
            resources=resources,
            params=params,
            buffer=input, 
            offset=invocation.instance_id, 
            stride=params.config.N // stage.fft_length, 
            register_list=resources.registers[invocation.register_selection],
            do_sdata_padding=do_sdata_padding
        )

        apply_cooley_tukey_twiddle_factors(
            resources=resources,
            params=params,
            register_list=resources.registers[invocation.register_selection], 
            twiddle_index=invocation.inner_block_offset, 
            twiddle_N=invocation.block_width
        )

        resources.registers[invocation.register_selection] = register_radix_composite(
            resources=resources,
            params=params,
            register_list=resources.registers[invocation.register_selection],
            primes=stage.primes
        )

    if stage.remainder_offset == 1:
        vc.end()

    if do_runtime_if: vc.end()

    if (input is None and output is None) or params.input_sdata:
        vc.memory_barrier()
        vc.barrier()

    if do_runtime_if: vc.if_statement(resources.tid < stage.thread_count)

    do_padding_next_list = []

    for ii, invocation in enumerate(stage_invocations):

        if stage.remainder_offset == 1 and ii == stage.extra_ffts:
            vc.if_statement(resources.tid < params.config.N // stage.registers_used)

        do_padding_next_list.append(store_registers_in_buffer(
            resources=resources,
            params=params,
            buffer=output,
            offset=invocation.sub_sequence_offset,
            stride=output_stride,
            register_list=resources.registers[invocation.register_selection]
        ))
    
    if stage.remainder_offset == 1:
        vc.end()

    if do_runtime_if: vc.end()

    assert any(do_padding_next_list) == all(do_padding_next_list), "do_sdata_padding must be the same for all invocations"

    return any(do_padding_next_list)

def plan(
        resources: FFTResources,
        params: FFTParams,
        input: Buff = None,
        output: Buff = None):

    set_batch_offsets(resources, params)

    output_stride = 1

    stage_count = len(params.config.stages)

    do_sdata_padding = False

    for i in range(stage_count):
        do_sdata_padding = process_fft_register_stage(
            resources,
            params,
            params.config.stages[i],
            output_stride,
            input=input if i == 0 else None,
            output=output if i == stage_count - 1 else None,
            do_sdata_padding=do_sdata_padding)
        
        output_stride *= params.config.stages[i].fft_length

        if i < stage_count - 1:
            vc.memory_barrier()
            vc.barrier()