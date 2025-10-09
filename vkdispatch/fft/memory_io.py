import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple, Optional

from .resources import FFTResources
from .config import FFTRegisterStageConfig, FFTParams

from .io_proxy import IOProxy

import dataclasses

@dataclasses.dataclass
class FFTRegisterStageInvocation:
    stage: FFTRegisterStageConfig
    output_stride: int
    block_width: int
    inner_block_offset: int
    block_index: int
    sub_sequence_offset: int
    register_selection: slice

    def __init__(self, stage: FFTRegisterStageConfig, output_stride: int, instance_index: int, tid: vc.ShaderVariable, N: int):
        self.stage = stage
        self.output_stride = output_stride

        self.block_width = output_stride * stage.fft_length

        instance_index_stride = N // (stage.fft_length * stage.instance_count)

        self.instance_id = tid + instance_index_stride * instance_index

        self.inner_block_offset = self.instance_id % output_stride

        if output_stride == 1:
            self.inner_block_offset = 0
        
        self.sub_sequence_offset = self.instance_id * stage.fft_length - self.inner_block_offset * (stage.fft_length - 1)

        if self.block_width == N:
            self.inner_block_offset = self.instance_id
            self.sub_sequence_offset = self.inner_block_offset
        
        self.register_selection = slice(instance_index * stage.fft_length, (instance_index + 1) * stage.fft_length)

def load_sdata_state_to_registers(
        resources: FFTResources,
        params: FFTParams,
        offset: Const[u32],
        stride: int,
        register_list: List[vc.ShaderVariable] = None,
        do_sdata_padding: bool = False) -> None:
    
    for i in range(len(register_list)):
        resources.io_index[:] = i * stride + offset

        if resources.sdata_offset is not None:
            resources.io_index[:] = resources.io_index + resources.sdata_offset

        if do_sdata_padding:
            resources.io_index[:] = resources.io_index + resources.io_index / params.sdata_row_size

        register_list[i][:] = resources.sdata[resources.io_index]

def load_buffer_to_registers(
        resources: FFTResources,
        params: FFTParams,
        buffer: Optional[IOProxy],
        offset: Const[u32],
        stride: int,
        register_list: List[vc.ShaderVariable] = None,
        do_sdata_padding: bool = False) -> None:
    if register_list is None:
        register_list = resources.registers

    vc.comment(f"Loading to registers from buffer {buffer} at offset {offset} and stride {stride}")

    if buffer is not None:
        resources.io_index[:] = offset * params.fft_stride + resources.input_batch_offset
        
        for i in range(len(register_list)):
            if i != 0:
                resources.io_index += stride * params.fft_stride
            
            if params.r2c and params.inverse:
                buffer.read_r2c_inverse(
                    register=register_list[i],
                    memory_index=resources.io_index,
                    fft_index=i * stride + offset,
                    spare_index=resources.io_index_2,
                    input_batch_offset=resources.input_batch_offset,
                    fft_size=params.config.N,
                    fft_stride=params.fft_stride
                )
            else:
                buffer.read(register_list[i], resources.io_index, spare_register=resources.omega_register, r2c=params.r2c)

        return
    
    if resources.sdata_offset is not None:
        resources.io_index[:] = offset + resources.sdata_offset
    else:
        resources.io_index[:] = offset

    for i in range(len(register_list)):
        if do_sdata_padding:
            resources.io_index_2[:] = resources.io_index + stride * i + ((resources.io_index + stride * i) / params.sdata_row_size)
            register_list[i][:] = resources.sdata[resources.io_index_2]
        else:
            register_list[i][:] = resources.sdata[resources.io_index + stride * i]
            
def store_register(
        resources: FFTResources,
        params: FFTParams,
        buffer: Optional[IOProxy],
        offset: Const[u32],
        register: vc.ShaderVariable,
        do_sdata_padding: bool = False) -> None:
    if buffer is None:
        sdata_index = offset

        if resources.sdata_offset is not None:
            sdata_index = sdata_index + resources.sdata_offset
        
        if do_sdata_padding:
            resources.io_index[:] = sdata_index
            resources.io_index[:] = resources.io_index + resources.io_index / params.sdata_row_size
            sdata_index = resources.io_index
        
        resources.sdata[sdata_index] = register
    else:
        if params.normalize and params.inverse:
            register[:] = register / params.config.N

        buffer.write(
            register=register,
            memory_index=resources.io_index,
            r2c=params.r2c,
            inverse=params.inverse,
            fft_size=params.config.N,
            fft_index=offset
        )

def store_registers_from_stages(
        resources: FFTResources,
        params: FFTParams,
        stage: FFTRegisterStageConfig,
        stage_invocations: List[FFTRegisterStageInvocation],
        output: IOProxy,
        stride: int):

    sdata_padding = params.sdata_row_size != params.sdata_row_size_padded and stride < 32 and output is None
    
    if output is not None:
        resources.io_index[:] = resources.tid * params.fft_stride + resources.output_batch_offset

    vc.comment(f"Storing from registers to buffer {output} ")
    
    instance_index_stride = params.config.N // (stage.fft_length * stage.instance_count)

    for jj in range(stage.fft_length):
        for ii, invocation in enumerate(stage_invocations):
            if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                vc.if_statement(resources.tid < params.config.N // stage.registers_used)

            if output is not None and jj != 0 or ii != 0:
                resources.io_index += instance_index_stride * params.fft_stride

            store_register(
                resources=resources,
                params=params,
                buffer=output,
                offset=invocation.sub_sequence_offset + jj * stride,
                register=resources.registers[invocation.register_selection][jj],
                do_sdata_padding=sdata_padding
            )

        if stage.remainder_offset == 1:
            vc.end()

    return sdata_padding