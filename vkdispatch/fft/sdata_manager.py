import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Literal, Union, List

from .config import FFTConfig
from .grid_manager import FFTGridManager
from .resources import FFTResources

class FFTSDataManager:
    sdata: vc.Buff[vc.c64]
    sdata_offset: Union[vc.Const[vc.u32], Literal[0]]

    sdata_row_size: int
    sdata_row_size_padded: int
    padding_enabled: bool

    use_padding: bool

    tid: vc.ShaderVariable
    fft_N: int

    def __init__(self, config: FFTConfig, grid: FFTGridManager):
        self.sdata_row_size = config.sdata_row_size
        self.sdata_row_size_padded = config.sdata_row_size_padded
        self.padding_enabled = self.sdata_row_size != self.sdata_row_size_padded
        self.use_padding = False
        self.fft_N = config.N
        self.tid = grid.tid

        total_inner_batches = grid.inline_batches_inner * grid.inline_batches_outer

        self.sdata = vc.shared_buffer(
            vd.complex64,
            config.sdata_allocation * total_inner_batches,
            var_name="sdata")
        
        self.sdata_offset = 0

        if total_inner_batches > 1:
            sdata_offset_value = grid.local_outer * grid.inline_batches_inner * config.N

            if grid.local_inner is not None:
                sdata_offset_value = sdata_offset_value + grid.local_inner * config.N

            self.sdata_offset = vc.new_uint(sdata_offset_value, var_name="sdata_offset")
    
    def read_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            stage_index: int = 0,
                            invocation_index: int = None,
                            registers: List[vc.ShaderVariable] = None):
        
        if invocation_index is None:
            resources.stage_begin(stage_index)

            for ii, invocation in enumerate(resources.invocations[stage_index]):
                resources.invocation_gaurd(stage_index, ii)

                register_selection = None

                if registers is not None:
                    register_selection = registers[invocation.register_selection]

                self.read_registers(resources, config, stage_index, ii, register_selection)

            resources.invocation_end(stage_index)
            resources.stage_end(stage_index)

            return

        vc.comment(f"Loading from shared data buffer to registers")

        invocation = resources.invocations[stage_index][invocation_index]
        
        if registers is None:
            registers = resources.registers[invocation.register_selection]

        resources.io_index[:] = invocation.instance_id + self.sdata_offset

        stride = self.fft_N // config.stages[stage_index].fft_length

        for i in range(len(registers)):
            if self.use_padding:
                resources.io_index_2[:] = resources.io_index + stride * i + ((resources.io_index + stride * i) / self.sdata_row_size)
                registers[i][:] = self.sdata[resources.io_index_2]
            else:
                registers[i][:] = self.sdata[resources.io_index + stride * i]

    def write_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            stage_index: int,
                            registers: List[vc.ShaderVariable] = None):
        stage = config.stages[stage_index]

        if registers is None:
            registers = resources.registers

        self.use_padding = self.padding_enabled and resources.output_strides[stage_index] < 32

        vc.comment(f"Storing from registers to shared data buffer with fft length {stage.fft_length} and invocations {len(resources.invocations[stage_index])}")

        resources.stage_begin(stage_index)

        for jj in range(stage.fft_length):
            for ii, invocation in enumerate(resources.invocations[stage_index]):
                resources.invocation_gaurd(stage_index, ii)

                sdata_index = self.sdata_offset + invocation.sub_sequence_offset + jj * resources.output_strides[stage_index]
                
                if self.use_padding:
                    resources.io_index[:] = sdata_index
                    resources.io_index[:] = resources.io_index + resources.io_index / self.sdata_row_size
                    sdata_index = resources.io_index
                
                self.sdata[sdata_index] = registers[invocation.register_selection][jj]

            resources.invocation_end(stage_index)
        
        resources.stage_end(stage_index)
