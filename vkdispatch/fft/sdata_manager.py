import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Literal, Union, List

from .config import FFTConfig
from .grid_manager import FFTGridManager
#from .resources import FFTResources
#from .registers import FFTRegisters

class FFTSDataManager:
    sdata: vc.Buff[vc.c64]
    sdata_offset: Union[vc.Const[vc.u32], Literal[0]]

    sdata_row_size: int
    sdata_row_size_padded: int
    padding_enabled: bool

    # None: not set yet
    # True: last operation was write
    # False: last operation was read
    last_op: bool

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
        self.last_op = None

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
    

    def do_op(self, op: bool):
        if self.last_op is not None and self.last_op != op:
            vc.barrier()

        self.last_op = op

    def op_read(self) -> bool:
        self.do_op(False)

    def op_write(self) -> bool:
        self.do_op(True)

    # def read_registers(self,
    #                         registers: FFTRegisters,
    #                         resources: FFTResources,
    #                         config: FFTConfig,
    #                         stage_index: int = 0):
        
    #     self.op_read()

    #     for read_op in registers.iter_read(stage_index=stage_index):
    #         if read_op.first_invocation_instance:
    #             resources.io_index[:] = read_op.offset + self.sdata_offset
    #         else:
    #             resources.io_index += read_op.stride

    #         if self.use_padding:
    #             resources.io_index_2[:] = resources.io_index + ((resources.io_index) / self.sdata_row_size)
    #             read_op.register[:] = self.sdata[resources.io_index_2]
    #         else:
    #             read_op.register[:] = self.sdata[resources.io_index]

        # resources.stage_begin(stage_index)

        # for invocation_index, invocation in enumerate(resources.invocations[stage_index]):
        #     resources.invocation_gaurd(stage_index, invocation_index)

        #     register_selection = registers.slice(invocation.register_selection)

        #     resources.io_index[:] = invocation.instance_id + self.sdata_offset

        #     stride = self.fft_N // config.stages[stage_index].fft_length

        #     for i in range(len(register_selection)):
        #         if self.use_padding:
        #             resources.io_index_2[:] = resources.io_index + stride * i + ((resources.io_index + stride * i) / self.sdata_row_size)
        #             register_selection[i][:] = self.sdata[resources.io_index_2]
        #         else:
        #             register_selection[i][:] = self.sdata[resources.io_index + stride * i]

        # resources.invocation_end(stage_index)
        # resources.stage_end(stage_index)
        

    # def write_registers(self,
    #                         registers: FFTRegisters,
    #                         resources: FFTResources,
    #                         config: FFTConfig,
    #                         stage_index: int):
    #     stage = config.stages[stage_index]

    #     self.use_padding = self.padding_enabled and resources.output_strides[stage_index] < 32

    #     vc.comment(f"Storing from registers to shared data buffer with fft length {stage.fft_length} and invocations {len(resources.invocations[stage_index])}")

    #     self.op_write()

    #     for write_op in registers.iter_write(stage_index=stage_index):
    #         sdata_index = write_op.fft_index

    #         if self.use_padding:
    #             resources.io_index[:] = sdata_index
    #             resources.io_index[:] = resources.io_index + resources.io_index / self.sdata_row_size
    #             sdata_index = resources.io_index

    #         self.sdata[sdata_index] = write_op.register

        # resources.stage_begin(stage_index)

        # for jj in range(stage.fft_length):
        #     for ii, invocation in enumerate(resources.invocations[stage_index]):
        #         resources.invocation_gaurd(stage_index, ii)

        #         sdata_index = self.sdata_offset + invocation.sub_sequence_offset + jj * resources.output_strides[stage_index]
                
        #         if self.use_padding:
        #             resources.io_index[:] = sdata_index
        #             resources.io_index[:] = resources.io_index + resources.io_index / self.sdata_row_size
        #             sdata_index = resources.io_index

        #         self.sdata[sdata_index] = registers.slice(invocation.register_selection)[jj]

        #     resources.invocation_end(stage_index)
        
        # resources.stage_end(stage_index)
