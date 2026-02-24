import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Literal, Union, List, Optional

from .config import FFTConfig
from .grid_manager import FFTGridManager
from .resources import FFTResources
from .registers import FFTRegisters

from .memory_iterators import memory_reads_iterator, memory_writes_iterator

class FFTSDataManager:
    sdata: vc.Buffer
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

    resources: FFTResources
    default_registers: FFTRegisters


    def __init__(self, config: FFTConfig, grid: FFTGridManager, default_registers: FFTRegisters):
        self.sdata_row_size = config.sdata_row_size
        self.sdata_row_size_padded = config.sdata_row_size_padded
        self.padding_enabled = self.sdata_row_size != self.sdata_row_size_padded
        self.use_padding = False
        self.fft_N = config.N
        self.tid = grid.tid
        self.last_op = None
        self.default_registers = default_registers
        self.resources = default_registers.resources

        total_inner_batches = grid.inline_batches_inner * grid.inline_batches_outer

        self.sdata = vc.shared_buffer(
            config.compute_type,
            config.sdata_allocation * total_inner_batches,
            var_name="sdata")
        
        self.sdata_offset = 0

        if total_inner_batches > 1:
            sdata_offset_value = grid.local_outer * grid.inline_batches_inner * config.N

            if grid.local_inner is not None:
                sdata_offset_value = sdata_offset_value + grid.local_inner * config.N

            self.sdata_offset = vc.new_uint_register(sdata_offset_value, var_name="sdata_offset")
    

    def do_op(self, op: bool):
        if self.last_op is not None and self.last_op != op:
            vc.barrier()

        self.last_op = op

    def op_read(self) -> bool:
        self.do_op(False)

    def op_write(self) -> bool:
        self.do_op(True)

    def read_from_sdata(self, registers: Optional[FFTRegisters] = None, stage_index: int = 0):
        self.op_read()

        if registers is None:
            registers = self.default_registers

        for read_op in memory_reads_iterator(self.resources, stage_index):
            self.resources.io_index[:] = read_op.fft_index + self.sdata_offset

            if self.use_padding:
                self.resources.io_index[:] = self.resources.io_index + (self.resources.io_index // self.sdata_row_size)
            
            registers[read_op.register_id] = self.sdata[self.resources.io_index]

    def write_to_sdata(self, registers: Optional[FFTRegisters] = None, stage_index: int = -1):
        self.op_write()

        self.use_padding = self.padding_enabled and self.resources.output_strides[stage_index] < 32

        if registers is None:
            registers = self.default_registers

        for write_op in memory_writes_iterator(self.resources, stage_index):
            self.resources.io_index[:] = write_op.fft_index + self.sdata_offset

            if self.use_padding:
                self.resources.io_index[:] = self.resources.io_index + (self.resources.io_index // self.sdata_row_size)

            self.sdata[self.resources.io_index] = registers[write_op.register_id]
