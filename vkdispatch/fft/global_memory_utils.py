import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional

import dataclasses

from .registers import FFTRegisters
from .memory_iterators import memory_reads_iterator, memory_writes_iterator, MemoryOp

@dataclasses.dataclass
class GlobalWriteOp:
    memory_op: MemoryOp
    register: vc.ShaderVariable
    io_index: vc.ShaderVariable
    r2c: bool
    inverse: Optional[bool]

    def write_to_buffer(self, buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        if register is None:
            register = self.register

        if not self.r2c:
            buffer[self.io_index] = register
            return

        if not self.inverse:
            vc.if_statement(self.memory_op.fft_index < (self.memory_op.fft_size // 2) + 1)
            buffer[self.io_index] = register
            vc.end()
            return

        buffer[self.io_index / 2][self.io_index % 2] = register.x

def global_writes_iterator(
        registers: FFTRegisters,
        r2c: bool = False,
        inverse: bool = None,
        stage_index: int = -1):
    
    if r2c:
        assert inverse is not None, "Must specify inverse for r2c write"

    vc.comment(f"Writing registers to global memory")

    resources = registers.resources
    config = registers.config
    grid = registers.resources.grid
    
    output_batch_stride_y = config.batch_outer_stride

    if r2c:
        assert inverse is not None, "Must specify inverse for r2c write"

        if not inverse:
            output_batch_stride_y = (config.N // 2) + 1
        if inverse:
            output_batch_stride_y = ((config.N // 2) + 1) * 2

    resources.output_batch_offset[:] = grid.global_outer * output_batch_stride_y + \
                                        grid.global_inner * config.batch_inner_stride

    for write_op in memory_writes_iterator(resources, stage_index):
        resources.io_index[:] = resources.output_batch_offset + write_op.fft_index * config.fft_stride

        global_write_op = GlobalWriteOp(
            memory_op=write_op,
            register=registers[write_op.register_id],
            io_index=resources.io_index,
            r2c=r2c,
            inverse=inverse
        )

        yield global_write_op

@dataclasses.dataclass
class GlobalReadOp:
    memory_op: MemoryOp
    register: vc.ShaderVariable
    io_index: vc.ShaderVariable
    io_index_2: vc.ShaderVariable
    r2c: bool
    inverse: Optional[bool]
    r2c_inverse_offset: vc.ShaderVariable

    def read_from_buffer(self, buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        if register is None:
            register = self.register

        if not self.r2c:
            register[:] = buffer[self.io_index]
            return

        if not self.inverse:
            real_value = buffer[self.io_index / 2][self.io_index % 2]
            register[:] = f"vec2({real_value}, 0)"
            return

        vc.if_statement(self.memory_op.fft_index >= (self.memory_op.fft_size // 2) + 1)
        self.io_index_2[:] = self.r2c_inverse_offset - self.io_index
        register[:] = buffer[self.io_index_2]
        register.y = -register.y
        vc.else_statement()
        register[:] = buffer[self.io_index]
        vc.end()

def global_reads_iterator(
        registers: FFTRegisters,
        r2c: bool = False,
        inverse: bool = None,
        stage_index: int = 0):
    
    if r2c:
        assert inverse is not None, "Must specify inverse for r2c read"

    vc.comment(f"Reading registers from global memory")

    input_batch_stride_y = registers.config.batch_outer_stride

    if r2c:
        assert inverse is not None, "Must specify inverse for r2c read"

        if not inverse:
            input_batch_stride_y = ((registers.config.N // 2) + 1) * 2
        if inverse:
            input_batch_stride_y = (registers.config.N // 2) + 1

    resources = registers.resources
    config = registers.config
    grid = registers.resources.grid
    
    resources.input_batch_offset[:] = grid.global_outer * input_batch_stride_y + grid.global_inner * config.batch_inner_stride
    r2c_inverse_offset = 2 * resources.input_batch_offset + \
                                config.N * config.fft_stride

    for read_op in memory_reads_iterator(resources, stage_index):
            resources.io_index[:] = resources.input_batch_offset + read_op.fft_index * config.fft_stride

            global_read_op = GlobalReadOp(
                memory_op=read_op,
                register=registers[read_op.register_id],
                io_index=resources.io_index,
                io_index_2=resources.io_index_2,
                r2c=r2c,
                inverse=inverse,
                r2c_inverse_offset=r2c_inverse_offset
            )

            yield global_read_op