import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional, Tuple

import numpy as np

import dataclasses

from .registers import FFTRegisters
from .resources import FFTResources
from .memory_iterators import memory_reads_iterator, memory_writes_iterator, MemoryOp

def transpose_io_index(resources: FFTResources):
    local_index = vc.local_invocation().z * vc.workgroup_size().x * vc.workgroup_size().y + \
                vc.local_invocation().y * vc.workgroup_size().x + vc.local_invocation().x

    transposed_local_index = local_index + vc.workgroup().x * (vc.workgroup_size().x * vc.workgroup_size().y * vc.workgroup_size().z)

    transpose_stride = np.prod(resources.grid.workgroup_count) * np.prod(resources.grid.local_size)

    transposed_batch = resources.io_index / transpose_stride

    transposed_index = transposed_local_index + transposed_batch * transpose_stride

    resources.io_index[:] = transposed_index

@dataclasses.dataclass
class GlobalWriteOp(MemoryOp):
    register: vc.ShaderVariable
    io_index: vc.ShaderVariable
    r2c: bool
    inverse: Optional[bool]

    @classmethod
    def from_memory_op(cls,
                       base: MemoryOp,
                       register: vc.ShaderVariable,
                       io_index: vc.ShaderVariable,
                       r2c: bool,
                       inverse: Optional[bool] = None) -> 'GlobalWriteOp':
        return cls(**vars(base),
                   register=register,
                   io_index=io_index,
                   r2c=r2c,
                   inverse=inverse)

    def write_to_buffer(self, buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        if register is None:
            register = self.register

        if not self.r2c:
            buffer[self.io_index] = register
            return

        if not self.inverse:
            vc.if_statement(self.fft_index < (self.fft_size // 2) + 1)
            buffer[self.io_index] = register
            vc.end()
            return

        buffer[self.io_index / 2][self.io_index % 2] = register.x

def global_writes_iterator(
        registers: FFTRegisters,
        r2c: bool = False,
        inverse: bool = None):

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

    for write_op in memory_writes_iterator(resources, -1):
        resources.io_index[:] = resources.output_batch_offset + write_op.fft_index * config.fft_stride

        global_write_op = GlobalWriteOp.from_memory_op(
            base=write_op,
            register=registers[write_op.register_id],
            io_index=resources.io_index,
            r2c=r2c,
            inverse=inverse
        )

        yield global_write_op

@dataclasses.dataclass
class GlobalReadOp(MemoryOp):
    register: vc.ShaderVariable
    io_index: vc.ShaderVariable
    io_index_2: vc.ShaderVariable
    r2c: bool
    inverse: Optional[bool]
    r2c_inverse_offset: vc.ShaderVariable
    format_transposed: bool
    signal_range: Tuple[int, int]

    @classmethod
    def from_memory_op(cls,
                       base: MemoryOp,
                       register: vc.ShaderVariable,
                       io_index: vc.ShaderVariable,
                       io_index_2: vc.ShaderVariable,
                       r2c: bool,
                       inverse: Optional[bool],
                       r2c_inverse_offset: vc.ShaderVariable,
                       format_transposed: bool,
                       signal_range: Tuple[int, int]) -> 'GlobalReadOp':
        return cls(**vars(base),
                   register=register,
                   io_index=io_index,
                   io_index_2=io_index_2,
                   r2c=r2c,
                   inverse=inverse,
                   r2c_inverse_offset=r2c_inverse_offset,
                   format_transposed=format_transposed,
                   signal_range=signal_range
                )

    def check_in_signal_range(self) -> bool:
        if self.signal_range == (0, self.fft_size):
            return
        
        if self.signal_range[0] == 0:
            vc.if_statement(self.fft_index < self.signal_range[1])
            return
        
        if self.signal_range[1] == self.fft_size:
            vc.if_statement(self.fft_index >= self.signal_range[0])
            return

        vc.if_all(self.fft_index >= self.signal_range[0], self.fft_index < self.signal_range[1])
        
    def signal_range_end(self, register: vc.ShaderVariable):
        if self.signal_range == (0, self.fft_size):
            return

        vc.else_statement()
        register[:] = "vec2(0)"
        vc.end()

    def read_from_buffer(self, buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        self.check_in_signal_range()

        if register is None:
            register = self.register

        if not self.r2c:
            register[:] = buffer[self.io_index]
            return

        if not self.inverse:
            real_value = buffer[self.io_index / 2][self.io_index % 2]
            register[:] = f"vec2({real_value}, 0)"
            return

        vc.if_statement(self.fft_index >= (self.fft_size // 2) + 1)
        self.io_index_2[:] = self.r2c_inverse_offset - self.io_index
        register[:] = buffer[self.io_index_2]
        register.y = -register.y
        vc.else_statement()
        register[:] = buffer[self.io_index]
        vc.end()

        self.signal_range_end(register)

def resolve_signal_range(
        signal_range: Optional[Tuple[Optional[int], Optional[int]]],
        N: int) -> Tuple[int, int]:
    if signal_range is None:
        return 0, N

    start, end = signal_range

    if start is None:
        start = 0
    if end is None:
        end = N

    return start, end

def global_reads_iterator(
        registers: FFTRegisters,
        r2c: bool = False,
        inverse: bool = None,
        format_transposed: bool = False,
        signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None):
    
    signal_range = resolve_signal_range(signal_range, registers.config.N)

    vc.comment(f"Reading registers from global memory")

    input_batch_stride_y = registers.config.batch_outer_stride

    if r2c:
        assert not format_transposed, "R2C transposed format not supported"
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

    for read_op in memory_reads_iterator(resources, 0):
        resources.io_index[:] = resources.input_batch_offset + read_op.fft_index * config.fft_stride

        if format_transposed:
            transpose_io_index(resources)

        global_read_op = GlobalReadOp.from_memory_op(
            base=read_op,
            register=registers[read_op.register_id],
            io_index=resources.io_index,
            io_index_2=resources.io_index_2,
            r2c=r2c,
            inverse=inverse,
            r2c_inverse_offset=r2c_inverse_offset,
            format_transposed=format_transposed,
            signal_range=signal_range
        )

        yield global_read_op


@dataclasses.dataclass
class GlobalTransposedWriteOp(MemoryOp):
    register: vc.ShaderVariable
    io_index: vc.ShaderVariable

    @classmethod
    def from_memory_op(cls,
                       base: MemoryOp,
                       register: vc.ShaderVariable,
                       io_index: vc.ShaderVariable) -> 'GlobalTransposedWriteOp':
        return cls(**vars(base),
                   register=register,
                   io_index=io_index
                )

    def write_to_buffer(self, buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        if register is None:
            register = self.register

        buffer[self.io_index] = register

def global_trasposed_write_iterator(registers: FFTRegisters):
    vc.comment(f"Writing registers to global memory in transposed format")

    resources = registers.resources
    
    resources = registers.resources
    config = registers.config
    grid = registers.resources.grid

    input_batch_stride_y = registers.config.batch_outer_stride
    
    resources.input_batch_offset[:] = grid.global_outer * input_batch_stride_y + grid.global_inner * config.batch_inner_stride

    for read_op in memory_reads_iterator(resources, 0): # Iterate in read order to match register format when reading
        resources.io_index[:] = resources.input_batch_offset + read_op.fft_index * config.fft_stride

        transpose_io_index(resources)

        global_trasposed_write_op = GlobalTransposedWriteOp.from_memory_op(
            base=read_op,
            register=registers[read_op.register_id],
            io_index=resources.io_index
        )

        yield global_trasposed_write_op