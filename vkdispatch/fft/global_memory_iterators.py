import vkdispatch.codegen as vc

from typing import Optional, Tuple

import numpy as np
import dataclasses

from .registers import FFTRegisters
from .memory_iterators import memory_reads_iterator, memory_writes_iterator, MemoryOp

def global_batch_offset(
        registers: FFTRegisters,
        r2c: bool = False,
        is_output: bool = None,
        inverse: bool = None):
    config = registers.config
    grid = registers.resources.grid

    outer_batch_stride = config.N * config.fft_stride

    if r2c:
        assert inverse is not None, "Must specify inverse for r2c io"
        assert is_output is not None, "Must specify is_output for r2c io"
        assert config.fft_stride == 1, "R2C io only supported for contiguous data"

        outer_batch_stride = (config.N // 2) + 1

        # for inverse-r2c write and forward-r2c read, the
        # outer batch stride is doubled, since we are writting
        # floats and not vec2s
        if inverse == is_output:
            outer_batch_stride *= 2

    return grid.global_outer * outer_batch_stride + grid.global_inner

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

    def write_to_buffer(self,
                        buffer: vc.Buff[vc.c64],
                        register: Optional[vc.ShaderVariable] = None,
                        io_index: Optional[vc.ShaderVariable] = None):
        if register is None:
            register = self.register

        if io_index is None:
            io_index = self.io_index

        if not self.r2c:
            buffer[io_index] = register
            return

        if not self.inverse:
            vc.if_statement(self.fft_index < (self.fft_size // 2) + 1)
            buffer[io_index] = register
            vc.end()
            return

        buffer[io_index // 2][io_index % 2] = register.real

def global_writes_iterator(
        registers: FFTRegisters,
        r2c: bool = False,
        inverse: bool = None):

    vc.comment(f"Writing registers to global memory")

    resources = registers.resources
    config = registers.config
    
    resources.output_batch_offset[:] = global_batch_offset(registers, r2c=r2c, is_output=True, inverse=inverse)

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
        register[:] = vc.to_complex(0)
        vc.end()

    def read_from_buffer(self,
                         buffer: vc.Buff[vc.c64],
                         register: Optional[vc.ShaderVariable] = None,
                         io_index: Optional[vc.ShaderVariable] = None):
                        # buffer: vc.Buff[vc.c64], register: Optional[vc.ShaderVariable] = None):
        self.check_in_signal_range()

        if io_index is None:
            io_index = self.io_index

        if register is None:
            register = self.register

        if not self.r2c:
            register[:] = buffer[io_index]
            self.signal_range_end(register)
            return

        if not self.inverse:
            real_value = buffer[io_index // 2][io_index % 2]
            register[:] = vc.to_complex(real_value)
            self.signal_range_end(register)
            return

        vc.if_statement(self.fft_index >= (self.fft_size // 2) + 1)
        self.io_index_2[:] = self.r2c_inverse_offset - io_index
        register[:] = buffer[self.io_index_2]
        register.imag = -register.imag
        vc.else_statement()
        register[:] = buffer[io_index]
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

    if r2c:
        assert not format_transposed, "R2C transposed format not supported"

    resources = registers.resources
    config = registers.config
    
    if format_transposed:
        work_index = vc.workgroup_id().z * vc.num_workgroups().x * vc.num_workgroups().y + \
                     vc.workgroup_id().y * vc.num_workgroups().x + vc.workgroup_id().x

        resources.input_batch_offset[:] = vc.local_invocation_index() + \
                                            work_index * (vc.workgroup_size().x * vc.workgroup_size().y * vc.workgroup_size().z)
        r2c_inverse_offset = None # Transposed r2c not supported anyways
        transpose_stride = np.prod(resources.grid.workgroup_count) * np.prod(resources.grid.local_size)
    else:
        resources.input_batch_offset[:] = global_batch_offset(registers, r2c=r2c, is_output=False, inverse=inverse)
        r2c_inverse_offset = 2 * resources.input_batch_offset + config.N * config.fft_stride

    for read_op in memory_reads_iterator(resources, 0):
        if format_transposed:
            resources.io_index[:] = resources.input_batch_offset + read_op.register_id * transpose_stride
        else:
            resources.io_index[:] = resources.input_batch_offset + read_op.fft_index * config.fft_stride

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

    def write_to_buffer(self,
                        buffer: vc.Buff[vc.c64],
                        register: Optional[vc.ShaderVariable] = None,
                        io_index: Optional[vc.ShaderVariable] = None):
        if io_index is None:
            io_index = self.io_index

        if register is None:
            register = self.register

        buffer[io_index] = register

def global_trasposed_write_iterator(registers: FFTRegisters):
    vc.comment(f"Writing registers to global memory in transposed format")

    resources = registers.resources
    
    work_index = vc.workgroup_id().z * vc.num_workgroups().x * vc.num_workgroups().y + \
                    vc.workgroup_id().y * vc.num_workgroups().x + vc.workgroup_id().x

    resources.input_batch_offset[:] = vc.local_invocation_index() + \
                                     work_index * (vc.workgroup_size().x * vc.workgroup_size().y * vc.workgroup_size().z)
    transpose_stride = np.prod(resources.grid.workgroup_count) * np.prod(resources.grid.local_size)

    for read_op in memory_reads_iterator(resources, 0): # Iterate in read order to match register format when reading
        resources.io_index[:] = resources.input_batch_offset + read_op.register_id * transpose_stride

        global_trasposed_write_op = GlobalTransposedWriteOp.from_memory_op(
            base=read_op,
            register=registers[read_op.register_id],
            io_index=resources.io_index
        )

        yield global_trasposed_write_op