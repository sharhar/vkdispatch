import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Dict

from .config import FFTConfig
from .sdata_manager import FFTSDataManager
from .resources import FFTResources

import dataclasses

@dataclasses.dataclass
class ReadOp:
    first_invocation_instance: bool
    register: vc.ShaderVariable
    offset: vc.ShaderVariable
    fft_index: vc.ShaderVariable
    stride: int

@dataclasses.dataclass
class WriteOp:
    register: vc.ShaderVariable
    fft_index: vc.ShaderVariable

class FFTRegisters:
    resources: FFTResources
    config: FFTConfig
    sdata: FFTSDataManager
    registers: List[vc.ShaderVariable]
    count: int

    def __init__(self, resources: FFTResources, sdata: FFTSDataManager, count: int, name: str):
        self.resources = resources
        self.config = resources.config
        self.sdata = sdata
        
        self.registers = [
            vc.new(vc.c64, 0, var_name=f"{name}_reg_{i}") for i in range(count)
        ]

        self.count = count

    def clear(self):
        for reg in self.registers:
            reg[:] = 0

    def slice(self, slc: slice) -> List[vc.ShaderVariable]:
        return self.registers[slc]
    
    def slice_set(self, slc: slice, values: List[vc.ShaderVariable]):
        self.registers[slc] = values

    def __getitem__(self, index: int) -> vc.ShaderVariable:
        return self.registers[index]
    
    def __setitem__(self, index: int, value: vc.ShaderVariable):
        self.registers[index][:] = value

    def get_input_format(self, stage_index: int = 0) -> Dict[int, int]:
        in_format = {}

        stride = self.config.N // self.config.stages[stage_index].fft_length

        register_count = len(self.registers)
        register_index_list = list(range(register_count))

        for invocation in self.resources.invocations[stage_index]:
            sub_registers = register_index_list[invocation.register_selection]
            
            for i in range(len(sub_registers)):
                in_format[invocation.get_read_index(stride * i)] = sub_registers[i]

        return in_format

    def get_output_format(self, stage_index: int = -1) -> Dict[int, int]:
        out_format = {}

        register_count = len(self.registers)
        register_index_list = list(range(register_count))

        for jj in range(self.config.stages[stage_index].fft_length):
            for invocation in self.resources.invocations[stage_index]:
                out_format[invocation.get_write_index(jj)] = register_index_list[invocation.register_selection][jj]

        return out_format

    def iter_read(self, stage_index: int = 0):
        self.resources.stage_begin(stage_index)

        for ii, invocation in enumerate(self.resources.invocations[stage_index]):
            self.resources.invocation_gaurd(stage_index, ii)

            register_list = self.slice(invocation.register_selection)

            offset = invocation.instance_id
            stride = self.config.N // self.config.stages[stage_index].fft_length

            for i in range(len(register_list)):
                fft_index = i * stride + offset

                read_op = ReadOp(
                    first_invocation_instance=(i == 0),
                    register=register_list[i],
                    offset=offset,
                    fft_index=fft_index,
                    stride=stride
                )

                yield read_op

        self.resources.invocation_end(stage_index)
        self.resources.stage_end(stage_index)

    def iter_write(self, stage_index: int = -1):
        self.resources.stage_begin(stage_index)

        for jj in range(self.config.stages[stage_index].fft_length):
            for ii, invocation in enumerate(self.resources.invocations[stage_index]):
                self.resources.invocation_gaurd(stage_index, ii)

                fft_index = invocation.sub_sequence_offset + jj * self.resources.output_strides[stage_index]

                write_op = WriteOp(
                    register=self.slice(invocation.register_selection)[jj],
                    fft_index=fft_index
                )

                yield write_op

        self.resources.invocation_end(stage_index)
        self.resources.stage_end(stage_index)

    def read_from_sdata(self, stage_index: int = 0):
        self.sdata.op_read()

        for read_op in self.iter_read(stage_index=stage_index):
            if read_op.first_invocation_instance:
                self.resources.io_index[:] = read_op.offset + self.sdata.sdata_offset
            else:
                self.resources.io_index += read_op.stride

            if self.sdata.use_padding:
                self.resources.io_index_2[:] = self.resources.io_index + ((self.resources.io_index) / self.sdata.sdata_row_size)
                read_op.register[:] = self.sdata.sdata[self.resources.io_index_2]
            else:
                read_op.register[:] = self.sdata.sdata[self.resources.io_index]

    def write_to_sdata(self, stage_index: int = -1):
        self.sdata.op_write()

        for write_op in self.iter_write(stage_index=stage_index):
            sdata_index = write_op.fft_index

            if self.sdata.use_padding:
                self.resources.io_index[:] = sdata_index
                self.resources.io_index[:] = self.resources.io_index + self.resources.io_index / self.sdata.sdata_row_size
                sdata_index = self.resources.io_index

            self.sdata.sdata[sdata_index] = write_op.register

    def shuffle(self, output_stage: int = -1, input_stage: int = 0):
        out_format = self.get_output_format(output_stage)
        in_format = self.get_input_format(input_stage)

        if out_format.keys() != in_format.keys():
            self.write_to_sdata(stage_index=output_stage)
            self.read_from_sdata(stage_index=input_stage)
            return

        shuffled_registers = [None] * len(self.registers)

        for i in range(len(self.registers)):
            format_key = None
            
            for k, v in in_format.items():
                if v == i:
                    format_key = k
                    break

            assert format_key is not None, "Could not find register in output format???"

            shuffled_registers[i] = self.registers[out_format[format_key]]

        for i in range(len(self.registers)):
            self.registers[i] = shuffled_registers[i]

    def read_from_registers(self, other: "FFTRegisters") -> "FFTRegisters":
        assert self.count == other.count, "Register counts must match for copy"

        for i in range(self.count):
            self.registers[i][:] = other.registers[i]