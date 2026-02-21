import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Dict

from .config import FFTConfig
from .resources import FFTResources

import dataclasses

@dataclasses.dataclass
class RegisterIOOp:
    register: vc.ShaderVariable
    offset: vc.ShaderVariable
    stride: int
    fft_index: vc.ShaderVariable
    register_id: int
    register_count: int
    element_id: int
    element_count: int
    instance_id: int
    instance_count: int

class FFTRegisters:
    resources: FFTResources
    config: FFTConfig
    registers: List[vc.ShaderVariable]
    count: int

    def __init__(self, resources: FFTResources, count: int, name: str):
        self.resources = resources
        self.config = resources.config
        
        self.registers = [
            vc.new_complex_register(var_name=f"{name}_reg_{i}") for i in range(count)
        ]

        self.count = count

    def clear(self):
        for reg in self.registers:
            reg[:] = 0

    def register_slice(self, slc: slice) -> List[vc.ShaderVariable]:
        return self.registers[slc]
    def slice_set(self, slc: slice, values: List[vc.ShaderVariable]):
        self.registers[slc] = values

    def __getitem__(self, index: int) -> vc.ShaderVariable:
        return self.registers[index]
    
    def __setitem__(self, index: int, value: vc.ShaderVariable):
        self.registers[index][:] = value

    def normalize(self):
        for i in range(self.count):
            self.registers[i][:] = self.registers[i] / self.config.N

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

    def try_shuffle(self, output_stage: int = -1, input_stage: int = 0) -> bool:
        out_format = self.get_output_format(output_stage)
        in_format = self.get_input_format(input_stage)

        if out_format.keys() != in_format.keys():
            return False

        # Some stages can use fewer registers than config.register_count.
        # Shuffle only registers that appear in the input format.
        shuffled_registers = list(self.registers)

        for format_key, input_register in in_format.items():
            shuffled_registers[input_register] = self.registers[out_format[format_key]]

        for i in range(len(self.registers)):
            self.registers[i] = shuffled_registers[i]
        
        return True

    def read_from_registers(self, other: "FFTRegisters") -> "FFTRegisters":
        assert self.count == other.count, "Register counts must match for copy"

        for i in range(self.count):
            self.registers[i][:] = other.registers[i]
