import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple, Optional

from .resources import FFTResources
from .config import FFTRegisterStageConfig, FFTParams

def read_mapped_input(resources: FFTResources, params: FFTParams, mapping_index: Const[i32], mapping_function: vd.MappingFunction, output_register: vc.ShaderVariable, index: Const[u32]):
    assert len(mapping_function.register_types) == 1, "Mapping function must have exactly one register type"
    assert mapping_function.register_types[0] == c64, "Mapping function register type does not match expected return type"

    if params.input_sdata:
        output_register[:] = resources.sdata[index + resources.sdata_offset]

    vc.set_mapping_index(mapping_index)
    vc.set_mapping_registers([output_register])

    mapping_function.mapping_function(*params.input_buffers)

def get_global_input(resources: FFTResources, params: FFTParams, buffer: Buff, index: Const[u32], output_register: vc.ShaderVariable):
    resources.io_index[:] = (index * params.fft_stride + resources.input_batch_offset).cast_to(i32)

    if not params.r2c:
        if isinstance(buffer, vd.MappingFunction):
            read_mapped_input(resources, params, resources.io_index, buffer, output_register, index)
        else:
            output_register[:] = buffer[resources.io_index]

        return
    
    if not params.inverse:
        if isinstance(buffer, vd.MappingFunction):
            read_mapped_input(resources, params, resources.io_index, buffer, output_register, index)
        else:
            real_value = buffer[resources.io_index / 2][resources.io_index % 2]
            output_register[:] = f"vec2({real_value}, 0)"

        return
    
    assert not isinstance(buffer, vd.MappingFunction), "Inverse R2C FFT does not support input mapping"
    
    vc.if_statement(index >= (params.config.N // 2) + 1)
    resources.io_index[:] = ((params.config.N - index) * params.fft_stride + resources.input_batch_offset).cast_to(i32)
    output_register[:] = buffer[resources.io_index]
    output_register.y = -output_register.y
    vc.else_statement()
    output_register[:] = buffer[resources.io_index]
    vc.end()
    
def load_buffer_to_registers(resources: FFTResources, params: FFTParams, buffer: Optional[Buff], offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
    if register_list is None:
        register_list = resources.registers

    vc.comment(f"Loading to registers {register_list} from buffer {buffer} at offset {offset} and stride {stride}")

    for i in range(len(register_list)):
        if buffer is None:
            register_list[i][:] = resources.sdata[i * stride + offset + resources.sdata_offset]
        else:
            get_global_input(resources, params, buffer, i * stride + offset, register_list[i])

def write_mapped_output(params: FFTParams, mapping_index: Const[i32], mapping_function: vd.MappingFunction, output_register: vc.ShaderVariable):
    assert len(mapping_function.register_types) == 1, "Mapping function must have exactly one register type"
    assert mapping_function.register_types[0] == c64, "Mapping function register type does not match expected return type"

    vc.set_mapping_index(mapping_index)
    vc.set_mapping_registers([output_register])

    mapping_function.mapping_function(*params.output_buffers)

def set_global_output(resources: FFTResources, params: FFTParams, buffer: Buff, index: Const[u32], value: Const[c64]):
    true_value = value

    if params.inverse and params.normalize:
        true_value[:] = true_value / params.config.N

    resources.io_index[:] = (index * params.fft_stride + resources.output_batch_offset).cast_to(i32)

    if not params.r2c:
        if isinstance(buffer, vd.MappingFunction):
            write_mapped_output(params, resources.io_index, buffer, true_value)
        else:
            buffer[resources.io_index] = true_value

        return

    if not params.inverse:
        vc.if_statement(index < (params.config.N // 2) + 1)
        
        if isinstance(buffer, vd.MappingFunction):
            write_mapped_output(params, resources.io_index, buffer, true_value)
        else:
            buffer[index * params.fft_stride + resources.output_batch_offset] = true_value

        vc.end()

        return

    if isinstance(buffer, vd.MappingFunction):
        write_mapped_output(params, resources.io_index, buffer, true_value)
    else:
        buffer[resources.io_index / 2][resources.io_index % 2] = true_value.x

def store_registers_in_buffer(resources: FFTResources, params: FFTParams, buffer: Optional[Buff], offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
    if register_list is None:
        register_list = resources.registers

    vc.comment(f"Storing registers {register_list} to buffer {buffer} at offset {offset} and stride {stride}")

    for i in range(len(register_list)):
        if buffer is None:
            resources.sdata[i * stride + offset + resources.sdata_offset] = register_list[i]
        else:
            set_global_output(resources, params, buffer, i * stride + offset, register_list[i])
