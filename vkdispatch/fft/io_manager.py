import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional, Tuple

from .io_proxy import IOProxy
from .registers import FFTRegisters
from .global_memory_iterators import global_writes_iterator, global_reads_iterator
from .global_memory_iterators import GlobalWriteOp, GlobalReadOp

__static_global_write_op = None
__static_global_read_op = None

def set_global_write_op(op: GlobalWriteOp):
    global __static_global_write_op
    __static_global_write_op = op

def mapped_write_op() -> GlobalWriteOp:
    return __static_global_write_op

def set_global_read_op(op: GlobalReadOp):
    global __static_global_read_op
    __static_global_read_op = op

def mapped_read_op() -> GlobalReadOp:
    return __static_global_read_op

class IOManager:
    default_registers: FFTRegisters
    output_proxy: IOProxy
    input_proxy: IOProxy
    kernel_proxy: IOProxy

    def __init__(self,
                    default_registers: FFTRegisters,
                    shader_context: vd.ShaderContext,
                    output_map: Optional[vd.MappingFunction],
                    input_map: Optional[vd.MappingFunction] = None,
                    kernel_map: Optional[vd.MappingFunction] = None):
            self.default_registers = default_registers
            self.output_proxy = IOProxy(vd.complex64 if output_map is None else output_map, "Output")
            self.input_proxy = IOProxy(input_map, "Input")
            self.kernel_proxy = IOProxy(kernel_map, "Kernel")
    
            output_types = self.output_proxy.buffer_types
            input_types = self.input_proxy.buffer_types
            kernel_types = self.kernel_proxy.buffer_types
    
            all_types = output_types + input_types + kernel_types
    
            if len(all_types) == 0:
                raise ValueError("A big error happened")
    
            sig_vars = shader_context.declare_input_arguments(all_types)
    
            output_count = len(output_types)
            input_count = len(input_types)
    
            self.output_proxy.set_variables(sig_vars[:output_count])
            self.input_proxy.set_variables(sig_vars[output_count:input_count + output_count])
            self.kernel_proxy.set_variables(sig_vars[input_count + output_count:])

            if input_count == 0:
                self.input_proxy = self.output_proxy

    def read_from_proxy(self,
                        proxy: IOProxy,
                        registers: Optional[FFTRegisters] = None,
                        r2c: bool = False,
                        inverse: bool = None,
                        format_transposed: bool = False,
                        signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None):

        if registers is None:
            registers = self.default_registers
        
        for read_op in global_reads_iterator(
                registers=registers,
                r2c=r2c,
                inverse=inverse,
                format_transposed=format_transposed,
                signal_range=signal_range
            ):
            
            if proxy.has_callback():
                set_global_read_op(read_op)
                proxy.do_callback()
                set_global_read_op(None)
            else:
                read_op.read_from_buffer(proxy.buffer_variables[0])

    def write_to_proxy(self,
                        proxy: IOProxy,
                        registers: Optional[FFTRegisters] = None,
                        r2c: bool = False,
                        inverse: bool = None):
        
        if registers is None:
            registers = self.default_registers
        
        for write_op in global_writes_iterator(
                registers=registers,
                r2c=r2c,
                inverse=inverse
            ):
            
            if proxy.has_callback():
                set_global_write_op(write_op)
                proxy.do_callback()
                set_global_write_op(None)
            else:
                write_op.write_to_buffer(proxy.buffer_variables[0])
    
    def read_input(self,
                   registers: Optional[FFTRegisters] = None,
                   r2c: bool = False,
                   inverse: bool = None,
                   signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None):
        self.read_from_proxy(
            self.input_proxy,
            registers,
            r2c=r2c,
            inverse=inverse,
            signal_range=signal_range
        )

    def write_output(self,
                     registers: Optional[FFTRegisters] = None,
                     r2c: bool = False,
                     inverse: bool = None):
        self.write_to_proxy(
            self.output_proxy,
            registers,
            r2c=r2c,
            inverse=inverse
        )
    
    def read_kernel(self, registers: Optional[FFTRegisters] = None, format_transposed: bool = False):
        self.read_from_proxy(
            self.kernel_proxy,
            registers,
            format_transposed=format_transposed
        )