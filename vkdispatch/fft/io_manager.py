import vkdispatch as vd
import vkdispatch.codegen as vc
import vkdispatch.base.dtype as dtypes

from typing import Optional, Tuple

import threading

from .io_proxy import IOProxy
from .registers import FFTRegisters
from .global_memory_iterators import global_writes_iterator, global_reads_iterator
from .global_memory_iterators import GlobalWriteOp, GlobalReadOp

_write_op = threading.local()
_read_op = threading.local()

def _get_write_op() -> Optional[GlobalWriteOp]:
    return getattr(_write_op, 'op', None)

def _get_read_op() -> Optional[GlobalReadOp]:
    return getattr(_read_op, 'op', None)

def write_op() -> GlobalWriteOp:
    op = _get_write_op()
    assert op is not None, "No global write operation is set for the current thread!"
    return op

def read_op() -> GlobalReadOp:
    op = _get_read_op()
    assert op is not None, "No global read operation is set for the current thread!"
    return op

def set_write_op(op: GlobalWriteOp):
    if op is None:
        _write_op.op = None
        return

    assert _get_write_op() is None, "A global write operation is already set for the current thread!"
    _write_op.op = op

def set_read_op(op: GlobalReadOp):
    if op is None:
        _read_op.op = None
        return

    assert _get_read_op() is None, "A global read operation is already set for the current thread!"
    _read_op.op = op

class IOManager:
    default_registers: FFTRegisters
    output_proxy: IOProxy
    input_proxy: IOProxy
    kernel_proxy: IOProxy

    def __init__(self,
                    default_registers: FFTRegisters,
                    shader_context: vc.ShaderContext,
                    output_map: Optional[vd.MappingFunction],
                    output_type: dtypes.dtype = vd.complex64,
                    input_type: Optional[dtypes.dtype] = None,
                    input_map: Optional[vd.MappingFunction] = None,
                    kernel_map: Optional[vd.MappingFunction] = None):
            self.default_registers = default_registers
            self.output_proxy = IOProxy(output_type if output_map is None else output_map, "Output")

            if input_map is not None:
                self.input_proxy = IOProxy(input_map, "Input")
            elif output_map is not None:
                if input_type is None:
                    raise ValueError("input_type must be provided when output_map is used without input_map")
                self.input_proxy = IOProxy(input_type, "Input")
            else:
                self.input_proxy = IOProxy(None, "Input")

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
                        inner_only: bool = False,
                        signal_range: Optional[Tuple[Optional[int], Optional[int]]] = None):

        if registers is None:
            registers = self.default_registers
        
        for read_op in global_reads_iterator(
                registers=registers,
                r2c=r2c,
                inverse=inverse,
                format_transposed=format_transposed,
                inner_only=inner_only,
                signal_range=signal_range
            ):
            
            if proxy.has_callback():
                set_read_op(read_op)
                proxy.do_callback()
                set_read_op(None)
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
                set_write_op(write_op)
                proxy.do_callback()
                set_write_op(None)
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
    
    def read_kernel(self, registers: Optional[FFTRegisters] = None, format_transposed: bool = False, inner_only: bool = False):
        self.read_from_proxy(
            self.kernel_proxy,
            registers,
            format_transposed=format_transposed,
            inner_only=inner_only
        )
