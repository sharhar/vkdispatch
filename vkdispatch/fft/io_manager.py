import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional

from .io_proxy import IOProxy

class IOManager:
    output_proxy: IOProxy
    input_proxy: IOProxy
    kernel_proxy: IOProxy

    signature: vd.ShaderSignature

    def __init__(self,
                    builder: vc.ShaderBuilder,
                    output: Optional[vd.MappingFunction],
                    input: Optional[vd.MappingFunction] = None,
                    kernel: Optional[vd.MappingFunction] = None):
            

            self.output_proxy = IOProxy(vd.complex64 if output is None else output, "Output")
            self.input_proxy = IOProxy(input, "Input")
            self.kernel_proxy = IOProxy(kernel, "Kernel")
    
            output_types = self.output_proxy.buffer_types
            input_types = self.input_proxy.buffer_types
            kernel_types = self.kernel_proxy.buffer_types
    
            all_types = output_types + input_types + kernel_types
    
            if len(all_types) == 0:
                raise ValueError("A big error happened")
    
            self.signature = vd.ShaderSignature.from_type_annotations(builder, all_types)
            sig_vars = self.signature.get_variables()
    
            output_count = len(output_types)
            input_count = len(input_types)
    
            self.output_proxy.set_variables(sig_vars[:output_count])
            self.input_proxy.set_variables(sig_vars[output_count:input_count + output_count])
            self.kernel_proxy.set_variables(sig_vars[input_count + output_count:])

            if input_count == 0:
                self.input_proxy = self.output_proxy
