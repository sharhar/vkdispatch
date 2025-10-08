import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

class IOProxy:
    buffer_variables: List[vc.Buffer]
    buffer_types: List[vd.dtype]
    map_func: Optional[vd.MappingFunction]
    enabled: bool

    def __init__(self, obj: Union[vd.Buffer, vd.MappingFunction] = None):
        self.buffer_variables = None

        if obj is None:
            self.buffer_types = []
            self.map_func = None
            self.enabled = False

        elif isinstance(obj, vd.Buffer):
            self.buffer_types = [vc.Buffer[obj.var_type]]
            self.map_func = None
            self.enabled = True

        elif isinstance(obj, vd.MappingFunction):
            self.buffer_types = obj.buffer_types
            self.map_func = obj
            self.enabled = True

        else:
            raise ValueError("IOObject must be initialized with a Buffer or MappingFunction")
    
    def set_variables(self, vars: List[vc.Buffer]) -> None:
        if self.map_func is None:
            assert len(vars) == 1, "Buffer IOObject must have exactly one buffer variable"
        
        assert len(vars) == len(self.buffer_types), "Number of buffer variables does not match number of buffer types"

        self.buffer_variables = vars

class IOManager:
    output_proxy: IOProxy
    input_proxy: IOProxy
    kernel_proxy: IOProxy

    signature: vd.ShaderSignature

    def __init__(self,
                    builder: vc.ShaderBuilder,
                    output: Union[vd.Buffer, vd.MappingFunction],
                    input: Union[vd.Buffer, vd.MappingFunction] = None,
                    kernel: Union[vd.Buffer, vd.MappingFunction] = None):
            
            self.output_proxy = IOProxy(output)
            self.input_proxy = IOProxy(input)
            self.kernel_proxy = IOProxy(kernel)
    
            input_types = self.input_proxy.buffer_types
            output_types = self.output_proxy.buffer_types
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
