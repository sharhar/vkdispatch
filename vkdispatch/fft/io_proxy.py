import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

class IOProxy:
    buffer_variables: List[vc.Buffer]
    buffer_types: List[type]
    map_func: Optional[vd.MappingFunction]
    enabled: bool
    name: str

    def __init__(self, obj: Union[type, vd.MappingFunction], name: str):
        self.buffer_variables = None
        self.name = name

        if obj is None:
            self.buffer_types = []
            self.map_func = None
            self.enabled = False

        elif isinstance(obj, type):
            self.buffer_types = [vc.Buffer[obj]]
            self.map_func = None
            self.enabled = True

        elif isinstance(obj, vd.MappingFunction):
            self.buffer_types = obj.buffer_types
            self.map_func = obj
            self.enabled = True

        else:
            raise ValueError("IOObject must be initialized with a Buffer or MappingFunction")
    
    def set_variables(self, vars: List[vc.Buffer]) -> None:
        assert len(vars) == len(self.buffer_types), "Number of buffer variables does not match number of buffer types"
        if len(vars) == 0:
            self.enabled = False
            return
        
        if self.map_func is None:
            assert len(vars) == 1, "Buffer IOObject must have exactly one buffer variable"

        self.buffer_variables = vars

    def read(self,
             register: vc.ShaderVariable,
             memory_index: vc.ShaderVariable,
             spare_register: vc.ShaderVariable = None,
             r2c: bool = False) -> vc.ShaderVariable:
        assert self.enabled, f"{self.name} IOProxy is not enabled"
        
        if self.map_func is not None:
            assert spare_register is not None, "Spare register must be provided when using a mapping function"

            vc.set_mapping_index(memory_index)
            vc.set_mapping_registers([register, spare_register])
            
            self.map_func.mapping_function(*self.buffer_variables)

            return
        
        if not r2c:
            register[:] = self.buffer_variables[0][memory_index]
            return
        
        real_value = self.buffer_variables[0][memory_index / 2][memory_index % 2]
        register[:] = f"vec2({real_value}, 0)"

    def read_r2c_inverse(self,
                         register: vc.ShaderVariable,
                         memory_index: vc.ShaderVariable,
                         fft_index: vc.ShaderVariable,
                         spare_index: vc.ShaderVariable,
                         input_batch_offset: vc.ShaderVariable,
                         fft_size: int,
                         fft_stride: int) -> vc.ShaderVariable:
        assert self.enabled, f"{self.name} IOProxy is not enabled"
        
        assert self.map_func is None, "Mapping functions do not support inverse r2c operations"
        
        vc.if_statement(fft_index >= (fft_size // 2) + 1)
        spare_index[:] = 2 * input_batch_offset + fft_size * fft_stride - memory_index
        register[:] = self.buffer_variables[0][spare_index]
        register.y = -register.y
        vc.else_statement()
        register[:] = self.buffer_variables[0][memory_index]
        vc.end()

    def write(self,
                register: vc.ShaderVariable,
                memory_index: vc.ShaderVariable,
                r2c: bool = False,
                inverse: bool = False,
                fft_index: vc.ShaderVariable = None,
                fft_size: int = None) -> vc.ShaderVariable:
            assert self.enabled, f"{self.name} IOProxy is not enabled"
            
            if self.map_func is not None:

                if not inverse and r2c:
                    assert fft_size is not None, "FFT size must be provided for forward r2c write"
                    assert fft_index is not None, "FFT index must be provided for forward r2c write"

                    vc.if_statement(fft_index < (fft_size // 2) + 1)

                vc.set_mapping_index(memory_index)
                vc.set_mapping_registers([register])
                self.map_func.mapping_function(*self.buffer_variables)

                if not inverse and r2c:
                    vc.end()

                return
            
            if not r2c:
                self.buffer_variables[0][memory_index] = register
                return
            
            if not inverse:
                assert fft_size is not None, "FFT size must be provided for forward r2c write"
                assert fft_index is not None, "FFT index must be provided for forward r2c write"

                vc.if_statement(fft_index < (fft_size // 2) + 1)
                self.buffer_variables[0][memory_index] = register
                vc.end()
                return


            self.buffer_variables[0][memory_index / 2][memory_index % 2] = register.x
            