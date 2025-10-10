import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

from .config import FFTConfig
from .grid_manager import FFTGridManager
from .resources import FFTResources

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

    def read_register(self,
             register: vc.ShaderVariable,
             memory_index: vc.ShaderVariable,
             spare_register: vc.ShaderVariable = None,
             r2c: bool = False) -> vc.ShaderVariable:
        assert self.enabled, f"{self.name} IOProxy is not enabled"
        
        if self.map_func is not None:
            assert spare_register is not None, "Spare register must be provided when using a mapping function"

            vc.set_mapping_index(memory_index)
            vc.set_mapping_registers([register, spare_register])

            self.map_func.callback(*self.buffer_variables)

            return
        
        if not r2c:
            register[:] = self.buffer_variables[0][memory_index]
            return
        
        real_value = self.buffer_variables[0][memory_index / 2][memory_index % 2]
        register[:] = f"vec2({real_value}, 0)"

    def read_r2c_inverse_register(self,
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

    def read_to_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            grid: FFTGridManager,
                            inverse: bool,
                            r2c: bool = False,
                            stage_index: int = 0,
                            registers: List[vc.ShaderVariable] = None):
        if registers is None:
            registers = resources.registers

        vc.comment(f"Loading to registers from buffer {self.buffer_variables[0]}")

        for ii, invocation in enumerate(resources.invocations[stage_index]):
            if config.stages[stage_index].remainder_offset == 1 and ii == config.stages[stage_index].extra_ffts:
                vc.if_statement(grid.tid < config.N // config.stages[stage_index].registers_used)

            offset = invocation.instance_id
            stride = config.N // config.stages[stage_index].fft_length

            resources.io_index[:] = offset * config.fft_stride + resources.input_batch_offset

            register_list = registers[invocation.register_selection]

            for i in range(len(register_list)):
                if i != 0:
                    resources.io_index += stride * config.fft_stride
                
                if r2c and inverse:
                    self.read_r2c_inverse_register(
                        register=register_list[i],
                        memory_index=resources.io_index,
                        fft_index=i * stride + offset,
                        spare_index=resources.io_index_2,
                        input_batch_offset=resources.input_batch_offset,
                        fft_size=config.N,
                        fft_stride=config.fft_stride
                    )
                else:
                    self.read_register(register_list[i], resources.io_index, spare_register=resources.omega_register, r2c=r2c)

        if config.stages[stage_index].remainder_offset == 1:
            vc.end()

    def write_register(self,
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
                self.map_func.callback(*self.buffer_variables)

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
    
    def write_from_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            grid: FFTGridManager,
                            inverse: bool,
                            r2c: bool = False,
                            normalize: bool = True,
                            stage_index: int = -1,
                            registers: List[vc.ShaderVariable] = None):
        if registers is None:
            registers = resources.registers

        stage = config.stages[stage_index]

        resources.io_index[:] = grid.tid * config.fft_stride + resources.output_batch_offset

        vc.comment(f"Storing from registers to buffer")
        
        instance_index_stride = config.N // (stage.fft_length * stage.instance_count)

        for jj in range(stage.fft_length):
            for ii, invocation in enumerate(resources.invocations[stage_index]):
                if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                    vc.if_statement(grid.tid < config.N // stage.registers_used)

                if jj != 0 or ii != 0:
                    resources.io_index += instance_index_stride * config.fft_stride

                register = registers[invocation.register_selection][jj]

                if normalize and inverse:
                    register[:] = register / config.N

                self.write_register(
                    register=register,
                    memory_index=resources.io_index,
                    r2c=r2c,
                    inverse=inverse,
                    fft_size=config.N,
                    fft_index=invocation.sub_sequence_offset + jj * resources.output_strides[stage_index]
                )

            if stage.remainder_offset == 1:
                vc.end()