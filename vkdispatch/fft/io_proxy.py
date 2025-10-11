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
             resources: FFTResources,
             config: FFTConfig,
             register: vc.ShaderVariable,
             r2c: bool = False,
             inverse: bool = None,
             fft_index: int = None) -> vc.ShaderVariable:
        assert self.enabled, f"{self.name} IOProxy is not enabled"

        if r2c:
            assert inverse is not None, "Must specify inverse for r2c read"

        if r2c and inverse:
            assert self.map_func is None, "Mapping functions do not support inverse r2c operations"
            assert fft_index is not None, "FFT index must be provided for inverse r2c read"
        
            vc.if_statement(fft_index >= (config.N // 2) + 1)
            resources.io_index_2[:] = 2 * resources.input_batch_offset + config.N * config.fft_stride - resources.io_index
            register[:] = self.buffer_variables[0][resources.io_index_2]
            register.y = -register.y
            vc.else_statement()
            register[:] = self.buffer_variables[0][resources.io_index]
            vc.end()

            return
        
        if self.map_func is not None:
            vc.set_mapping_index(resources.io_index)
            vc.set_mapping_registers([register, resources.omega_register])

            self.map_func.callback(*self.buffer_variables)

            return
        
        if not r2c:
            register[:] = self.buffer_variables[0][resources.io_index]
            return
        
        real_value = self.buffer_variables[0][resources.io_index / 2][resources.io_index % 2]
        register[:] = f"vec2({real_value}, 0)"

    def read_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            grid: FFTGridManager,
                            r2c: bool = False,
                            inverse: bool = None,
                            stage_index: int = 0,
                            registers: List[vc.ShaderVariable] = None):
        if registers is None:
            registers = resources.registers

        vc.comment(f"Loading to registers from buffer {self.buffer_variables[0]}")

        input_batch_stride_y = config.batch_outer_stride

        resources.stage_begin(stage_index)

        if r2c:
            assert inverse is not None, "Must specify inverse for r2c read"

            if not inverse:
                input_batch_stride_y = ((config.N // 2) + 1) * 2
            if inverse:
                input_batch_stride_y = (config.N // 2) + 1

        resources.input_batch_offset[:] = grid.global_outer * input_batch_stride_y + grid.global_inner * config.batch_inner_stride

        for ii, invocation in enumerate(resources.invocations[stage_index]):
            #if config.stages[stage_index].remainder_offset == 1 and ii == config.stages[stage_index].extra_ffts:
            #    vc.if_statement(grid.tid < config.N // config.stages[stage_index].registers_used)

            resources.invocation_gaurd(stage_index, ii)

            offset = invocation.instance_id
            stride = config.N // config.stages[stage_index].fft_length

            resources.io_index[:] = offset * config.fft_stride + resources.input_batch_offset

            register_list = registers[invocation.register_selection]

            for i in range(len(register_list)):
                if i != 0:
                    resources.io_index += stride * config.fft_stride
                
                self.read_register(
                    resources,
                    config,
                    register_list[i],
                    r2c=r2c,
                    inverse=inverse,
                    fft_index=i * stride + offset
                )

        resources.invocation_end(stage_index)

        # if config.stages[stage_index].remainder_offset == 1:
        #     vc.end()

        resources.stage_end(stage_index)

    def write_register(self,
                resources: FFTResources,
                config: FFTConfig,
                register: vc.ShaderVariable,
                r2c: bool = False,
                inverse: bool = None,
                fft_index: vc.ShaderVariable = None) -> vc.ShaderVariable:
            assert self.enabled, f"{self.name} IOProxy is not enabled"
            
            if self.map_func is not None:

                do_if = False

                if r2c:
                    assert inverse is not None, "Must specify inverse for r2c write"
                    if not inverse:
                        do_if = True

                if do_if:
                    assert fft_index is not None, "FFT index must be provided for forward r2c write"

                    vc.if_statement(fft_index < (config.N // 2) + 1)

                vc.set_mapping_index(resources.io_index)
                vc.set_mapping_registers([register])
                self.map_func.callback(*self.buffer_variables)

                if do_if:
                    vc.end()

                return
            
            if not r2c:
                self.buffer_variables[0][resources.io_index] = register
                return
            
            assert inverse is not None, "Must specify inverse for r2c write"
            
            if not inverse:
                assert fft_index is not None, "FFT index must be provided for forward r2c write"

                vc.if_statement(fft_index < (config.N // 2) + 1)
                self.buffer_variables[0][resources.io_index] = register
                vc.end()
                return


            self.buffer_variables[0][resources.io_index / 2][resources.io_index % 2] = register.x
    
    def write_registers(self,
                            resources: FFTResources,
                            config: FFTConfig,
                            grid: FFTGridManager,
                            r2c: bool = False,
                            inverse: bool = None,
                            stage_index: int = -1,
                            registers: List[vc.ShaderVariable] = None):
        if registers is None:
            registers = resources.registers

        stage = config.stages[stage_index]

        vc.comment(f"Storing from registers to buffer")

        #do_runtime_if = config.stages[stage_index].thread_count < config.batch_threads
        #if do_runtime_if: vc.if_statement(grid.tid < config.stages[stage_index].thread_count)
        
        resources.stage_begin(stage_index)

        output_batch_stride_y = config.batch_outer_stride

        if r2c:
            assert inverse is not None, "Must specify inverse for r2c write"

            if not inverse:
                output_batch_stride_y = (config.N // 2) + 1
            if inverse:
                output_batch_stride_y = ((config.N // 2) + 1) * 2

        resources.output_batch_offset[:] = grid.global_outer * output_batch_stride_y + grid.global_inner * config.batch_inner_stride

        resources.io_index[:] = grid.tid * config.fft_stride + resources.output_batch_offset
        
        instance_index_stride = config.N // (stage.fft_length * stage.instance_count)

        for jj in range(stage.fft_length):
            for ii, invocation in enumerate(resources.invocations[stage_index]):
                #if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                #    vc.if_statement(grid.tid < config.N // stage.registers_used)

                resources.invocation_gaurd(stage_index, ii)

                if jj != 0 or ii != 0:
                    resources.io_index += instance_index_stride * config.fft_stride

                register = registers[invocation.register_selection][jj]

                self.write_register(
                    resources,
                    config,
                    register,
                    r2c=r2c,
                    inverse=inverse,
                    fft_index=invocation.sub_sequence_offset + jj * resources.output_strides[stage_index]
                )

            resources.invocation_end(stage_index)

            # if stage.remainder_offset == 1:
            #     vc.end()

        resources.stage_end(stage_index)

        #if do_runtime_if: vc.end()