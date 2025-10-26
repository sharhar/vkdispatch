import vkdispatch as vd
import vkdispatch.codegen as vc

import contextlib
from typing import Optional, Tuple, Union, List, Dict

from .io_manager import IOManager
from .config import FFTConfig
from .grid_manager import FFTGridManager
from .sdata_manager import FFTSDataManager
from .resources import FFTResources
from .registers import FFTRegisters
from .cooley_tukey import radix_composite, apply_twiddle_factors

class FFTCallable:
    shader_function: vd.ShaderFunction
    exec_size: Tuple[int, int, int]

    def __init__(self, shader_function: vd.ShaderFunction, exec_size: Tuple[int, int, int]):
        self.shader_function = shader_function
        self.exec_size = exec_size

    def __call__(self, *args, **kwargs):
        self.shader_function(*args, exec_size=self.exec_size, **kwargs)

    def __repr__(self):
        return repr(self.shader_function)

class FFTContext:
    shader_context: vd.ShaderContext
    io_manager: IOManager
    config: FFTConfig
    grid: FFTGridManager
    registers: FFTRegisters
    sdata: FFTSDataManager
    resources: FFTResources
    fft_callable: FFTCallable
    name: str

    def __init__(self,
                shader_context: vd.ShaderContext,
                buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                output_map: Union[vd.MappingFunction, type, None] = None,
                input_map: Union[vd.MappingFunction, type, None] = None,
                kernel_map: Union[vd.MappingFunction, type, None] = None,
                name: str = None):
        self.shader_context = shader_context
        
        self.config = FFTConfig(buffer_shape, axis, max_register_count)
        self.grid = FFTGridManager(self.config, True)
        self.resources = FFTResources(self.config, self.grid)

        self.io_manager = IOManager(shader_context, output_map, input_map, kernel_map)
        self.sdata = FFTSDataManager(self.config, self.grid)
        
        self.registers = self.allocate_registers("fft")
        
        self.fft_callable = None
        self.name = name if name is not None else f"fft_shader_{buffer_shape}_{axis}"

    def allocate_registers(self, name: str, count: int = None) -> FFTRegisters:
        assert name is not None, "Must provide a name for allocated registers"

        if count is None:
            count = self.config.register_count

        return FFTRegisters(self.resources, self.sdata, count, name)

    def read_input(self,
                   r2c: bool = False,
                   inverse: bool = None,
                   registers: Optional[FFTRegisters] = None):
        if r2c:
            assert inverse is not None, "Must specify inverse for r2c read"

        if registers is None:
            registers = self.registers

        self.io_manager.input_proxy.read_registers(
            registers,
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse
        )

    def write_output(self,
                    r2c: bool = False,
                    inverse: bool = None,
                    normalize: bool = None,
                    registers: Optional[FFTRegisters] = None):
        
        if registers is None:
            registers = self.registers
    
        if inverse is not None:
            if inverse:
                assert normalize is not None, "Must specify normalize when specifying inverse"

                for i in range(registers.count):
                    if normalize:
                        registers[i] = registers[i] / self.config.N

        self.io_manager.output_proxy.write_registers(
            registers,
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse
        )

    def read_kernel(self, registers: Optional[FFTRegisters] = None):
        if registers is None:
            registers = self.registers
        
        self.io_manager.kernel_proxy.read_registers(
            registers,
            self.resources,
            self.config,
            self.grid
        )

    def write_kernel(self, registers: Optional[FFTRegisters] = None):
        if registers is None:
            registers = self.registers
        
        self.io_manager.kernel_proxy.write_registers(
            registers,
            self.resources,
            self.config,
            self.grid
        )

    def compile_shader(self):
        self.fft_callable = FFTCallable(self.shader_context.get_function(self.grid.local_size), self.grid.exec_size)

    def get_callable(self) -> FFTCallable:
        assert self.fft_callable is not None, "Shader not compiled yet... something is wrong"
        return self.fft_callable

    def execute(self, inverse: bool = False):
        stage_count = len(self.config.stages)

        for i in range(stage_count):
            stage = self.config.stages[i]

            vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")

            if i != 0:
                self.registers.shuffle(output_stage=i-1, input_stage=i)

            self.resources.stage_begin(i)
            for ii, invocation in enumerate(self.resources.invocations[i]):
                self.resources.invocation_gaurd(i, ii)

                apply_twiddle_factors(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.registers.slice(invocation.register_selection), 
                    twiddle_index=invocation.inner_block_offset, 
                    twiddle_N=invocation.block_width
                )

                self.registers.slice_set(invocation.register_selection, radix_composite(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.registers.slice(invocation.register_selection),
                    primes=stage.primes
                ))

            self.resources.invocation_end(i)
            self.resources.stage_end(i)

@contextlib.contextmanager
def fft_context(buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                output_map: Union[vd.MappingFunction, type, None] = None,
                input_map: Union[vd.MappingFunction, type, None] = None,
                kernel_map: Union[vd.MappingFunction, type, None] = None):

    try:
        with vd.shader_context(vc.ShaderFlags.NO_EXEC_BOUNDS) as context:
            fft_context = FFTContext(
                shader_context=context,
                buffer_shape=buffer_shape,
                axis=axis,
                max_register_count=max_register_count,
                output_map=output_map,
                input_map=input_map,
                kernel_map=kernel_map
            )

            yield fft_context

            fft_context.compile_shader()

    finally:
        pass        