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

class FFTContext:
    shader_context: vd.ShaderContext
    config: FFTConfig
    grid: FFTGridManager
    registers: FFTRegisters
    sdata: FFTSDataManager
    resources: FFTResources
    fft_callable: vd.ShaderFunction
    name: str

    declared_shader_args: bool
    declarer: str

    def __init__(self,
                shader_context: vd.ShaderContext,
                buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                name: str = None):
        self.shader_context = shader_context
        self.declared_shader_args = False
        self.declarer = None
        
        self.config = FFTConfig(buffer_shape, axis, max_register_count)
        self.grid = FFTGridManager(self.config, True)
        self.resources = FFTResources(self.config, self.grid)

        self.registers = self.allocate_registers("fft")
        
        self.sdata = FFTSDataManager(self.config, self.grid, self.registers)
        
        self.fft_callable = None
        self.name = name if name is not None else f"fft_shader_{buffer_shape}_{axis}"

    def allocate_registers(self, name: str, count: int = None) -> FFTRegisters:
        assert name is not None, "Must provide a name for allocated registers"

        if count is None:
            count = self.config.register_count

        return FFTRegisters(self.resources, count, name)

    def declare_shader_args(self, types: List) -> List[vc.ShaderVariable]:
        assert not self.declared_shader_args, f"Shader arguments already declared with {self.declarer}"
        self.declared_shader_args = True
        self.declarer = "declare_shader_args"
        return self.shader_context.declare_input_arguments(types)

    def make_io_manager(self,
                        output_map: Optional[vd.MappingFunction],
                        input_map: Optional[vd.MappingFunction] = None,
                        kernel_map: Optional[vd.MappingFunction] = None) -> IOManager:
        assert not self.declared_shader_args, f"Shader arguments already declared with {self.declarer}"
        self.declared_shader_args = True
        self.declarer = "make_io_manager"
        return IOManager(
            default_registers=self.registers,
            shader_context=self.shader_context,
            output_map=output_map,
            input_map=input_map,
            kernel_map=kernel_map
        )

    def register_shuffle(self,
                         registers: Optional[FFTRegisters] = None,
                         output_stage: int = -1,
                         input_stage: int = 0) -> bool:
        if registers is None:
            registers = self.registers
        
        if registers.try_shuffle(
            output_stage=output_stage,
            input_stage=input_stage
        ):
            return True

        self.sdata.write_to_sdata(
            registers=registers,
            stage_index=output_stage
        )

        self.sdata.read_from_sdata(
            registers=registers,
            stage_index=input_stage
        )

    def compile_shader(self):
        self.fft_callable = self.shader_context.get_function(
            local_size=self.grid.local_size,
            exec_count=self.grid.exec_size
        )

    def get_callable(self) -> vd.ShaderFunction:
        assert self.fft_callable is not None, "Shader not compiled yet... something is wrong"
        return self.fft_callable

    def execute(self, inverse: bool):
        stage_count = len(self.config.stages)

        for i in range(stage_count):
            stage = self.config.stages[i]

            vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")

            if i != 0:
                self.register_shuffle(output_stage=i-1, input_stage=i)

            self.resources.stage_begin(i)
            for ii, invocation in enumerate(self.resources.invocations[i]):
                self.resources.invocation_gaurd(i, ii)

                apply_twiddle_factors(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.registers.register_slice(invocation.register_selection), 
                    twiddle_index=invocation.inner_block_offset, 
                    twiddle_N=invocation.block_width
                )

                self.registers.slice_set(invocation.register_selection, radix_composite(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.registers.register_slice(invocation.register_selection),
                    primes=stage.primes
                ))

            self.resources.invocation_end(i)
            self.resources.stage_end(i)

@contextlib.contextmanager
def fft_context(buffer_shape: Tuple,
                axis: Optional[int] = None,
                max_register_count: Optional[int] = None):

    try:
        with vd.shader_context(vc.ShaderFlags.NO_EXEC_BOUNDS) as context:
            fft_context = FFTContext(
                shader_context=context,
                buffer_shape=buffer_shape,
                axis=axis,
                max_register_count=max_register_count
            )

            yield fft_context

            fft_context.compile_shader()

    finally:
        pass        