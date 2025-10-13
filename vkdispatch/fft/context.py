import vkdispatch as vd
import vkdispatch.codegen as vc

import contextlib
from typing import Optional, Tuple, Union, List

from .io_manager import IOManager
from .config import FFTConfig
from .grid_manager import FFTGridManager
from .sdata_manager import FFTSDataManager
from .resources import FFTResources
from .cooley_tukey import radix_composite, apply_twiddle_factors

class FFTCallable:
    shader_object: vd.ShaderObject
    exec_size: Tuple[int, int, int]

    def __init__(self, shader_object: vd.ShaderObject, exec_size: Tuple[int, int, int]):
        self.shader_object = shader_object
        self.exec_size = exec_size

    def __call__(self, *args, **kwargs):
        self.shader_object(*args, exec_size=self.exec_size, **kwargs)

    def __repr__(self):
        return repr(self.shader_object)

class FFTContext:
    builder: vc.ShaderBuilder
    io_manager: IOManager
    config: FFTConfig
    grid: FFTGridManager
    sdata: FFTSDataManager
    resources: FFTResources
    fft_callable: FFTCallable
    name: str

    def __init__(self,
                builder: vc.ShaderBuilder,
                buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                output_map: Union[vd.MappingFunction, type, None] = None,
                input_map: Union[vd.MappingFunction, type, None] = None,
                kernel_map: Union[vd.MappingFunction, type, None] = None,
                name: str = None):
        self.builder = builder
        
        self.config = FFTConfig(buffer_shape, axis, max_register_count)
        self.grid = FFTGridManager(self.config, True)
        self.resources = FFTResources(self.config, self.grid)
        
        self.io_manager = IOManager(builder, output_map, input_map, kernel_map)
        self.sdata = FFTSDataManager(self.config, self.grid)
        
        self.fft_callable = None
        self.name = name if name is not None else f"fft_shader_{buffer_shape}_{axis}"

    def read_input(self,
                   r2c: bool = False,
                   inverse: bool = None,
                   registers: Optional[List[vc.ShaderVariable]] = None):
        if r2c:
            assert inverse is not None, "Must specify inverse for r2c read"

        self.io_manager.input_proxy.read_registers(
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse,
            registers=registers
        )

    def write_output(self,
                    r2c: bool = False,
                    inverse: bool = None,
                    normalize: bool = None,
                    registers: Optional[List[vc.ShaderVariable]] = None):
        if inverse is not None:
            if inverse:
                assert normalize is not None, "Must specify normalize when specifying inverse"
            
                if registers is None:
                    registers = self.resources.registers

                for register in registers:
                    if normalize:
                        register[:] = register / self.config.N

        self.io_manager.output_proxy.write_registers(
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse,
            registers=registers
        )

    def read_kernel(self,
                   r2c: bool = False,
                   inverse: bool = None,
                   registers: Optional[List[vc.ShaderVariable]] = None):
        if r2c:
            assert inverse is not None, "Must specify inverse for r2c read"

        self.io_manager.kernel_proxy.read_registers(
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse,
            registers=registers
        )

    def write_kernel(self,
                    r2c: bool = False,
                    inverse: bool = None,
                    normalize: bool = None,
                    registers: Optional[List[vc.ShaderVariable]] = None):
        if inverse is not None:
            if inverse:
                assert normalize is not None, "Must specify normalize when specifying inverse"
            
                if registers is None:
                    registers = self.resources.registers

                for register in registers:
                    if normalize:
                        register[:] = register / self.config.N

        self.io_manager.kernel_proxy.write_registers(
            self.resources,
            self.config,
            self.grid,
            r2c=r2c,
            inverse=inverse,
            registers=registers
        )

    def read_sdata(self,
                   stage_index: int = 0,
                   invocation_index: int = None,
                   registers: Optional[List[vc.ShaderVariable]] = None):
        self.sdata.read_registers(
            self.resources,
            self.config,
            stage_index,
            invocation_index,
            registers
        )

    def write_sdata(self, stage_index: int = -1, registers: Optional[List[vc.ShaderVariable]] = None):
        self.sdata.write_registers(self.resources, self.config, stage_index, registers)
        
    def compile_shader(self):
        self.fft_callable = FFTCallable(vd.ShaderObject(
                self.builder.build(self.name),
                self.io_manager.signature,
                local_size=self.grid.local_size
            ),
            self.grid.exec_size
        )

    def get_callable(self) -> FFTCallable:
        assert self.fft_callable is not None, "Shader not compiled yet... something is wrong"
        return self.fft_callable

    def reorder_registers(self, registers: List[vc.ShaderVariable] = None):
        if registers is None:
            registers = self.resources.registers

        new_order = [None] * len(registers)

        stage = self.config.stages[-1]

        invocation_count = len(self.resources.invocations[-1])

        for jj in range(stage.fft_length):
            for ii, invocation in enumerate(self.resources.invocations[-1]):
                new_order[jj * invocation_count + ii] = registers[invocation.register_selection][jj]

        for i in range(len(registers)):
            registers[i] = new_order[i]

    def execute(self, inverse: bool = False):
        stage_count = len(self.config.stages)

        for i in range(stage_count):
            stage = self.config.stages[i]

            vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")

            self.resources.stage_begin(i)
            for ii, invocation in enumerate(self.resources.invocations[i]):
                
                self.resources.invocation_gaurd(i, ii)

                if i != 0:
                    self.sdata.read_registers(
                        resources=self.resources,
                        config=self.config,
                        stage_index=i,
                        invocation_index=ii
                    )

                apply_twiddle_factors(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.resources.registers[invocation.register_selection], 
                    twiddle_index=invocation.inner_block_offset, 
                    twiddle_N=invocation.block_width
                )

                self.resources.registers[invocation.register_selection] = radix_composite(
                    resources=self.resources,
                    inverse=inverse,
                    register_list=self.resources.registers[invocation.register_selection],
                    primes=stage.primes
                )

            self.resources.invocation_end(i)
            self.resources.stage_end(i)

            if i < stage_count - 1:
                if i != 0:
                    vc.barrier()

                self.sdata.write_registers(
                    resources=self.resources,
                    config=self.config,
                    stage_index=i
                )

                vc.barrier()

        # self.reorder_registers()

@contextlib.contextmanager
def fft_context(buffer_shape: Tuple,
                axis: int = None,
                max_register_count: int = None,
                output_map: Union[vd.MappingFunction, type, None] = None,
                input_map: Union[vd.MappingFunction, type, None] = None,
                kernel_map: Union[vd.MappingFunction, type, None] = None):

    try:
        with vc.builder_context(enable_exec_bounds=False) as builder:
            fft_context = FFTContext(
                builder=builder,
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