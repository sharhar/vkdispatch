import vkdispatch as vd
import vkdispatch.codegen as vc

import contextlib
from typing import Optional, Tuple, Union, List, Dict

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

    def register_input_format(self, stage_index: int = 0) -> Dict[int, int]:
        in_format = {}

        stride = self.config.N // self.config.stages[stage_index].fft_length

        register_count = len(self.resources.registers)
        register_index_list = list(range(register_count))

        for invocation in self.resources.invocations[stage_index]:
            sub_registers = register_index_list[invocation.register_selection]
            
            for i in range(len(sub_registers)):
                in_format[invocation.get_read_index(stride * i)] = sub_registers[i]

        return in_format

    def register_output_format(self, stage_index: int = -1) -> Dict[int, int]:
        out_format = {}

        register_count = len(self.resources.registers)
        register_index_list = list(range(register_count))

        for jj in range(self.config.stages[stage_index].fft_length):
            for invocation in self.resources.invocations[stage_index]:
                out_format[invocation.get_write_index(jj)] = register_index_list[invocation.register_selection][jj]

        return out_format

    def register_shuffle(self, output_stage: int = -1, input_stage: int = 0, registers: List[vc.ShaderVariable] = None) -> Dict[int, int]:
        out_format = self.register_output_format(output_stage)
        in_format = self.register_input_format(input_stage)

        if out_format.keys() != in_format.keys():
            self.write_sdata(stage_index=output_stage, registers=registers)
            self.read_sdata(stage_index=input_stage, registers=registers)
            return
        
        if registers is None:
            registers = self.resources.registers

        shuffled_registers = [None] * len(registers)

        for i in range(len(registers)):
            format_key = None
            
            for k, v in in_format.items():
                if v == i:
                    format_key = k
                    break

            assert format_key is not None, "Could not find register in output format???"

            shuffled_registers[i] = registers[out_format[format_key]]

        for i in range(len(registers)):
            registers[i] = shuffled_registers[i]

    def execute(self, inverse: bool = False):
        stage_count = len(self.config.stages)

        for i in range(stage_count):
            stage = self.config.stages[i]

            vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")

            if i != 0:
                self.sdata.read_registers(
                    resources=self.resources,
                    config=self.config,
                    stage_index=i
                )

            self.resources.stage_begin(i)
            for ii, invocation in enumerate(self.resources.invocations[i]):
                self.resources.invocation_gaurd(i, ii)

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
                self.sdata.write_registers(
                    resources=self.resources,
                    config=self.config,
                    stage_index=i
                )

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