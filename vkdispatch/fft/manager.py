import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional, Tuple, Union

from .io_manager import IOManager
from .config import FFTConfig
from .resources import FFTResources, allocate_fft_resources

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

class FFTManager:
    builder: vc.ShaderBuilder
    io_manager: IOManager
    config: FFTConfig
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
        self.io_manager = IOManager(builder, output_map, input_map, kernel_map)
        self.config = FFTConfig(buffer_shape, axis, max_register_count)
        self.resources = allocate_fft_resources(self.config, True)
        self.fft_callable = None
        self.name = name if name is not None else f"fft_shader_{buffer_shape}_{axis}"
        
    def compile_shader(self):
        self.fft_callable = FFTCallable(vd.ShaderObject(
                self.builder.build(self.name),
                self.io_manager.signature,
                local_size=self.resources.local_size
            ),
            self.resources.exec_size
        )

    def get_callable(self) -> FFTCallable:
        assert self.fft_callable is not None, "Shader not compiled yet... something is wrong"
        return self.fft_callable
