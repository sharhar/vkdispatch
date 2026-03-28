import vkdispatch.codegen as vc

from typing import List, Optional, Any

import contextlib

class ShaderContext:
    builder: vc.ShaderBuilder
    shader_description: vc.ShaderDescription

    def __init__(self, builder: vc.ShaderBuilder):
        self.builder = builder
        self.shader_description = None

    def get_description(self, name: Optional[str] = None):
        if self.shader_description is not None:
            return self.shader_description

        self.shader_description = self.builder.build("shader" if name is None else name)

        return self.shader_description
    
    def declare_input_arguments(self,
                                type_annotations: List,
                                names: Optional[List[str]] = None,
                                defaults: Optional[List[Any]] = None):
        self.builder.declare_shader_arguments(type_annotations, names, defaults)
        return self.builder.get_shader_arguments()

@contextlib.contextmanager
def shader_context(flags: vc.ShaderFlags = vc.ShaderFlags.NONE):

    builder = vc.ShaderBuilder(flags=flags)
    old_builder = vc.set_builder(builder)

    context = ShaderContext(builder)

    try:
        yield context
    finally:
        vc.set_builder(old_builder)