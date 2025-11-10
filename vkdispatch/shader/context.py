import vkdispatch as vd
import vkdispatch.codegen as vc

from .signature import ShaderSignature

from typing import List

import contextlib

class ShaderContext:
    builder: vc.ShaderBuilder
    signature: ShaderSignature
    shader_function: vd.ShaderFunction

    def __init__(self, builder: vc.ShaderBuilder):
        self.builder = builder
        self.signature = None
    
    def get_function(self,
                     local_size=None,
                      workgroups=None,
                      exec_count=None) -> vd.ShaderFunction:
        return vd.ShaderFunction.from_description(
            self.builder.build("shader"),
            self.signature,
            local_size=local_size,
            workgroups=workgroups,
            exec_count=exec_count
        )
    
    def declare_input_arguments(self, annotations: List):
        self.signature = ShaderSignature.from_type_annotations(self.builder, annotations)
        return self.signature.get_variables()

@contextlib.contextmanager
def shader_context(flags: vc.ShaderFlags = vc.ShaderFlags.NONE):

    builder = vc.ShaderBuilder(flags=flags, is_apple_device=vd.get_context().is_apple())
    old_builder = vc.set_global_builder(builder)

    context = ShaderContext(builder)

    try:
        yield context
    finally:
        vc.set_global_builder(old_builder)