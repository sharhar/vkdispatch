import vkdispatch as vd
import vkdispatch.codegen as vc

from ..codegen.shader_description import ShaderArgumentType
from typing import List, Optional, Any

import contextlib

class ShaderContext:
    builder: vc.ShaderBuilder
    shader_function: vd.ShaderFunction

    def __init__(self, builder: vc.ShaderBuilder):
        self.builder = builder
        self.shader_function = None

    def get_function(self,
                     local_size=None,
                      workgroups=None,
                      exec_count=None,
                      name: Optional[str] = None) -> vd.ShaderFunction:
        if self.shader_function is not None:
            return self.shader_function

        description = self.builder.build("shader" if name is None else name)
        
        # Resource bindings are declared before final shader layout is known.
        # For some shader construction paths (e.g. from_description), signatures are
        # pre-populated and still hold logical bindings assuming a reserved UBO at 0.
        binding_shift = description.resource_binding_base - 1
        if binding_shift != 0:
            binding_access_len = len(description.binding_access)
            needs_remap = False

            for shader_arg in description.shader_arg_infos:
                if (
                    shader_arg.binding is not None
                    and (
                        shader_arg.arg_type == ShaderArgumentType.BUFFER
                        or shader_arg.arg_type == ShaderArgumentType.IMAGE
                    )
                    and shader_arg.binding >= binding_access_len
                ):
                    needs_remap = True
                    break

            if needs_remap:
                for shader_arg in description.shader_arg_infos:
                    if (
                        shader_arg.binding is not None
                        and (
                            shader_arg.arg_type == ShaderArgumentType.BUFFER
                            or shader_arg.arg_type == ShaderArgumentType.IMAGE
                        )
                    ):
                        shader_arg.binding += binding_shift

        self.shader_function = vd.ShaderFunction(
            description,
            local_size=local_size,
            workgroups=workgroups,
            exec_count=exec_count
        )

        return self.shader_function
    
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