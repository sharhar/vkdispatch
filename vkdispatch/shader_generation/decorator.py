import vkdispatch as vd
import vkdispatch.codegen as vc

import functools
import typing
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

def shader(exec_size=None, local_size=None, workgroups=None, annotations: tuple = None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def decorator(func: F) -> F:
        shader_name = f"{func.__module__}.{func.__name__}"

        builder = vc.ShaderBuilder()
        signature = vd.ShaderSignature()

        old_builder = vc.set_global_builder(builder)
        func(*signature.make_for_decorator(builder, func, annotations))
        vc.set_global_builder(old_builder)

        shader_obj = vd.ShaderObject(
            shader_name, 
            builder.build(shader_name), 
            signature,
            local_size=local_size,
            workgroups=workgroups,
            exec_count=exec_size
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return shader_obj

        wrapper.__signature__ = typing.signature(func)
        return typing.cast(F, wrapper)
    
    return decorator