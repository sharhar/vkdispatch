import vkdispatch as vd
import vkdispatch.codegen as vc

import functools
import inspect
import typing
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

def shader(exec_size=None, local_size=None, workgroups=None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def decorator(func: F) -> F:
        shader_name = f"{func.__module__}.{func.__name__}"

        builder = vc.ShaderBuilder()
        old_builder = vc.set_global_builder(builder)
        
        signature = vd.ShaderSignature.from_inspectable_function(builder, func)
        
        func(*signature.get_variables())
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
            return shader_obj.__call__(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return typing.cast(F, wrapper)
        #return functools.update_wrapper(wrapper, func)
    
    return decorator