import vkdispatch as vd
import vkdispatch.codegen as vc

import inspect
from typing import Callable, TypeVar

import sys

RetType = TypeVar('RetType')
RetType2 = TypeVar('RetType2')

if sys.version_info >= (3, 10):
    from typing import ParamSpec
    P = ParamSpec('P')
    P2 = ParamSpec('P2')
else:
    P = ...  # Placeholder for older Python versions
    P2 = ...  # Placeholder for older Python versions

def shader(exec_size=None, local_size=None, workgroups=None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def decorator(func: Callable[P, None]) -> Callable[P, None]:
        shader_name = f"{func.__module__}.{func.__name__}"

        builder = vc.ShaderBuilder()
        old_builder = vc.set_global_builder(builder)
        
        signature = vd.ShaderSignature.from_inspectable_function(builder, func)
        
        func(*signature.get_variables())
        vc.set_global_builder(old_builder)

        return vd.ShaderObject(
            shader_name, 
            builder.build(shader_name), 
            signature,
            local_size=local_size,
            workgroups=workgroups,
            exec_count=exec_size
        )
    
    return decorator

def reduce(identity, axes=None, group_size=None):
    def decorator(func: Callable[..., RetType]) -> Callable[[vd.Buffer[RetType]], vd.Buffer[RetType]]:
        func_signature = inspect.signature(func)

        if func_signature.return_annotation == inspect.Parameter.empty:
            raise ValueError("Return type must be annotated")

        return vd.ReductionObject(
            reduction=vd.ReductionOperation(
                name=func.__name__,
                reduction=func,
                identity=identity
            ),
            out_type=func_signature.return_annotation,
            group_size=group_size,
            axes=axes
        )
    
    return decorator

def map_reduce(reduction: vd.ReductionOperation, axes=None, group_size=None):
    def decorator(func: Callable[P2, RetType2]) -> Callable[P2, vd.Buffer[RetType2]]:
        func_signature = inspect.signature(func)

        if func_signature.return_annotation == inspect.Parameter.empty:
            raise ValueError("Return type must be annotated")
        
        input_types = []

        for param in func_signature.parameters.values():
            my_annotation = param.annotation

            if my_annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(my_annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{my_annotation}' must have a type annotation")

            input_types.append(my_annotation)

        return vd.ReductionObject(
            reduction=reduction,
            out_type=func_signature.return_annotation,
            group_size=group_size,
            axes=axes,
            input_types=input_types,
            map_func=func
        )
    
    return decorator