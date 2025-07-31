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

def shader(
        exec_size=None,
        local_size=None,
        workgroups=None,    
        enable_subgroup_ops: bool = True,
        enable_atomic_float_ops: bool = True,
        enable_printf: bool = True,
        enable_exec_bounds: bool = True):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def decorator(func: Callable[P, None]) -> Callable[P, None]:
        shader_name = f"{func.__module__}.{func.__name__}"

        with vc.builder_context(
            enable_subgroup_ops=enable_subgroup_ops,
            enable_atomic_float_ops=enable_atomic_float_ops,
            enable_printf=enable_printf,
            enable_exec_bounds=enable_exec_bounds
        ) as builder:
            signature = vd.ShaderSignature.from_inspectable_function(builder, func)
            
            func(*signature.get_variables())

            return vd.ShaderObject(
                builder.build(shader_name), 
                signature,
                local_size=local_size,
                workgroups=workgroups,
                exec_count=exec_size
            )
    
    return decorator

def reduce(identity, axes=None, group_size=None, mapping_function: vd.MappingFunction = None):
    def decorator(func: Callable[..., RetType]) -> Callable[[vd.Buffer[RetType]], vd.Buffer[RetType]]:
        used_mapping_function = mapping_function
        
        func_signature = inspect.signature(func)

        if func_signature.return_annotation == inspect.Parameter.empty:
            raise ValueError("Return type must be annotated")
        
        if used_mapping_function is None:
            used_mapping_function = vd.map(
                func = lambda buffer: buffer[vc.mapping_index()],
                return_type=func_signature.return_annotation,
                input_types=[vc.Buffer[func_signature.return_annotation]])
        else:
            assert used_mapping_function.return_type == func_signature.return_annotation, "Mapping function return type must match the return type of the reduction function"

        return vd.ReductionObject(
            reduction=vd.ReductionOperation(
                name=func.__name__,
                reduction=func,
                identity=identity
            ),

            group_size=group_size,
            axes=axes,
            mapping_function=used_mapping_function
        )
    
    return decorator

def map_reduce(reduction: vd.ReductionOperation, axes=None, group_size=None):
    def decorator(func: Callable[P2, RetType2]) -> Callable[P2, vd.Buffer[RetType2]]:
        mapping_func = vd.map(func)

        return vd.ReductionObject(
           reduction=reduction,
            group_size=group_size,
            axes=axes,
            mapping_function=mapping_func
        )
    
    return decorator