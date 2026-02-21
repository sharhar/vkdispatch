import vkdispatch as vd
import vkdispatch.codegen as vc

import inspect
from typing import Callable, TypeVar

from .stage import mapped_io_index, ReduceOp
from .reduce_function import ReduceFunction

import sys

RetType = TypeVar('RetType')
RetType2 = TypeVar('RetType2')

if sys.version_info >= (3, 10):
    from typing import ParamSpec
    P = ParamSpec('P')
else:
    P = ...  # Placeholder for older Python versions


def reduce(identity, axes=None, group_size=None, mapping_function: vd.MappingFunction = None):
    def decorator(func: Callable[..., RetType]) -> Callable[[vd.Buffer[RetType]], vd.Buffer[RetType]]:
        used_mapping_function = mapping_function
        
        func_signature = inspect.signature(func)

        if func_signature.return_annotation == inspect.Parameter.empty:
            raise ValueError("Return type must be annotated")
        
        if used_mapping_function is None:
            used_mapping_function = vd.map(
                func = lambda buffer: buffer[mapped_io_index()],
                return_type=func_signature.return_annotation,
                input_types=[vc.Buffer[func_signature.return_annotation]])
        else:
            assert used_mapping_function.return_type == func_signature.return_annotation, "Mapping function return type must match the return type of the reduction function"

        return ReduceFunction(
            reduction=ReduceOp(
                name=func.__name__,
                reduction=func,
                identity=identity
            ),

            group_size=group_size,
            axes=axes,
            mapping_function=used_mapping_function
        )
    
    return decorator

def map_reduce(reduction: ReduceOp, axes=None, group_size=None):
    def decorator_callback(func: Callable[P, RetType2]) -> Callable[P, vd.Buffer[RetType2]]:
        mapping_func = vd.map(func)

        return ReduceFunction(
           reduction=reduction,
            group_size=group_size,
            axes=axes,
            mapping_function=mapping_func
        )
    
    return decorator_callback