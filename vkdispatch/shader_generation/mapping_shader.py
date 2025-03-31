import vkdispatch as vd
import dataclasses

import inspect
from typing import List, Callable

@dataclasses.dataclass
class MappingFunction:
    buffer_types: List[vd.dtype]
    return_type: vd.dtype
    mapping_function: Callable

def map(func: Callable):
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

    return MappingFunction(
        buffer_types=input_types,
        return_type=func_signature.return_annotation,
        mapping_function=func
    )