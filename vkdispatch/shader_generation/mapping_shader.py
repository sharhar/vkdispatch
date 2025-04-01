import vkdispatch as vd
import dataclasses

import uuid

import inspect
from typing import List, Callable, Union

@dataclasses.dataclass(frozen=True)
class MappingFunction:
    buffer_types: List[vd.dtype]
    register_types: List[vd.dtype]
    return_type: vd.dtype
    mapping_function: Callable

    # Unique identifier for each instance
    instance_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, compare=False)
    
    def __hash__(self) -> int:
        # Hash based only on the instance_id
        return hash(self.instance_id)
    
    def __eq__(self, other):
        if not isinstance(other, MappingFunction):
            return False
        # Two instances are equal only if they have the same instance_id
        return self.instance_id == other.instance_id

def map(func: Callable, register_types: List[vd.dtype] = None, return_type: vd.dtype = None, input_types: List[vd.dtype] = None) -> MappingFunction:
    if register_types is None:
        register_types = []

    if return_type is None:
        func_signature = inspect.signature(func)

        if not func_signature.return_annotation == inspect.Parameter.empty:
            return_type = func_signature.return_annotation
    
    if input_types is None:
        input_types = []

        func_signature = inspect.signature(func)

        for param in func_signature.parameters.values():
            my_annotation = param.annotation

            if my_annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(my_annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{my_annotation}' must have a type annotation")

            input_types.append(my_annotation)

    return MappingFunction(
        buffer_types=input_types,
        return_type=return_type,
        mapping_function=func,
        register_types=register_types
    )

def map_registers(register_types: List[vd.dtype]) -> Callable[[Callable], MappingFunction]:
    def decorator(func: Callable):
        return map(func, register_types)
    
    return decorator