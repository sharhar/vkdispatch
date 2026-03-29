import vkdispatch.base.dtype as dtypes

from ..variables.variables import ShaderVariable

from . import type_casting

from . import utils

from typing import List, Union, Tuple

def sanitize_input(value: Union[ShaderVariable, Tuple[int, ...]]) -> Tuple[List[Union[ShaderVariable, int]], bool]:
    axes_lengths = []

    if isinstance(value, ShaderVariable):
        if not (dtypes.is_vector(value.var_type) or dtypes.is_scalar(value.var_type)):
            raise ValueError(f"Value is of type '{value.var_type.name}', but it must be a vector or integer!")
        
        if not dtypes.is_integer_dtype(value.var_type):
            raise ValueError(f"Value is of type '{value.var_type.name}', but it must be of integer type!")

        if dtypes.is_scalar(value.var_type):
            axes_lengths.append(value)
            return axes_lengths
        
        elem_count = value.var_type.child_count
        if elem_count < 2 or elem_count > 4:
            raise ValueError(f"Value is of type '{value.var_type.name}', but it must have 2, 3 or 4 components!")

        # Since buffer shapes store total elem count in the 4th component, we ignore it here.
        if elem_count == 4:
            elem_count = 3

        for i in range(elem_count):
            axes_lengths.append(value[i])
    else:
        if utils.check_is_int(value):
            return [value]

        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Value must be a ShaderVariable or a list/tuple of integers, but got {type(value)}!")

        elem_count = len(value)
        
        if elem_count < 1 or elem_count > 3:
            raise ValueError(f"Value has {elem_count} elements, but it must have 1, 2, or 3 elements!")

        for i in range(elem_count):
            if not utils.check_is_int(value[i]):
                raise ValueError(f"When value is a list/tuple, all its elements must be integers, but element {i} is of type '{type(value[i])}'!")

            axes_lengths.append(value[i])

    return axes_lengths

def ravel_index(index: Union[ShaderVariable, int], shape: Union[ShaderVariable, Tuple[int, ...]]):
    sanitized_shape = sanitize_input(shape)
    sanitized_index = sanitize_input(index)

    if len(sanitized_index) != 1:
        raise ValueError(f"Index must be a single integer value, not '{index}'!")
    
    if len(sanitized_shape) != 2 and len(sanitized_shape) != 3:
        raise ValueError(f"Shape must have 2 or 3 elements, not '{shape}'!")

    if len(sanitized_shape) == 2:
        x = sanitized_index[0] // sanitized_shape[1]
        y = sanitized_index[0] % sanitized_shape[1]

        return type_casting.to_uvec2(x, y)
    elif len(sanitized_shape) == 3:
        x = sanitized_index[0] // (sanitized_shape[1] * sanitized_shape[2])
        y = (sanitized_index[0] // sanitized_shape[2]) % sanitized_shape[1]
        z = sanitized_index[0] % sanitized_shape[2]

        return type_casting.to_uvec3(x, y, z)
    else:
        raise RuntimeError("Ravel index only supports shapes with 2 or 3 elements!")

def unravel_index(index: Union[ShaderVariable, Tuple[int, ...]], shape: Union[ShaderVariable, Tuple[int, ...]]):
    sanitized_shape = sanitize_input(shape)
    sanitized_index = sanitize_input(index)

    if len(sanitized_index) > len(sanitized_shape):
        raise ValueError(f"Index ({index}) must have the same number of elements as shape ({sanitized_shape})!")

    if len(sanitized_index) == 1:
        return index

    if len(sanitized_index) == 2:
        return sanitized_index[0] * sanitized_shape[1] + sanitized_index[1]

    elif len(sanitized_index) == 3:
        return sanitized_index[0] * (sanitized_shape[1] * sanitized_shape[2]) + sanitized_index[1] * sanitized_shape[2] + sanitized_index[2]
    else:
        raise RuntimeError("Ravel index only supports shapes with 2 or 3 elements!")