import vkdispatch.base.dtype as dtypes

from ..utils import check_is_int
from ..builder import ShaderVariable
from ..global_builder import make_var

from typing import List, Union, Tuple

def sanitize_input(value: Union[ShaderVariable, Tuple[int, ...]]) -> Tuple[List[Union[ShaderVariable, int]], bool]:
    axes_lengths = []
    is_static = None

    if isinstance(value, ShaderVariable):
        is_static = False
        assert dtypes.is_vector(value.var_type) or dtypes.is_scalar(value.var_type), f"Value is of type '{value.var_type.name}', but it must be a vector or integer!"
        assert dtypes.is_integer_dtype(value.var_type), f"Value is of type '{value.var_type.name}', but it must be of integer type!"
        
        if dtypes.is_scalar(value.var_type):
            axes_lengths.append(value)
            return axes_lengths, is_static
        
        elem_count = value.var_type.child_count
        assert elem_count >= 2 and elem_count <= 4, f"Value is of type '{value.var_type.name}', but it must have 2, 3 or 4 components!"

        # Since buffer shapes store total elem count in the 4th component, we ignore it here.
        if elem_count == 4:
            elem_count = 3

        for i in range(elem_count):
            axes_lengths.append(value[i])
    else:
        if check_is_int(value):
            return [value], True

        is_static = True
        assert isinstance(value, (list, tuple)), "Value must be a ShaderVariable or a list/tuple of integers!"

        elem_count = len(value)
        assert elem_count >= 1 or elem_count <= 3, f"Value has {elem_count} elements, but it must have 1, 2, or 3 elements!"

        for i in range(elem_count):
            assert check_is_int(value[i]), "When value is a list/tuple, all its elements must be integers!"

            axes_lengths.append(value[i])

    return axes_lengths, is_static

def ravel_index(index: Union[ShaderVariable, int], shape: Union[ShaderVariable, Tuple[int, ...]]):
    sanitized_shape, static_shape = sanitize_input(shape)
    sanitized_index, static_index = sanitize_input(index)

    assert len(sanitized_index) == 1, f"Index must be a single integer value, not '{index}'!"
    assert len(sanitized_shape) == 2 or len(sanitized_shape) == 3, f"Shape must have 2 or 3 elements, not '{shape}'!"

    if len(sanitized_shape) == 2:
        out_type = dtypes.ivec2

        if static_index and static_shape:
            x = sanitized_index[0] // sanitized_shape[1]
            y = sanitized_index[0] % sanitized_shape[1]
        else:
            x = sanitized_index[0] / sanitized_shape[1]
            y = sanitized_index[0] % sanitized_shape[1]

        variable_text = f"uvec2({x}, {y})"

    elif len(sanitized_shape) == 3:
        out_type = dtypes.ivec3

        if static_index and static_shape:
            x = sanitized_index[0] // (sanitized_shape[1] * sanitized_shape[2])
            y = (sanitized_index[0] // sanitized_shape[2]) % sanitized_shape[1]
            z = sanitized_index[0] % sanitized_shape[2]
        else:
            x = sanitized_index[0] / (sanitized_shape[1] * sanitized_shape[2])
            y = (sanitized_index[0] / sanitized_shape[2]) % sanitized_shape[1]
            z = sanitized_index[0] % sanitized_shape[2]

        variable_text = f"uvec3({x}, {y}, {z})"
    else:
        raise RuntimeError("Ravel index only supports shapes with 2 or 3 elements!")

    return make_var(
        out_type,
        variable_text,
        [index, shape],
        lexical_unit=True
    )

def unravel_index(index: Union[ShaderVariable, Tuple[int, ...]], shape: Union[ShaderVariable, Tuple[int, ...]]):
    sanitized_shape, _ = sanitize_input(shape)
    sanitized_index, _ = sanitize_input(index)

    assert len(sanitized_index) <= len(sanitized_shape), f"Index ({index}) must have the same number of elements as shape ({sanitized_shape})!"

    if len(sanitized_index) == 1:
        return index

    if len(sanitized_index) == 2:
        return sanitized_index[0] * sanitized_shape[1] + sanitized_index[1]

    elif len(sanitized_index) == 3:
        return sanitized_index[0] * (sanitized_shape[1] * sanitized_shape[2]) + sanitized_index[1] * sanitized_shape[2] + sanitized_index[2]
    else:
        raise RuntimeError("Ravel index only supports shapes with 2 or 3 elements!")