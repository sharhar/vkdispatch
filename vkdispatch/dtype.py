from enum import Enum

import numpy as np

import vkdispatch as vd


class dtype_structure(Enum):  # TODO: Adhere to python class naming conventions
    DATA_STRUCTURE_SCALAR = (1,)
    DATA_STRUCTURE_VECTOR = (2,)
    DATA_STRUCTURE_MATRIX = (3,)
    DATA_STRUCTURE_BUFFER = (4,)


class dtype:
    def __init__(
        self,
        name: str,
        item_size: int,
        glsl_type: str,
        structure: dtype_structure,
        child_count: int,
        format_str: str,
        parent: "dtype" = None,
        is_complex: bool = False,
    ) -> None:
        self.name = name
        self.glsl_type = glsl_type
        self.item_size = item_size
        self.alignment_size = item_size

        if structure == dtype_structure.DATA_STRUCTURE_BUFFER:
            self.alignment_size = parent.alignment_size

        self.structure = structure
        self.format_str = format_str
        self.parent = self if parent is None else parent
        self.is_complex = is_complex if self.structure == dtype_structure.DATA_STRUCTURE_SCALAR else parent.is_complex
        self.scalar = (
            self
            if self.structure == dtype_structure.DATA_STRUCTURE_SCALAR
            else self.parent.scalar
        )

        self.child_count = child_count
        self.total_count = (
            1 if parent is None else child_count * self.parent.total_count
        )
        self.scalar_count = (
            1
            if self.structure == dtype_structure.DATA_STRUCTURE_SCALAR
            else child_count * self.parent.scalar_count
        )
        self._true_shape = (
            () if parent is None else (self.child_count, *self.parent._true_shape)
        )
        self.shape = (1,) if parent is None else self._true_shape

        self._true_numpy_shape = (
            ()
            if self.structure == dtype_structure.DATA_STRUCTURE_SCALAR
            else (self.child_count, *self.parent._true_numpy_shape)
        )
        self.numpy_shape = (
            (1,)
            if self.structure == dtype_structure.DATA_STRUCTURE_SCALAR
            else self._true_numpy_shape
        )

    def __repr__(self) -> str:
        return f"<{self.name}, glsl_type={self.glsl_type} scalar_count={self.scalar_count} item_size={self.item_size} bytes>"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, dtype):
            return False

        return (
            self.name == value.name
            and self.item_size == value.item_size
            and self.glsl_type == value.glsl_type
            and self.structure == value.structure
        )

    # NOTE: How is this grabbing an item in a class? wouldn't this create a new object?
    def __getitem__(self, index: int) -> "dtype":  # TODO: Typehinting for class
        index_str = f"[{index}]"

        if index == 0:
            index_str = "[]"

        return dtype(
            f"{self.name}{index_str}",
            self.item_size * index,
            self.glsl_type,
            dtype_structure.DATA_STRUCTURE_BUFFER,
            index,
            self.format_str,
            parent=self
        )


# NOTE: These should be constant values, then imported from some other class? Living at
# base level not a great idea
int32 = dtype("int32", 4, "int", dtype_structure.DATA_STRUCTURE_SCALAR, 1, "%d")
uint32 = dtype("uint32", 4, "uint", dtype_structure.DATA_STRUCTURE_SCALAR, 1, "%u")
float32 = dtype("float32", 4, "float", dtype_structure.DATA_STRUCTURE_SCALAR, 1, "%f")
complex64 = dtype("complex64", 8, "vec2", dtype_structure.DATA_STRUCTURE_SCALAR, 2, "(%f, %f)", float32, True)

vec2 = dtype("vec2", 8, "vec2", dtype_structure.DATA_STRUCTURE_VECTOR, 2, "(%f, %f)", float32)
vec4 = dtype("vec4", 16, "vec4", dtype_structure.DATA_STRUCTURE_VECTOR, 4, "(%f, %f, %f, %f)", float32)

ivec2 = dtype( "ivec2", 8, "ivec2", dtype_structure.DATA_STRUCTURE_VECTOR, 2, "(%d, %d)", int32)
ivec4 = dtype( "ivec4", 16, "ivec4", dtype_structure.DATA_STRUCTURE_VECTOR, 4, "(%d, %d, %d, %d)", int32)

uvec2 = dtype( "uvec2", 8, "uvec2", dtype_structure.DATA_STRUCTURE_VECTOR, 2, "(%u, %u)", uint32)
uvec4 = dtype( "uvec4", 16, "uvec4", dtype_structure.DATA_STRUCTURE_VECTOR, 4, "(%u, %u, %u, %u)", uint32)

mat2 = dtype( "mat2", 16, "mat2", dtype_structure.DATA_STRUCTURE_MATRIX, 2, "\\\\n[%f, %f]\\\\n[%f, %f]\\\\n", vec2)
mat4 = dtype( "mat4", 64, "mat4", dtype_structure.DATA_STRUCTURE_MATRIX, 4, "\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n[%f, %f, %f, %f]\\\\n", vec4)


def from_numpy_dtype(dtype: type) -> dtype:
    if dtype == np.int32:
        return int32
    elif dtype == np.uint32:
        return uint32
    elif dtype == np.float32:
        return float32
    elif dtype == np.complex64:
        return complex64
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")


def to_numpy_dtype(shader_type: dtype) -> type:
    if shader_type == int32:
        return np.int32
    elif shader_type == uint32:
        return np.uint32
    elif shader_type == float32:
        return np.float32
    elif shader_type == complex64:
        return np.complex64
    else:
        raise ValueError(f"Unsupported shader_type ({shader_type})!")
