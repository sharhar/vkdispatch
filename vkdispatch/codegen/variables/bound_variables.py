from .variables import ShaderVariable
import vkdispatch.base.dtype as dtypes

from ..functions import type_casting

from typing import Callable, Optional

class BoundVariable(ShaderVariable):
    binding: int = -1

    def __init__(self,
                 var_type: dtypes.dtype,
                 binding: int,
                 name: str,
            ) -> None:
            super().__init__(var_type, name, lexical_unit=True)

            self.binding = binding

class BufferVariable(BoundVariable):
    read_lambda: Callable[[], None]
    write_lambda: Callable[[], None]

    def __init__(self,
                 var_type: dtypes.dtype,
                 binding: int,
                 name: str,
                 shape_var: "ShaderVariable" = None,
                 shape_name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 read_lambda: Callable[[], None] = None,
                 write_lambda: Callable[[], None] = None,
            ) -> None:
            super().__init__(var_type, binding, name)

            self.name = name if name is not None else self.name
            self.raw_name = raw_name if raw_name is not None else self.raw_name
            self.settable = True

            self.read_lambda = read_lambda
            self.write_lambda = write_lambda

            self.shape = shape_var
            self.shape_name = shape_name
            self.can_index = True
            self.use_child_type = False

            #self._register_shape(shape_var=shape_var, shape_name=shape_name, use_child_type=False)

    def read_callback(self):
        self.read_lambda()

    def write_callback(self):
        self.write_lambda()

class ImageVariable(BoundVariable):
    dimensions: int = 0
    read_lambda: Callable[[], None]
    write_lambda: Callable[[], None]

    def __init__(self,
                 var_type: dtypes.dtype,
                 binding: int,
                 dimensions: int,
                 name: str,
                 read_lambda: Callable[[], None] = None,
                 write_lambda: Callable[[], None] = None,
            ) -> None:
            super().__init__(var_type, binding, name)

            self.read_lambda = read_lambda
            self.write_lambda = write_lambda
            self.dimensions = dimensions

    def read_callback(self):
        self.read_lambda()

    def write_callback(self):
        self.write_lambda() 

    def sample(self, coord: "ShaderVariable", lod: "ShaderVariable" = None) -> "ShaderVariable":
        if self.dimensions == 0:
            raise ValueError("Cannot sample a texture with dimension 0!")
        
        sample_coord_string = ""

        if self.dimensions == 1:
            sample_coord_string = f"((({coord.resolve()}) + 0.5) / textureSize({self.resolve()}, 0))"        
        elif self.dimensions == 2:
            sample_coord_string = f"((vec2({coord.resolve()}.xy) + 0.5) / vec2(textureSize({self.resolve()}, 0)))"
        elif self.dimensions == 3:
            sample_coord_string = f"((vec3({coord.resolve()}.xyz) + 0.5) / vec3(textureSize({self.resolve()}, 0)))"
        else:
            raise ValueError("Unsupported number of dimensions!")

        if lod is None:
            return type_casting.str_to_dtype(
                 dtypes.vec4,
                 f"texture({self.resolve()}, {sample_coord_string})",
                 [self],
                 lexical_unit=True)
        
        return type_casting.str_to_dtype(
                 dtypes.vec4,
                 f"texture({self.resolve()}, {sample_coord_string}, {lod.resolve()})",
                 [self, lod],
                 lexical_unit=True)
        