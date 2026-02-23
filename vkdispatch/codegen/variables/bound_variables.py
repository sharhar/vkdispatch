from .variables import ShaderVariable
import vkdispatch.base.dtype as dtypes

from ..functions import type_casting
from ..global_builder import get_codegen_backend

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

        backend = get_codegen_backend()
        backend.mark_texture_sample_dimension(self.dimensions)
        
        sample_coord_string = ""

        if self.dimensions == 1:
            sample_coord_string = f"((({coord.resolve()}) + 0.5) / {backend.texture_size_expr(self.resolve(), 0, self.dimensions)})"
        elif self.dimensions == 2:
            coord_expr = backend.constructor(
                dtypes.vec2,
                [
                    backend.component_access_expr(coord.resolve(), "x", coord.var_type),
                    backend.component_access_expr(coord.resolve(), "y", coord.var_type),
                ]
            )
            tex_size_expr = backend.constructor(
                dtypes.vec2,
                [backend.texture_size_expr(self.resolve(), 0, self.dimensions)]
            )
            sample_coord_string = f"(({coord_expr} + 0.5) / {tex_size_expr})"
        elif self.dimensions == 3:
            coord_expr = backend.constructor(
                dtypes.vec3,
                [
                    backend.component_access_expr(coord.resolve(), "x", coord.var_type),
                    backend.component_access_expr(coord.resolve(), "y", coord.var_type),
                    backend.component_access_expr(coord.resolve(), "z", coord.var_type),
                ]
            )
            tex_size_expr = backend.constructor(
                dtypes.vec3,
                [backend.texture_size_expr(self.resolve(), 0, self.dimensions)]
            )
            sample_coord_string = f"(({coord_expr} + 0.5) / {tex_size_expr})"
        else:
            raise ValueError("Unsupported number of dimensions!")

        if lod is None:
            return type_casting.str_to_dtype(
                 dtypes.vec4,
                 backend.sample_texture_expr(self.resolve(), sample_coord_string),
                 [self],
                 lexical_unit=True)
        
        return type_casting.str_to_dtype(
                 dtypes.vec4,
                 backend.sample_texture_expr(self.resolve(), sample_coord_string, lod.resolve()),
                 [self, lod],
                 lexical_unit=True)
        
