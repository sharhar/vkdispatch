from .variables import ShaderVariable
import vkdispatch.base.dtype as dtypes

from ..functions import type_casting
from ..functions.base_functions import base_utils
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
    scalar_expr: Optional[str]
    codegen_backend: Optional[object]

    def __init__(self,
                 var_type: dtypes.dtype,
                 binding: int,
                 name: str,
                 shape_var: "ShaderVariable" = None,
                 shape_var_factory: Optional[Callable[[], "ShaderVariable"]] = None,
                 shape_name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 scalar_expr: Optional[str] = None,
                 codegen_backend: Optional[object] = None,
                 read_lambda: Callable[[], None] = None,
                 write_lambda: Callable[[], None] = None,
            ) -> None:
            super().__init__(var_type, binding, name)

            self.name = name if name is not None else self.name
            self.raw_name = raw_name if raw_name is not None else self.raw_name
            self.settable = True

            self.read_lambda = read_lambda
            self.write_lambda = write_lambda

            self._shape_var = shape_var
            self._shape_var_factory = shape_var_factory
            self.shape_name = shape_name
            self.scalar_expr = scalar_expr
            self.codegen_backend = codegen_backend
            self.can_index = True
            self.use_child_type = False

    @property
    def shape(self) -> "ShaderVariable":
        if self._shape_var is None:
            if self._shape_var_factory is None:
                raise ValueError("Buffer shape variable factory is not available!")
            
            self._shape_var = self._shape_var_factory()

        return self._shape_var

    def read_callback(self):
        self.read_lambda()

    def write_callback(self):
        self.write_lambda()

    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise TypeError(f"Variable '{self.resolve()}' of type '{self.var_type.name}' cannot be indexed into!")

        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, tuple):
            if len(index) != 1:
                raise ValueError("Only single index is supported, cannot use multi-dimentional indexing!")
            
            index = index[0]

        if base_utils.is_int_number(index):
            backend = self.codegen_backend if self.codegen_backend is not None else get_codegen_backend()
            packed_expr = None
            if self.scalar_expr is not None:
                packed_expr = backend.packed_buffer_read_expr(
                    self.scalar_expr,
                    return_type,
                    str(index),
                )

            if packed_expr is not None:
                return ShaderVariable(
                    return_type,
                    packed_expr,
                    parents=[self],
                    settable=self.settable,
                    lexical_unit=True,
                    buffer_root=self,
                    buffer_index_expr=str(index),
                )

            return ShaderVariable(
                return_type,
                f"{self.resolve()}[{index}]",
                parents=[self],
                settable=self.settable,
                lexical_unit=True,
                buffer_root=self,
                buffer_index_expr=str(index),
            )

        if not isinstance(index, ShaderVariable):
            raise TypeError(f"Index must be a ShaderVariable or int type, not {type(index)}!")
        
        if not dtypes.is_scalar(index.var_type):
            raise TypeError("Indexing variable must be a scalar!")

        if not dtypes.is_integer_dtype(index.var_type):
            raise TypeError("Indexing variable must be an integer type!")

        backend = self.codegen_backend if self.codegen_backend is not None else get_codegen_backend()
        packed_expr = None
        if self.scalar_expr is not None:
            packed_expr = backend.packed_buffer_read_expr(
                self.scalar_expr,
                return_type,
                index.resolve(),
            )

        if packed_expr is not None:
            return ShaderVariable(
                return_type,
                packed_expr,
                parents=[self, index],
                settable=self.settable,
                lexical_unit=True,
                buffer_root=self,
                buffer_index_expr=index.resolve(),
            )

        return ShaderVariable(
            return_type,
            f"{self.resolve()}[{index.resolve()}]",
            parents=[self, index],
            settable=self.settable,
            lexical_unit=True,
            buffer_root=self,
            buffer_index_expr=index.resolve(),
        )

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
        
