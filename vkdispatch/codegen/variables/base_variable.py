import vkdispatch.base.dtype as dtypes

from ..global_codegen_callbacks import new_name

from typing import List, Optional

class BaseVariable:
    var_type: dtypes.dtype
    name: str
    raw_name: str
    can_index: bool = False
    use_child_type: bool = True
    lexical_unit: bool = False
    settable: bool = False
    parents: List["BaseVariable"]

    def __init__(self,
                 var_type: dtypes.dtype, 
                 name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 parents: List["BaseVariable"] = None
        ) -> None:
        self.var_type = var_type
        self.lexical_unit = lexical_unit

        self.name = name if name is not None else new_name()
        self.raw_name = raw_name if raw_name is not None else self.name

        self.settable = settable

        if parents is None:
            parents = []

        self.parents = []

        for parent_var in parents:
            if isinstance(parent_var, BaseVariable):
                self.parents.append(parent_var)

        if dtypes.is_complex(self.var_type):
            self.real = self.new_var(self.var_type.child_type, f"{self.resolve()}.x", [self], lexical_unit=True, settable=settable)
            self.imag = self.new_var(self.var_type.child_type, f"{self.resolve()}.y", [self], lexical_unit=True, settable=settable)
            self.x = self.real
            self.y = self.imag

            self._register_shape()
        
        if dtypes.is_vector(self.var_type):
            self.x = self.new_var(self.var_type.child_type, f"{self.resolve()}.x", [self], lexical_unit=True, settable=settable)
            
            if self.var_type.child_count >= 2:
                self.y = self.new_var(self.var_type.child_type, f"{self.resolve()}.y", [self], lexical_unit=True, settable=settable)

            if self.var_type.child_count >= 3:
                self.z = self.new_var(self.var_type.child_type, f"{self.resolve()}.z", [self], lexical_unit=True, settable=settable)

            if self.var_type.child_count == 4:
                self.w = self.new_var(self.var_type.child_type, f"{self.resolve()}.w", [self], lexical_unit=True, settable=settable)
            
            self._register_shape()
        
        if dtypes.is_matrix(self.var_type):
            self._register_shape()

        self._initilized = True
    
    def _register_shape(self, shape_var: "BaseVariable" = None, shape_name: str = None, use_child_type: bool = True):
        self.shape = shape_var
        self.shape_name = shape_name
        self.can_index = True
        self.use_child_type = use_child_type

    def is_setable(self):
        return self.settable

    def resolve(self) -> str:
        if self.lexical_unit:
            return self.name

        return f"({self.name})"
    
    def read_callback(self):
        for parent in self.parents:
            parent.read_callback()

    def write_callback(self):
        for parent in self.parents:
            parent.write_callback()

    def cast_to(self, var_type: dtypes.dtype) -> "BaseVariable":
        return self.new_var(var_type, f"{var_type.glsl_type}({self.name})", [self], lexical_unit=True)

    def new_var(self,
                var_type: dtypes.dtype,
                name: str,
                parents: List["BaseVariable"],
                lexical_unit: bool = False,
                settable: bool = False):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def new_scaled_var(self,
                        var_type: dtypes.dtype,
                        name: str,
                        scale: int = 1,
                        offset: int = 0,
                        parents: List["BaseVariable"] = None):
        raise NotImplementedError("Subclasses should implement this method.")