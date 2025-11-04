import vkdispatch.base.dtype as dtypes
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
                 name: str,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 register: bool = False,
                 parents: List["BaseVariable"] = None
        ) -> None:
        self.var_type = var_type
        self.lexical_unit = lexical_unit

        assert name is not None, "Variable name cannot be None!"

        self.name = name
        self.raw_name = raw_name if raw_name is not None else self.name

        self.settable = settable
        self.register = register

        if parents is None:
            parents = []

        self.parents = []

        for parent_var in parents:
            if isinstance(parent_var, BaseVariable):
                self.parents.append(parent_var)

    def is_setable(self):
        return self.settable

    def is_register(self):
        return self.register

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

    # def cast_to(self, var_type: dtypes.dtype) -> "BaseVariable":
    #     return self.new_var(var_type, f"{var_type.glsl_type}({self.name})", [self], lexical_unit=True)

    # def new_var(self,
    #             var_type: dtypes.dtype,
    #             name: str,
    #             parents: List["BaseVariable"],
    #             lexical_unit: bool = False,
    #             settable: bool = False):
    #     raise NotImplementedError("Subclasses should implement this method.")
    
    # def new_scaled_var(self,
    #                     var_type: dtypes.dtype,
    #                     name: str,
    #                     scale: int = 1,
    #                     offset: int = 0,
    #                     parents: List["BaseVariable"] = None):
    #     raise NotImplementedError("Subclasses should implement this method.")