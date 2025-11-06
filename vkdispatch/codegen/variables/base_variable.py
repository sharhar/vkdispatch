import vkdispatch.base.dtype as dtypes
from typing import List, Optional

import numpy as np

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

        if register:
            assert settable, "An unsettable register makes no sense"

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

    def printf_args(self) -> str:
        total_count = np.prod(self.var_type.shape)

        if total_count == 1:
            return self.name

        args_list = []

        for i in range(0, total_count):
            args_list.append(f"{self.name}[{i}]")

        return ",".join(args_list)