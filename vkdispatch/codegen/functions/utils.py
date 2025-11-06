import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable

from .base_functions.base_utils import *

from ..shader_writer import scope_increment, scope_decrement

def new_var(var_type: dtypes.dtype,
            var_name: Optional[str],
            parents: list,
            lexical_unit: bool = False,
            settable: bool = False,
            register: bool = False) -> ShaderVariable:
    return new_base_var(var_type, var_name, parents, lexical_unit, settable, register)