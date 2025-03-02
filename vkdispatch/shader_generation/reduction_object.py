import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Callable

class ReductionObject:
    def __init__(self,
                 reduction: vd.ReductionOperation,
                 out_type: vd.dtype, 
                 group_size: int, 
                 map_func: Callable = None):
        pass