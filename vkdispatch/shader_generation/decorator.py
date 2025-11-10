import vkdispatch as vd
import vkdispatch.codegen as vc

import inspect
from typing import Callable, TypeVar

import sys

if sys.version_info >= (3, 10):
    from typing import ParamSpec
    P = ParamSpec('P')
else:
    P = ...  # Placeholder for older Python versions

def shader(
        exec_size=None,
        local_size=None,
        workgroups=None,    
        flags: vc.ShaderFlags = vc.ShaderFlags.NONE):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def decorator_callback(func: Callable[P, None]) -> Callable[P, None]:
        return vd.ShaderFunction(
            func,
            local_size=local_size,
            workgroups=workgroups,
            exec_count=exec_size,
            flags=flags
        )
    
    return decorator_callback
