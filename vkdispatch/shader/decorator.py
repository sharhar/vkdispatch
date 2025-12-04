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
    """
    A decorator that transforms a Python function into a GPU Compute Shader.

    The decorated function will undergo runtime inspection. Operations performed on 
    ``vkdispatch`` types (buffers, registers) within the function are recorded and 
    transpiled to GLSL.

    :param exec_size: The total number of threads to dispatch (x, y, z). The number of 
                      workgroups is calculated automatically based on ``local_size``.
                      Mutually exclusive with ``workgroups``.
    :type exec_size: Union[int, Tuple[int, ...], Callable]
    :param local_size: The number of threads per workgroup (x, y, z). Defaults to 
                       the device's maximum supported workgroup size.
    :type local_size: Union[int, Tuple[int, ...]]
    :param workgroups: The explicit number of workgroups to dispatch (x, y, z). 
                       Mutually exclusive with ``exec_size``.
    :type workgroups: Union[int, Tuple[int, ...], Callable]
    :param flags: Compilation flags (e.g., ``vc.ShaderFlags.NO_EXEC_BOUNDS``).
    :type flags: vkdispatch.codegen.ShaderFlags
    :return: A ``ShaderFunction`` wrapper that can be called to execute the kernel.
    :raises ValueError: If both ``exec_size`` and ``workgroups`` are provided.
    """
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
