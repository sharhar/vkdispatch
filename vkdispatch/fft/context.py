import vkdispatch as vd
import contextlib
from typing import List, Union, Optional

@contextlib.contextmanager
def fft_context():

    builder = ShaderBuilder(
        enable_atomic_float_ops=enable_atomic_float_ops,
        enable_subgroup_ops=enable_subgroup_ops,
        enable_printf=enable_printf,
        enable_exec_bounds=enable_exec_bounds
    )
    old_builder = set_global_builder(builder)

    try:
        yield builder
    finally:
        set_global_builder(old_builder)