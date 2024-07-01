import vkdispatch as vd
import numpy as np

import inspect

from typing import Callable

def kernel(func):
    signature = inspect.signature(func)

    for param in signature.parameters.values():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError("All parameters must be annotated")

        print(param.name)
        print(param.annotation)

    print(signature.parameters)

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    return wrapper