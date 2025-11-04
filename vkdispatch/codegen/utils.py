import numpy as np

def check_is_int(variable):
    return isinstance(variable, int) or np.issubdtype(type(variable), np.integer)