# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
import sys

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "init.h":
    int add_one(int x)

cpdef add_one_py(x):
    cdef int x_c = int(x)

    return add_one(x_c)