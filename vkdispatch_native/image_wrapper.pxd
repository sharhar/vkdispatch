import numpy as np
cimport numpy as cnp
from libcpp cimport bool
import sys

from libc.stdlib cimport malloc, free

cdef extern from "image.h":
    struct Context
    struct Image

    Image* image_create_extern(Context* context, unsigned int width, unsigned int height, unsigned int depth, unsigned int format, unsigned int type);

cpdef inline image_create(unsigned long long context, unsigned int width, unsigned int height, unsigned int depth, unsigned int format, unsigned int type):
    return <unsigned long long>image_create_extern(<Context*>context, width, height, depth, format, type)
