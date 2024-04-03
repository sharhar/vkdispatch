from setuptools import setup
from distutils.extension import Extension
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

include_directories = [
    numpy_include,
]

sources = [
    'vkdispatch_native/wrapper.pyx',
    'vkdispatch_native/init.cpp',
]

setup(
    name='vkdispatch',
    packages=['vkdispatch'],
    ext_modules=[
        Extension('vkdispatch_native',
                  sources=sources,
                  language='c++',
                  library_dirs=[], 
                  libraries=[],
                  extra_compile_args=['-g', '-std=c++11'],
                  extra_link_args=['-g'],
                  include_dirs=include_directories
        )
    ],
    zip_safe=False
)