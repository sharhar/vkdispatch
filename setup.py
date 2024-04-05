from setuptools import setup, Extension
import numpy
import os

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

proj_root = os.path.abspath(os.path.dirname(__file__))

vulkan_root = os.environ.get('VULKAN_SDK')

if vulkan_root is None:
    raise ValueError('VULKAN_SDK environment variable not set')

include_directories = [
    numpy_include,
    vulkan_root + '/include',
    vulkan_root + '/include/glslang/Include',
    proj_root + "/deps/VKL/include",
    proj_root + "/deps/VkFFT/vkFFT"
]

sources = [
    'vkdispatch_native/wrapper.pyx',
    'vkdispatch_native/init.cpp',
    'vkdispatch_native/device_context.cpp',

    'deps/VKL/src/VKLBuffer.cpp',
    'deps/VKL/src/VKLCommandBuffer.cpp',
    'deps/VKL/src/VKLDescriptorSet.cpp',
    'deps/VKL/src/VKLDevice.cpp',
    'deps/VKL/src/VKLFramebuffer.cpp',
    'deps/VKL/src/VKLImage.cpp',
    'deps/VKL/src/VKLImageView.cpp',
    'deps/VKL/src/VKLInstance.cpp',
    'deps/VKL/src/VKLPhysicalDevice.cpp',
    'deps/VKL/src/VKLPipeline.cpp',
    'deps/VKL/src/VKLPipelineLayout.cpp',
    'deps/VKL/src/VKLQueue.cpp',
    'deps/VKL/src/VKLRenderPass.cpp',
    'deps/VKL/src/VKLSurface.cpp',
    'deps/VKL/src/VKLStaticAllocator.cpp',
    'deps/VKL/src/VKLSwapChain.cpp'
]

vulkan_lib_dir = vulkan_root + '/lib'

unix_libs = ['dl', 'pthread', 'vulkan']
windows_libs = ['vulkan-1']

platform_libs = unix_libs if os.name == 'posix' else windows_libs

compile_libs = platform_libs + ['glslang', 'SPIRV', 'MachineIndependent', 'GenericCodeGen']
for file in os.listdir(vulkan_lib_dir):
    if "OGLCompiler" in file:
        compile_libs.append('OGLCompiler')
        break

compile_libs.extend(['OSDependent', 'SPIRV-Tools', 'SPIRV-Tools-opt',
                             'SPIRV-Tools-link', 'SPIRV-Tools-reduce',
                             'glslang-default-resource-limits'])


setup(
    name='vkdispatch',
    packages=['vkdispatch'],
    ext_modules=[
        Extension('vkdispatch_native',
                  sources=sources,
                  language='c++',
                  library_dirs=[vulkan_lib_dir],
                  libraries=compile_libs,
                  extra_compile_args=['-g', '-std=c++11'],
                  extra_link_args=['-g', f'-Wl,-rpath,{vulkan_root}/lib'],
                  include_dirs=include_directories
        )
    ],
    zip_safe=False
)