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
    vulkan_root + '/include/utility',
    vulkan_root + '/include/glslang/Include',
    proj_root + "/deps/VKL/include",
    proj_root + "/deps/VKL/deps/VMA/include",
    proj_root + "/deps/VKL/deps/volk",
    proj_root + "/deps/VkFFT/vkFFT"
]

sources = []

def append_to_sources(prefix, source_list):
    global sources

    for source in source_list:
        sources.append(prefix + source)

append_to_sources('vkdispatch_native/', [
    'wrapper.pyx',
    'init.cpp',
    'context.cpp',
    'buffer.cpp',
    'image.cpp',
    'command_list.cpp',
    'stage_transfer.cpp',
    'stage_fft.cpp',
    'stage_compute.cpp',
    'descriptor_set.cpp'
])

append_to_sources('deps/VKL/src/', [
    'VKLBuffer.cpp',
    'VKLCommandBuffer.cpp',
    'VKLDescriptorSet.cpp',
    'VKLDevice.cpp',
    'VKLFramebuffer.cpp',
    'VKLImage.cpp',
    'VKLImageView.cpp',
    'VKLInstance.cpp',
    'VKLPhysicalDevice.cpp',
    'VKLPipeline.cpp',
    'VKLPipelineLayout.cpp',
    'VKLQueue.cpp',
    'VKLRenderPass.cpp',
    'VKLSurface.cpp',
    'VKLStaticAllocator.cpp',
    'VKLSwapChain.cpp',
    'VMAImpl.cpp',
    'VolkImpl.cpp'
])

vulkan_lib_dir = vulkan_root + '/lib'

unix_libs = ['dl', 'pthread']
windows_libs = []

platform_libs = unix_libs if os.name == 'posix' else windows_libs

#compile_libs = platform_libs + ['SPIRV', 'SPIRV-Tools-opt', 'SPIRV-Tools-link', 'SPIRV-Tools-reduce', 'SPIRV-Tools']

#"""

compile_libs = platform_libs + ['glslang', 'SPIRV', 'MachineIndependent', 'GenericCodeGen']

for file in os.listdir(vulkan_lib_dir):
    if "OGLCompiler" in file:
        compile_libs.append('OGLCompiler')
        break

compile_libs.extend(['OSDependent',  'SPIRV-Tools-opt',
                             'SPIRV-Tools-link', 'SPIRV-Tools-reduce',
                             'SPIRV-Tools', 'glslang-default-resource-limits'])

#"""

setup(
    name='vkdispatch',
    packages=['vkdispatch'],
    ext_modules=[
        Extension('vkdispatch_native',
                  sources=sources,
                  language='c++',
                  library_dirs=[vulkan_lib_dir],
                  libraries=compile_libs,
                  extra_compile_args=['-g', '-std=c++17'],
                  extra_link_args=['-g', f'-Wl'],
                  include_dirs=include_directories
        )
    ],
    zip_safe=False
)