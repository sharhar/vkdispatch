from setuptools import setup, Extension
import numpy
import os
import platform

system = platform.system()

platform_library_dirs = ['./deps/MoltenVK/MoltenVK/MoltenVK/static/MoltenVK.xcframework/macos-arm64_x86_64/'] if system == 'Darwin' else []
platform_link_libraries = [] if system == 'Windows' else ['dl', 'pthread']

if system == 'Darwin':
    platform_link_libraries.append('MoltenVK')

platform_extra_link_args = ['-g'] if not system == 'Darwin' else ['-g', '-framework', 'Metal', '-framework', 'AVFoundation', '-framework', 'AppKit']

platform_extra_compile_args = ['/W3', '/GL', '/DNDEBUG', '/MD', '/EHsc', '/std:c++17'] if system == 'Windows' else ['-g', '-std=c++17']

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

proj_root = os.path.abspath(os.path.dirname(__file__))

include_directories = [
    numpy_include,
    proj_root + "/include_ext",
    #proj_root + "/deps/VKL/include",
    proj_root + "/deps/Vulkan-Headers/include",
    proj_root + "/deps/Vulkan-Utility-Libraries/include",
    proj_root + "/deps/VMA/include",
    proj_root + "/deps/volk",
    proj_root + "/deps/glslang",
    proj_root + "/deps/glslang/glslang/Include",
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
    'descriptor_set.cpp',
    'stream.cpp',
    'VMAImpl.cpp'
])

"""
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
    'VMAImpl.cpp'
])

if not system == 'Darwin':
    sources.append("deps/VKL/src/VolkImpl.cpp")

    """

append_to_sources('deps/glslang/glslang/', [
    "CInterface/glslang_c_interface.cpp",
    "GenericCodeGen/CodeGen.cpp",
    "GenericCodeGen/Link.cpp",
    "MachineIndependent/glslang_tab.cpp",
    "MachineIndependent/attribute.cpp",
    "MachineIndependent/Constant.cpp",
    "MachineIndependent/iomapper.cpp",
    "MachineIndependent/InfoSink.cpp",
    "MachineIndependent/Initialize.cpp",
    "MachineIndependent/IntermTraverse.cpp",
    "MachineIndependent/Intermediate.cpp",
    "MachineIndependent/ParseContextBase.cpp",
    "MachineIndependent/ParseHelper.cpp",
    "MachineIndependent/PoolAlloc.cpp",
    "MachineIndependent/RemoveTree.cpp",
    "MachineIndependent/Scan.cpp",
    "MachineIndependent/ShaderLang.cpp",
    "MachineIndependent/SpirvIntrinsics.cpp",
    "MachineIndependent/SymbolTable.cpp",
    "MachineIndependent/Versions.cpp",
    "MachineIndependent/intermOut.cpp",
    "MachineIndependent/limits.cpp",
    "MachineIndependent/linkValidate.cpp",
    "MachineIndependent/parseConst.cpp",
    "MachineIndependent/reflection.cpp",
    "MachineIndependent/preprocessor/Pp.cpp",
    "MachineIndependent/preprocessor/PpAtom.cpp",
    "MachineIndependent/preprocessor/PpContext.cpp",
    "MachineIndependent/preprocessor/PpScanner.cpp",
    "MachineIndependent/preprocessor/PpTokens.cpp",
    "MachineIndependent/propagateNoContraction.cpp",
    "ResourceLimits/ResourceLimits.cpp",
    "ResourceLimits/resource_limits_c.cpp"
])

append_to_sources('deps/glslang/SPIRV/', [
    "GlslangToSpv.cpp",
    "InReadableOrder.cpp",
    "Logger.cpp",
    "SpvBuilder.cpp",
    "SpvPostProcess.cpp",
    "doc.cpp",
    "SpvTools.cpp",
    "disassemble.cpp",
    "CInterface/spirv_c_interface.cpp"
])

setup(
    name='vkdispatch',
    packages=['vkdispatch'],
    ext_modules=[
        Extension('vkdispatch_native',
                  sources=sources,
                  language='c++',
                  define_macros=[('VKDISPATCH_USE_MVK', 1)] if system == 'Darwin' else [],
                  library_dirs=platform_library_dirs,
                  libraries=platform_link_libraries,
                  extra_compile_args=platform_extra_compile_args,
                  extra_link_args=platform_extra_link_args,
                  include_dirs=include_directories
        )
    ],
    version='0.0.6',
    zip_safe=False
)