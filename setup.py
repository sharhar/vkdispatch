from setuptools import setup, Extension
import numpy
import os

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

proj_root = os.path.abspath(os.path.dirname(__file__))

include_directories = [
    numpy_include,
    proj_root + "/deps/VKL/include",
    proj_root + "/deps/VKL/deps/Vulkan-Headers/include",
    proj_root + "/deps/VKL/deps/Vulkan-Utility-Libraries/include",
    proj_root + "/deps/VKL/deps/VMA/include",
    proj_root + "/deps/VKL/deps/volk",
    proj_root + "/deps/VKL/deps/glslang",
    proj_root + "/deps/VKL/deps/glslang/glslang/Include",
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

append_to_sources('deps/VKL/deps/glslang/glslang/', [
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

append_to_sources('deps/VKL/deps/glslang/SPIRV/', [
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

compile_libs = ['dl', 'pthread'] if os.name == 'posix' else [] 

setup(
    name='vkdispatch',
    packages=['vkdispatch'],
    ext_modules=[
        Extension('vkdispatch_native',
                  sources=sources,
                  language='c++',
                  library_dirs=[],
                  libraries=compile_libs,
                  extra_compile_args=['-g', '-std=c++17'],
                  extra_link_args=['-g'],
                  include_dirs=include_directories
        )
    ],
    zip_safe=False
)